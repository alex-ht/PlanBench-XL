from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
import requests
from openai import OpenAI

from env.core.types import LLMResponse, ModelProfile


_CONFIG_ENV_PATH = Path(__file__).resolve().parents[1] / "config" / ".env"

_LOGGER = logging.getLogger("llm_client")

# Statuses worth retrying on the raw chat-completions transports: 429 (rate limit)
# plus the usual transient gateway/server errors.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})

# Pattern used to recover the rate-limit reset instant from gateway 429 bodies,
# e.g. "Limit resets at: 2026-06-14 03:55:49 UTC".
_RESET_AT_PATTERN = re.compile(r"resets? at:?\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2})\s*UTC", re.IGNORECASE)


class _SlidingWindowRateLimiter:
    """Thread-safe rate limiter shared across all query threads.

    Two cooperating constraints keep us safely under a provider's *fixed*-window
    cap (e.g. 100 requests per rolling 60s that resets on the provider's own
    boundary):

    * a sliding window of at most ``max_requests`` per ``window_seconds`` (a hard
      backstop), and
    * a minimum spacing of ``min_interval_seconds`` between consecutive
      acquisitions, which spreads traffic evenly so that no fixed window ever
      sees a burst. Even pacing — rather than allowing a full-size burst the
      instant the sliding window empties — is what actually avoids the 429s that
      a fixed-window provider would otherwise return at window boundaries.

    ``acquire`` blocks until both constraints are satisfied.
    """

    def __init__(self, max_requests: int, window_seconds: float, min_interval_seconds: float = 0.0) -> None:
        self.max_requests = max(int(max_requests), 1)
        self.window_seconds = float(window_seconds)
        self.min_interval_seconds = max(float(min_interval_seconds), 0.0)
        self._timestamps: deque[float] = deque()
        self._last_acquire: float | None = None
        self._condition = threading.Condition()

    def acquire(self) -> None:
        with self._condition:
            while True:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
                    self._timestamps.popleft()

                waits: list[float] = []
                if len(self._timestamps) >= self.max_requests:
                    waits.append(self.window_seconds - (now - self._timestamps[0]))
                if self.min_interval_seconds and self._last_acquire is not None:
                    waits.append(self.min_interval_seconds - (now - self._last_acquire))

                blocking_wait = max((w for w in waits if w > 0), default=0.0)
                if blocking_wait <= 0:
                    self._timestamps.append(now)
                    self._last_acquire = now
                    self._condition.notify_all()
                    return
                self._condition.wait(timeout=max(blocking_wait, 0.01))


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = _strip_optional_quotes(value.strip())
    if not key:
        return None
    return key, value


def _load_config_env() -> dict[str, str]:
    if not _CONFIG_ENV_PATH.exists():
        return {}
    values: dict[str, str] = {}
    for line in _CONFIG_ENV_PATH.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values


# Rate limiters are shared process-wide and keyed by (base_url, max_requests,
# window, min_interval) so that every query thread hitting the same gateway
# draws from one budget.
_RATE_LIMITERS: dict[tuple[str, int, float, float], _SlidingWindowRateLimiter] = {}
_RATE_LIMITERS_LOCK = threading.Lock()


def _get_shared_rate_limiter(key: tuple[str, int, float, float]) -> _SlidingWindowRateLimiter:
    with _RATE_LIMITERS_LOCK:
        limiter = _RATE_LIMITERS.get(key)
        if limiter is None:
            limiter = _SlidingWindowRateLimiter(
                max_requests=key[1],
                window_seconds=key[2],
                min_interval_seconds=key[3],
            )
            _RATE_LIMITERS[key] = limiter
        return limiter


class LLMClient:
    def __init__(self, model: ModelProfile) -> None:
        self.model = model
        self.config_env = _load_config_env()
        self._rate_limiter = self._build_rate_limiter()

    def _build_rate_limiter(self) -> _SlidingWindowRateLimiter | None:
        max_requests = self.model.capabilities.get("rate_limit_max_requests")
        if max_requests is None:
            return None
        try:
            max_requests_int = int(max_requests)
        except (TypeError, ValueError):
            return None
        if max_requests_int <= 0:
            return None
        window_seconds = self.model.capabilities.get("rate_limit_window_seconds", 60)
        try:
            window_seconds_float = float(window_seconds)
        except (TypeError, ValueError):
            window_seconds_float = 60.0
        # Even pacing: derive a default min-interval from the budget so a full
        # window's worth of requests is spread across the whole window rather
        # than fired as a single burst. Can be overridden explicitly.
        min_interval = self.model.capabilities.get("rate_limit_min_interval_seconds")
        if min_interval is None:
            min_interval_float = window_seconds_float / max_requests_int if max_requests_int else 0.0
        else:
            try:
                min_interval_float = max(float(min_interval), 0.0)
            except (TypeError, ValueError):
                min_interval_float = window_seconds_float / max_requests_int if max_requests_int else 0.0
        base_url = self._resolve_base_url(self.model.auth) or self.model.model_id
        return _get_shared_rate_limiter(
            (base_url, max_requests_int, window_seconds_float, min_interval_float)
        )

    def _max_transport_attempts(self) -> int:
        configured = self.model.capabilities.get("chat_completions_max_attempts")
        if configured is not None:
            try:
                attempts = int(configured)
            except (TypeError, ValueError):
                attempts = 0
            if attempts > 0:
                return attempts
        # Fall back to the model's request.max_retries (total attempts = retries + 1).
        return max(int(self.model.request.max_retries), 0) + 1

    def _retry_base_delay(self) -> float:
        configured = self.model.capabilities.get("chat_completions_retry_base_delay")
        try:
            value = float(configured)
        except (TypeError, ValueError):
            return 2.0
        return value if value > 0 else 2.0

    @staticmethod
    def _parse_reset_delay(detail: str) -> float | None:
        if not detail:
            return None
        match = _RESET_AT_PATTERN.search(detail)
        if not match:
            return None
        raw = match.group(1).replace("T", " ")
        try:
            reset_at = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        delay = (reset_at - datetime.now(timezone.utc)).total_seconds()
        if delay <= 0:
            return None
        return delay

    def _call_with_retries(self, label: str, send: Callable[[], requests.Response | httpx.Response]) -> LLMResponse:
        max_attempts = self._max_transport_attempts()
        base_delay = self._retry_base_delay()
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            try:
                response = send()
            except (requests.RequestException, httpx.HTTPError) as exc:
                last_error = RuntimeError(f"transport error: {exc}")
                if attempt >= max_attempts:
                    break
                self._sleep_before_retry(attempt, base_delay, None)
                continue

            status_code = response.status_code
            if status_code < 400 or status_code not in _RETRYABLE_STATUS_CODES or attempt >= max_attempts:
                return self._parse_raw_chat_completions_response(response)

            detail = self._extract_error_detail(response)
            reset_delay = self._parse_reset_delay(detail) if status_code == 429 else None
            _LOGGER.warning(
                "Retryable response (status=%s) for %s on attempt %s/%s; backing off. detail=%s",
                status_code,
                label,
                attempt,
                max_attempts,
                detail[:200],
            )
            last_error = RuntimeError(f"status={status_code} | detail={detail}")
            self._sleep_before_retry(attempt, base_delay, reset_delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"chat completions request to {label} failed without a response.")

    @staticmethod
    def _sleep_before_retry(attempt: int, base_delay: float, reset_delay: float | None) -> None:
        if reset_delay is not None:
            # Honor the gateway's own reset instant, plus small jitter to avoid a thundering herd.
            time.sleep(reset_delay + random.uniform(0.1, 1.0))
            return
        backoff = base_delay * (2 ** (attempt - 1))
        time.sleep(min(backoff, 60.0) + random.uniform(0.0, 1.0))

    @staticmethod
    def _extract_error_detail(response: requests.Response | httpx.Response) -> str:
        detail = (response.text or "").strip()
        try:
            parsed = response.json()
        except Exception:
            return detail
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            return json.dumps(parsed, ensure_ascii=False)
        return detail

    def generate(
        self,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        provider = self.model.provider
        if provider == "openai":
            return self._generate_openai(history, tools=tools)
        if provider == "anthropic":
            raise RuntimeError(
                "Anthropic runtime path is not available yet in this environment. "
                "Please install the anthropic package or use an OpenAI model first."
            )
        if provider == "google":
            raise RuntimeError(
                "Google Gemini runtime path is not available yet in this environment. "
                "Please install google-genai or use an OpenAI model first."
            )
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _build_messages_for_api(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal history to API messages, preserving native tool_calls fields."""
        messages: list[dict[str, Any]] = []
        for item in history:
            role = item.get("role", "user")
            if role == "assistant":
                msg: dict[str, Any] = {"role": "assistant"}
                content = item.get("content")
                if content is not None:
                    msg["content"] = content
                tool_calls = item.get("tool_calls")
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                # Only pass reasoning_content back for native tool_calls turns (per DeepSeek docs for tool calls).
                # For prompted XML mode, we still capture it for logging but don't echo in messages to avoid 400 on some endpoints.
                if tool_calls:
                    reasoning = item.get("reasoning_content")
                    if reasoning:
                        msg["reasoning_content"] = reasoning
                if "content" not in msg and not tool_calls:
                    msg["content"] = ""
                messages.append(msg)
            elif role == "tool":
                msg = {"role": "tool", "content": item.get("content", "")}
                tool_call_id = item.get("tool_call_id")
                if tool_call_id:
                    msg["tool_call_id"] = tool_call_id
                messages.append(msg)
            else:
                messages.append({"role": role, "content": item.get("content", "")})
        return messages

    def _generate_openai(
        self,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        auth = self.model.auth
        api_key = self._resolve_api_key(auth)
        base_url = self._resolve_base_url(auth)
        organization = self._resolve_organization(auth)

        transport = self._raw_chat_completions_transport(base_url)
        if self.model.api_style == "chat_completions" and transport == "requests":
            token_param = self.model.capabilities.get("chat_completion_token_param", "max_tokens")
            messages = self._build_messages_for_api(history)
            return self._generate_openai_chat_completions_via_requests(messages, token_param, tools=tools)
        if self.model.api_style == "chat_completions" and transport == "httpx":
            token_param = self.model.capabilities.get("chat_completion_token_param", "max_tokens")
            messages = self._build_messages_for_api(history)
            return self._generate_openai_chat_completions_via_httpx(messages, token_param, tools=tools)

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": self.model.request.max_retries,
            "timeout": self.model.request.timeout_seconds,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization

        client = OpenAI(**client_kwargs)

        if self.model.api_style == "responses":
            return self._generate_openai_responses(client, history)
        if self.model.api_style == "chat_completions":
            return self._generate_openai_chat_completions(client, history, tools=tools)
        raise ValueError(f"Unsupported OpenAI api_style: {self.model.api_style}")

    def _generate_openai_responses(self, client: OpenAI, history: list[dict[str, Any]]) -> LLMResponse:
        request_kwargs: dict[str, Any] = {
            "model": self.model.model_name,
            "input": [{"role": item["role"], "content": item.get("content", "")} for item in history],
            "max_output_tokens": self.model.request.max_tokens,
        }
        if self.model.request.temperature is not None and not self._responses_model_disallows_temperature():
            request_kwargs["temperature"] = self.model.request.temperature
        reasoning = self._build_openai_reasoning()
        if reasoning is not None:
            request_kwargs["reasoning"] = reasoning

        response = client.responses.create(**request_kwargs)
        # Responses API reasoning is more complex (summary etc.); capture basic if available
        reasoning = getattr(response, "reasoning", None) or getattr(response, "reasoning_content", None)
        reasoning_text = reasoning.get("content", "") if isinstance(reasoning, dict) else None
        return LLMResponse(content=response.output_text, reasoning_content=reasoning_text)

    def _generate_openai_chat_completions(
        self,
        client: OpenAI,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        token_param = self.model.capabilities.get("chat_completion_token_param", "max_tokens")
        messages = self._build_messages_for_api(history)
        base_url = self._resolve_base_url(self.model.auth)
        transport = self._raw_chat_completions_transport(base_url)
        if transport == "requests":
            return self._generate_openai_chat_completions_via_requests(messages, token_param, tools=tools)
        if transport == "httpx":
            return self._generate_openai_chat_completions_via_httpx(messages, token_param, tools=tools)
        request_kwargs: dict[str, Any] = {
            "model": self.model.model_name,
            "messages": messages,
        }
        request_kwargs[token_param] = self.model.request.max_tokens
        chat_supports_temperature = self.model.capabilities.get("chat_completions_supports_temperature", True)
        if self.model.request.temperature is not None and chat_supports_temperature and not self._is_thinking_mode():
            request_kwargs["temperature"] = self.model.request.temperature
        extra_body = self._build_openai_chat_extra_body()
        thinking_extra = self._build_deepseek_thinking_extra()
        if thinking_extra:
            extra_body = extra_body or {}
            extra_body.update(thinking_extra)
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        # Support reasoning_effort for providers that accept it on chat.completions (DeepSeek thinking mode)
        if self.model.request.reasoning_effort:
            request_kwargs["reasoning_effort"] = self.model.request.reasoning_effort

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as e:
            # Surface full server error for debugging (e.g. 400 reasons from thinking mode)
            if hasattr(e, "response") and e.response is not None:
                try:
                    body = e.response.json()
                    raise RuntimeError(f"Chat completions 400/err: {body}") from e
                except Exception:
                    raise RuntimeError(f"Chat completions error: {e.response.text}") from e
            raise
        msg = response.choices[0].message

        # Normalise content: SDK returns str | None; some providers return a list of blocks.
        raw_content = msg.content
        if isinstance(raw_content, list):
            parts = []
            for block in raw_content:
                text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
                if text:
                    parts.append(str(text))
            content: str | None = "\n".join(parts) if parts else None
        else:
            content = raw_content  # str or None

        tool_calls_raw: list[dict[str, Any]] | None = None
        if msg.tool_calls:
            tool_calls_raw = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        # Capture reasoning/thinking traces (DeepSeek reasoning_content, some Qwen/others)
        reasoning_content: str | None = None
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning_content = msg.reasoning_content
        elif isinstance(getattr(msg, "model_extra", None), dict):
            reasoning_content = msg.model_extra.get("reasoning_content") or msg.model_extra.get("thinking")

        if content is None and not tool_calls_raw:
            raise RuntimeError("OpenAI chat.completions returned an empty assistant message.")
        return LLMResponse(content=content, tool_calls=tool_calls_raw, reasoning_content=reasoning_content)

    def _resolve_api_key(self, auth: Any) -> str:
        if auth.api_key:
            return auth.api_key

        if auth.api_key_env:
            env_value = self.config_env.get(auth.api_key_env)
            if env_value:
                return env_value

        base_url = self._resolve_base_url(auth)
        if self._looks_like_local_base_url(base_url):
            return "EMPTY"

        raise RuntimeError(
            f"Missing API key for model {self.model.model_name}. "
            f"Add {auth.api_key_env} to {_CONFIG_ENV_PATH}."
        )

    def _resolve_base_url(self, auth: Any) -> str | None:
        if auth.base_url:
            return auth.base_url
        if auth.base_url_env:
            return self.config_env.get(auth.base_url_env) or None
        return None

    def _resolve_organization(self, auth: Any) -> str | None:
        if auth.organization:
            return auth.organization
        if auth.organization_env:
            return self.config_env.get(auth.organization_env) or None
        return None

    def _looks_like_local_base_url(self, base_url: str | None) -> bool:
        if not base_url:
            return False
        parsed = urlparse(base_url)
        hostname = parsed.hostname
        return hostname in {"localhost", "127.0.0.1", "0.0.0.0"}

    def _build_openai_reasoning(self) -> dict[str, Any] | None:
        if self.model.request.reasoning_effort:
            return {"effort": self.model.request.reasoning_effort}

        # For Responses API models, honor the shared "reasoning: false" capability
        # by requesting the lowest supported reasoning budget.
        if self.model.capabilities.get("reasoning") is False and self._responses_model_supports_reasoning_controls():
            return {"effort": "minimal"}

        return None

    def _build_openai_chat_extra_body(self) -> dict[str, Any] | None:
        raw_chat_template_kwargs = self.model.capabilities.get("chat_template_kwargs")
        if raw_chat_template_kwargs is not None and not isinstance(raw_chat_template_kwargs, dict):
            return None

        if raw_chat_template_kwargs is None:
            return None

        merged_chat_template_kwargs = dict(raw_chat_template_kwargs)
        if not merged_chat_template_kwargs:
            return None

        return {"chat_template_kwargs": merged_chat_template_kwargs}

    def _build_deepseek_thinking_extra(self) -> dict[str, Any] | None:
        """Build extra_body for DeepSeek Thinking Mode.

        WARNING: On some Azure endpoints (e.g. alex-g4), including a top-level
        "thinking" field causes 400 "unrecognized request argument".
        Only enable this block for endpoints that explicitly support it.
        For most DeepSeek-on-Azure, just setting reasoning_effort is sufficient.
        """
        thinking_cfg = self.model.capabilities.get("thinking")
        if isinstance(thinking_cfg, dict) and thinking_cfg.get("type"):
            return {"thinking": dict(thinking_cfg)}
        return None

    def _is_thinking_mode(self) -> bool:
        """Whether this model is using DeepSeek-style thinking mode (which disallows temperature etc)."""
        if self.model.capabilities.get("thinking"):
            return True
        if self.model.request.reasoning_effort:
            return True
        return False

    def _responses_model_disallows_temperature(self) -> bool:
        model_name = self.model.model_name.lower()
        return model_name.startswith("gpt-5") or model_name.startswith("o")

    def _responses_model_supports_reasoning_controls(self) -> bool:
        model_name = self.model.model_name.lower()
        return model_name.startswith("gpt-5") or model_name.startswith("o")

    def _raw_chat_completions_transport(self, base_url: str | None) -> str | None:
        transport = self.model.capabilities.get("chat_completions_transport")
        if isinstance(transport, str):
            normalized = transport.strip().lower()
            if normalized in {"requests", "httpx"}:
                return normalized
        if not base_url:
            return None
        parsed = urlparse(base_url)
        hostname = (parsed.hostname or "").lower()
        if hostname == "xlabapi.com":
            return "httpx"
        return None

    def _build_raw_chat_completions_payload(
        self,
        messages: list[dict[str, Any]],
        token_param: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model.model_name,
            "messages": messages,
        }
        payload[token_param] = self.model.request.max_tokens
        chat_supports_temperature = self.model.capabilities.get("chat_completions_supports_temperature", True)
        if self.model.request.temperature is not None and chat_supports_temperature and not self._is_thinking_mode():
            payload["temperature"] = self.model.request.temperature
        extra_body = self._build_openai_chat_extra_body()
        thinking_extra = self._build_deepseek_thinking_extra()
        if thinking_extra:
            extra_body = extra_body or {}
            extra_body.update(thinking_extra)
        if extra_body:
            payload.update(extra_body)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.model.request.reasoning_effort:
            payload["reasoning_effort"] = self.model.request.reasoning_effort
        return payload

    def _build_raw_chat_completions_headers(self, auth: Any) -> dict[str, str]:
        api_key = self._resolve_api_key(auth)
        auth_header_name = self.model.capabilities.get("chat_completions_auth_header", "Authorization")
        auth_scheme = self.model.capabilities.get("chat_completions_auth_scheme", "Bearer")

        headers = {"Content-Type": "application/json"}
        if isinstance(auth_header_name, str) and auth_header_name.strip():
            header_name = auth_header_name.strip()
            if auth_scheme in (None, "", "raw"):
                headers[header_name] = api_key
            else:
                headers[header_name] = f"{auth_scheme} {api_key}"

        extra_headers = self.model.capabilities.get("chat_completions_headers")
        if isinstance(extra_headers, dict):
            for key, value in extra_headers.items():
                if isinstance(key, str) and isinstance(value, str) and key.strip():
                    headers[key.strip()] = value

        organization = self._resolve_organization(auth)
        if organization:
            headers["OpenAI-Organization"] = organization
        return headers

    def _build_raw_chat_completions_url(self, base_url: str) -> str:
        path = self.model.capabilities.get("chat_completions_path", "/chat/completions")
        if not isinstance(path, str) or not path.strip():
            path = "/chat/completions"
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base_url.rstrip('/')}{path}"

    def _raw_chat_completions_verify_ssl(self) -> bool:
        verify_ssl = self.model.capabilities.get("chat_completions_verify_ssl", True)
        if not isinstance(verify_ssl, bool):
            return True
        return verify_ssl

    def _generate_openai_chat_completions_via_httpx(
        self,
        messages: list[dict[str, Any]],
        token_param: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        auth = self.model.auth
        base_url = self._resolve_base_url(auth)
        if not base_url:
            raise RuntimeError("Missing base_url for httpx chat completions fallback.")

        payload = self._build_raw_chat_completions_payload(messages, token_param, tools=tools)
        headers = self._build_raw_chat_completions_headers(auth)
        url = self._build_raw_chat_completions_url(base_url)
        verify_ssl = self._raw_chat_completions_verify_ssl()
        timeout = self.model.request.timeout_seconds

        def _send() -> httpx.Response:
            with httpx.Client(timeout=timeout, follow_redirects=True, verify=verify_ssl) as session:
                return session.post(url, headers=headers, json=payload)

        return self._call_with_retries(f"httpx {url}", _send)

    def _generate_openai_chat_completions_via_requests(
        self,
        messages: list[dict[str, Any]],
        token_param: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        auth = self.model.auth
        base_url = self._resolve_base_url(auth)
        if not base_url:
            raise RuntimeError("Missing base_url for requests chat completions transport.")

        payload = self._build_raw_chat_completions_payload(messages, token_param, tools=tools)
        headers = self._build_raw_chat_completions_headers(auth)
        url = self._build_raw_chat_completions_url(base_url)
        verify_ssl = self._raw_chat_completions_verify_ssl()
        timeout = self.model.request.timeout_seconds

        def _send() -> requests.Response:
            return requests.post(url, headers=headers, json=payload, timeout=timeout, verify=verify_ssl)

        return self._call_with_retries(f"requests {url}", _send)

    def _parse_raw_chat_completions_response(self, response: httpx.Response | requests.Response) -> LLMResponse:
        if response.status_code >= 400:
            detail = response.text.strip()
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    error = parsed.get("error")
                    if isinstance(error, dict):
                        detail = str(error.get("message") or error)
                    else:
                        detail = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass
            raise RuntimeError(f"status={response.status_code} | detail={detail}")

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON from chat completions endpoint: {response.text[:500]}") from exc

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("chat.completions response missing choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("chat.completions first choice is not an object.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("chat.completions message is missing or invalid.")

        content = message.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            content = "\n".join(parts) if parts else None

        tool_calls_raw: list[dict[str, Any]] | None = None
        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            tool_calls_raw = [tc for tc in raw_tool_calls if isinstance(tc, dict)]

        # Capture reasoning/thinking traces from providers that return it (DeepSeek, etc.)
        reasoning_content: str | None = (
            message.get("reasoning_content")
            or message.get("thinking")
            or message.get("reasoning")
        )

        if content is None and not tool_calls_raw:
            raise RuntimeError("chat.completions returned an empty assistant message.")
        return LLMResponse(content=content, tool_calls=tool_calls_raw, reasoning_content=reasoning_content)
