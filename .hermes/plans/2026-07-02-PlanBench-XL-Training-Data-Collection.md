# PlanBench-XL Training Data Collection Implementation Plan

> **For Hermes / Grok Build:** Use subagent-driven-development or grok-build (with plan approval) to implement this plan task-by-task. Follow TDD, frequent commits, exact paths. Reference this plan in prompts. Do not skip verification steps. Prioritize working collector first, then polish.

**Goal:** Repurpose/extend PlanBench-XL (currently a long-horizon planning benchmark with massive tool retrieval + noise + blockers) into a high-quality training data collector. Produce OpenAI-compatible chat messages + tools format JSONL (suitable for SFT on tool-use / agentic planning, including Azure OpenAI endpoints). Support recording full per-turn or per-trajectory interactions while preserving benchmark capabilities.

**Architecture:**
- Keep core runtime (retrieval, noisy/blocker events, state tracking, resume via progress/, executor) as-is for trajectory generation.
- Add explicit "data collection" layer that captures at every LLM decision point: the exact `messages` list sent, the `tools` (OpenAI function schema) available at that step, the raw model response, plus rich metadata (query_id, step, action_type, ground-truth hints if available, success signals).
- Support dual modes: (1) original prompted XML-tag style (current system prompt), (2) optional native OpenAI `tools=` + `tool_calls` for modern models (preferred for training data quality).
- Output: structured logs (jsonl or per-trajectory) under `outputs/<run_id>/collect/` or dedicated `data/trajectories/`. Post-process via existing `sft-data-preparation` patterns (apply target model's chat template to produce final `{"text": ...}` jsonl).
- Config-driven via run YAMLs (new `data_collection` section). Reuse model_registry + OpenAI-compatible auth (base_url/api_key for Azure).
- Tools data already nearly matches OpenAI function schema (type/function/name/description/parameters/strict) — strip extras for API, keep for metadata.

**Tech Stack:**
- Python 3.11+, existing deps (openai, pyyaml, etc.) + optional for SFT later (transformers for chat_template).
- OpenAI client (or raw requests/httpx fallback) for Azure-compatible endpoints.
- Existing dataclasses (RunnerConfig, OutputConfig, AgentState, ModelProfile).
- Atomic `dump_json` from core/utils.
- Git branches: main (current) + collect (data-oriented).

**Current State (as of 2026-07-02 inspection):**
- `src/env/run.py`: Stub ("under construction").
- Full runtime in `src/env/runtime/runner.py` (1566 LOC), `llm.py`, `parsing.py`, `prompts.py`, events, retriever, core/.
- Tools in `src/data/retail/baseline_tools.json` etc. use OpenAI-like `{"type": "function", "name": ..., "parameters": {...}, ...}` + domain extras.
- LLM calls use simple `history: list[dict role, content]` → plain text. No native `tools=` or `tool_calls` yet. Custom `<retrieve_tools>JSON</retrieve_tools>` etc. parsing.
- Rich progress: `outputs/.../progress/queries/*.json` (steps_trace, checkpoints), `result.jsonl`, `save_raw_llm_response` flag already exists in OutputConfig.
- Configs support base_url for local/Azure-compatible.
- System prompt (`src/env/prompt/system_runtime_prompt.txt`): Instructs single XML-tag action + retrieval guidance.
- No .hermes/ or active outputs/ yet in this checkout.
- collect branch exists (mostly data/configs; run.py also stub).

**Assumptions:**
- User will provide Azure endpoint details (base_url like `https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=...`, key).
- Focus on retail domain first (327 queries, 1.6k+ tools).
- Data for SFT on planning/tool-use (long-horizon, retrieval decisions, multi-step composition).
- Preserve eval/benchmark for validation of collected trajectories.
- Later integration with `sft-data-preparation` skill for templated jsonl.

**Risks & Tradeoffs:**
- Changing to native tools may require prompt updates and dual parsing paths → risk to benchmark fidelity. Mitigate: feature flag `use_native_tools`.
- Massive tools: always use *retrieved subset* only (never full 1665). Good for realistic training.
- History building: must accurately reconstruct messages (including tool results as "tool" role or injected text) for SFT.
- Azure quirks: api-version, headers, rate limits (current rate limiter + retry logic helps). Test with small samples.
- Data quality: Need diverse good trajectories (use blocker/noise configs to generate variety). Filter low-quality later.
- Volume: Per-turn logging can be large; support sampling + compression.
- Stubbed run.py: Must restore functional entrypoint.
- Open questions: Full trajectory vs per-turn records? Include ground-truth path labels for supervised? Versioning of collected data?

**Step-by-Step Plan (Bite-sized tasks, 2-10 min each where possible; TDD preferred for new logic)**

### Task 1: Project Hygiene & Exploration Baseline
**Objective:** Ensure clean workspace, capture current state, prepare for changes.

**Files:**
- Modify: none (or .gitignore if needed)
- Create: (if missing) `src/env/config/runs/retail/collect/` examples later
- Test: n/a

**Steps:**
1. `cd /home/alex/Workspace/PlanBench-XL`
2. `git status && git branch -a`
3. `python -c "import json; print(len(json.load(open('src/data/retail/baseline_tools.json'))))"` (confirm ~1665 tools)
4. Verify tool schema matches OpenAI: already close (see sample).
5. `mkdir -p outputs/debug data/trajectories .hermes/plans` (already partially done)
6. Commit: `git add -A && git commit -m "chore: baseline state for data collection planning" --allow-empty` (optional)

**Verification:** `ls src/data/retail/*.json | wc -l` == 8 (datatypes, database, baseline/noisy/blocker_tools, tasks, queries, paths_set_catalog).

### Task 2: Make run.py Functional Entry Point (Restore + Collect Mode)
**Objective:** Replace stub with working CLI that loads RunnerConfig and supports `--collect` or config flag.

**Files:**
- Modify: `src/env/run.py` (full rewrite from stub)
- Test: `tests/` (create minimal if none)

**Steps:**
1. Read full current run.py (stub).
2. Implement `main()` that:
   - Parses args: `--run_config`, `--set`, `--collect-only`, `--sample 5`
   - Loads config via existing `from env.core.config import load_config`
   - Instantiates runner
   - Runs with data collection enabled
3. Add minimal argparse or use existing patterns from batch script.
4. Write failing "smoke test" first if tests added: `python -c "from env.run import main; ..."` or pytest.
5. Run with small override to verify loads without full execution.

**Exact sketch (high-level for impl):**
```python
# src/env/run.py
import argparse
from pathlib import Path
from env.core.config import load_config
from env.runtime.runner import run_queries  # or class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", required=True)
    parser.add_argument("--collect", action="store_true")
    # ...
    args = parser.parse_args()
    cfg = load_config(Path(args.run_config), ...)
    if args.collect or cfg.get("data_collection", {}).get("enabled"):
        # enable logging hooks
    # call runner
if __name__ == "__main__":
    ...
```
**Verification:** `python src/env/run.py --run_config src/env/config/runs/retail/... --set query_sample.size=1 --help` runs without "under construction".

### Task 3: Extend Configs for Data Collection
**Objective:** Add `data_collection` section to types + default base config + example run YAML.

**Files:**
- Modify: `src/env/core/types.py` (add DataCollectionConfig dataclass)
- Modify: `src/env/core/config.py` (merge support)
- Modify: `src/env/config/base/env.yaml`
- Create: `src/env/config/runs/retail/collect-example.yaml` (or under gpt-*/)
- Create: example model if needed for Azure

**DataCollectionConfig sketch:**
```python
@dataclass(slots=True)
class DataCollectionConfig:
    enabled: bool = False
    output_subdir: str = "collect"
    format: str = "openai_chat"  # or "raw", "full_trajectory"
    use_native_tools: bool = False
    log_per_turn: bool = True
    log_full_trajectories: bool = True
    include_metadata: bool = True
    max_trajectories: Optional[int] = None
    filter_success_only: bool = False  # post or during
```

**Steps:**
1. Add dataclass in types.py (after OutputConfig).
2. Update RunnerConfig to include it.
3. Update load_config / deep_merge.
4. Add to base/env.yaml under top level or output.
5. Create minimal run YAML that sets `data_collection.enabled: true`, reuses existing retail default + small sample.
6. Document in README section.

**Verification:** Load a config YAML and assert `cfg.data_collection.enabled`.

### Task 4: Enhance LLMClient for Tools + Structured Logging
**Objective:** Support optional `tools` param + return richer response. Add hook for collection.

**Files:**
- Modify: `src/env/runtime/llm.py` (extend generate, add `_generate_..._with_tools`, handle tool_calls in response)
- Modify: `src/env/core/types.py` if needed for response model
- Test: unit for client (new or inline)

**Key changes:**
- `def generate(self, history: list[dict], tools: list[dict] | None = None) -> dict | str:`
  - If tools and native supported: pass `tools=[{"type":"function", "function": t} for t in cleaned_tools]` to client.chat.completions.create(..., tools=..., tool_choice="auto")
  - Parse response.choices[0].message (content or tool_calls)
  - Always return dict: {"content": ..., "tool_calls": [...], "raw": response} or similar.
- Fallback to text for non-native.
- Add `save_raw_llm_request` or always capture request payload (messages + tools + model params).
- Keep raw requests/httpx paths updated for tools (payload has "tools").

**Verification:** 
- Call with tools=None → backward compat (str or same).
- Call with tools → response has tool_calls or content.
- Small test: `python -c 'from env.runtime.llm import LLMClient; ...'`

### Task 5: Instrument Runner for Message/Tool Capture + Logging
**Objective:** At every LLM decision, capture full request context and persist to collect logs. Update history building to support tool roles if native.

**Files:**
- Modify: `src/env/runtime/runner.py` (major: find LLM call sites, wrap, use dump_json for collect records; update AgentState or add collector class)
- Modify: `src/env/core/types.py` (perhaps add TurnRecord or use existing steps_trace)
- Modify: `src/env/runtime/parsing.py` (extend for native tool_calls if enabled)
- Modify: `src/env/runtime/prompts.py` (minor, if needed for native prompts)

**Capture at LLM call:**
```python
# pseudo in loop
available_tools = [strip_extras(t) for t in state.available_tools]  # to OpenAI schema
request = {
    "messages": build_messages(history_so_far, state),  # ensure correct roles
    "tools": available_tools if collect_cfg.use_native_tools else None,
    "model": model.model_name,
    ...
}
raw_response = llm.generate(history, tools=... )
record = {
    "query_id": qid,
    "step": state.total_step_count,
    "request": request,
    "response": raw_response,
    "metadata": {"action_parsed": ..., "success": ..., "blocker": ...}
}
dump_json(collect_dir / f"{qid}_step{step}.json", record)  # or append jsonl
# also augment steps_trace
```

**History building notes:** Ensure messages include prior assistant tool_calls + tool results (role="tool" or injected). For prompted mode, messages can embed tool defs in system or as user.

**Steps (TDD):**
1. Add helper `def _build_openai_messages(...)` and `def _to_openai_tools(tools: list[dict]) -> list[dict]`.
2. Locate LLM call sites (search for generate or llm_client).
3. Wrap calls when data_collection.enabled.
4. Save atomically with existing dump_json.
5. Update resume to load prior traces if needed.
6. Write a small integration test or smoke: run 1 query with collect, inspect output json.

**Verification:** After small run: `find outputs -name '*collect*' | head -3`; `python -c "
import json
rec = json.load(open(...))
assert 'messages' in rec['request']
assert isinstance(rec.get('response'), (str, dict))
print('OK')
" `

### Task 6: Tool Schema Normalization + Native Support in Executor/Retriever
**Objective:** Clean conversion from internal tool dicts to pure OpenAI function schema. Support execution from native tool_calls.

**Files:**
- Modify: `src/env/domains/executor.py`
- Modify: `src/env/runtime/runner.py` (call sites)
- Modify: `src/env/events/...` if needed for blockers on native
- Add helper in core/utils.py: `def to_openai_tool_schema(internal_tool: dict) -> dict`

**Sketch:**
```python
def to_openai_tool_schema(t: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters", t.get("input_schema", {})),
            "strict": t.get("strict", True),
        }
    }
```
Keep internal fields in metadata.

For native: if response has tool_calls, map name/arguments to executor, get result, append as tool message.

**Verification:** Roundtrip a baseline tool → schema → back; execute via native path in smoke test.

### Task 7: Output Formats, Resume, and Batch Support
**Objective:** Define stable collect output. Support resume for long collections. Update batch script.

**Files:**
- Modify: `scripts/run_retail_batch.py`
- Modify: runner.py (collect writer)
- Create: `src/env/collector.py` (new dedicated class for separation, recommended for cleanliness)
- Update README

**Outputs sketch:**
- `outputs/<run>/collect/turns.jsonl` : one line per LLM turn `{"messages": [...], "tools": [...], "response": {...}, "meta": {...}}`
- `outputs/<run>/collect/trajectories/<qid>.json` : full per-query
- Or unified with progress/.

Support `format: "openai_chat"` exactly matching what would be sent to /chat/completions.

**Verification:** Run batch small sample; count lines in jsonl == expected turns.

### Task 8: SFT Post-Processing Integration
**Objective:** Provide example/script to turn collected logs into model-specific SFT jsonl using sft-data-preparation patterns.

**Files:**
- Create: `scripts/collect_to_sft.py` (or in data/)
- Update docs

**Example flow (from skill):**
```python
# After collection
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/path/to/target-model")
for line in open("turns.jsonl"):
    data = json.loads(line)
    # reconstruct full_msgs from request.messages + response
    text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
```

**Verification:** End-to-end on 5 examples; spot-check templated output has correct tags for the model.

### Task 9: Documentation, Examples, and Azure Config
**Objective:** Update README, add Azure example config, quickstart for collection.

**Files:**
- Modify: `README.md` (new section "Collecting Training Data", Azure notes)
- Create: `src/env/config/models/openai/azure-example.yaml`
- Create: `src/env/config/runs/retail/collect/retail_collect_default.yaml`
- Add example command in README.

**Azure config sketch:**
```yaml
# model yaml
base_url: "https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOY/chat/completions?api-version=2024-08-01-preview"
api_key_env: AZURE_OPENAI_API_KEY
model: gpt-4o  # or deployment name
api_style: chat_completions
# capabilities for tools support etc.
```

**Verification:** Docs build/lint if any; `grep -A5 "Training Data" README.md`

### Task 10: Validation, Testing, and Polish
**Objective:** Ensure collected data is usable, benchmark still works, add basic tests/smokes.

**Files:**
- Modify: existing tests if present (search showed none obvious)
- Create: `tests/test_data_collection.py` (minimal)
- Scripts for smoke: `python scripts/smoke_collect.py`

**Steps:**
1. Run benchmark small sample (collect disabled) → eval passes.
2. Run with collect enabled on 3-5 queries → inspect 10+ records.
3. Validate schema: every record has "messages" list with roles, optional "tools".
4. Optional: end-to-end with sft prep on a toy model tokenizer.
5. Commit per task.
6. Update .gitignore for outputs/ if needed.

**Verification Commands:**
```bash
python src/env/run.py --run_config ... --set query_sample.size=2 --set data_collection.enabled=true
python -c "
import json, glob
recs = [json.loads(l) for f in glob.glob('outputs/*/collect/*.jsonl') for l in open(f)]
print(len(recs), 'records')
assert all('messages' in r.get('request', {}) for r in recs)
print('Schema OK')
"
```

**Overall Timeline / Phasing (recommended):**
1. Tasks 1-3 (setup + config + entrypoint) — foundation.
2. Tasks 4-5 (LLM + runner instrumentation) — core collection.
3. Task 6 (native/tools) — optional but high value.
4. 7-8 (formats + SFT) .
5. 9-10 (docs + validate).

**Post-Plan Handoff:**
- After this plan is approved: Use `grok-build` or delegate_task with prompt referencing this exact file + "Implement Task X only. Await direction before next."
- "先實做完，再補好文件" preference: Prioritize running collector + real jsonl output + smoke verification before heavy docs.
- For complex phases: phased grok -p ... --continue --cwd /home/alex/Workspace/PlanBench-XL

**Open Items for User/BOSS:**
- Confirm native_tools vs pure prompted for first data batch.
- Target model(s) for SFT template (for post-proc).
- Azure endpoint example (redacted) or test creds.
- Desired output volume / filtering criteria.
- Any preference on collect branch vs main?

**Success Criteria:**
- `python src/env/run.py --run_config <collect-yaml>` produces valid `messages` + `tools` records.
- Records can be fed to `apply_chat_template` or OpenAI fine-tune format.
- Small benchmark run + collect still produces evaluable results.
- Works with Azure-compatible base_url (user verified).

This plan makes implementation obvious. All paths, commands, and sketches are concrete.

---
**References (from inspection):**
- README.md (original benchmark usage)
- src/env/runtime/*.py, core/types.py, utils.py
- src/data/retail/baseline_tools.json (tool schema)
- src/env/prompt/system_runtime_prompt.txt
- sft-data-preparation skill
- grok-build / writing-plans / plan skills

**Next:** Review plan → explicit "Go" / approval for specific task or full delegation.


---
**Progress Update (2026-07-02):**

- Tasks 1-3: Completed (hygiene, run.py CLI with --collect by Claude + manual, DataCollectionConfig + example YAML).
- Tasks 4-5 (runner instrumentation): Core implemented. `_log_collection_turn` + hook after every `llm_client.generate`. Captures `messages` + `tools` (OpenAI schema) + raw response. Per-turn JSON + JSONL + optional `trajectory.json` for full trajectories.
- Full trajectory support added in `_finalize_query`.
- Verified via direct unit smoke + integration test (messages/tools/trajectory files produced correctly).
- Documentation: Added "📥 Collecting Training Data for SFT" section to README.md.
- Commits on collect branch include feature work + verification.

Core collector is functional. Next: post-processing to SFT, native tools dual-mode, batch integration.
