# Runpod LLM (vLLM) — Chat · Completion · JSON · Tools

[![Runpod](https://api.runpod.io/badge/your-org/runpod-llm)](https://console.runpod.io/hub)

Serverless GPU LLM inference worker built on **[vLLM](https://github.com/vllm-project/vllm)** with an OpenAI-compatible request shape. One worker, one loaded model — vLLM owns CUDA memory and refuses model swaps at runtime, so picking your backbone is an env-var concern, not a per-request one. Switch via `LLM_MODEL` and redeploy.

Inside a single invocation, vLLM batches as many prompts as you send — so the worker is dramatically more efficient when you pass a `requests` array.

## Highlights

- **Modes:** `chat` (OpenAI Chat Completions shape), `complete` (raw text), `json` (schema-constrained output)
- **Curated model allow-list** with VRAM hints (Mistral 7B, Llama 3 8B, Qwen 2.5 7B, Phi-3 Mini 128k, Gemma 2 9B). Override with `LLM_ALLOW_ANY=true` for any HF repo.
- **Sampling controls:** temperature, top_p, top_k, max_tokens, stop, presence/frequency/repetition_penalty, seed, best_of, n
- **Tool-calling shape** (`tools` + `tool_choice` — passed through; output parsed as a JSON tool-call when `tool_choice="required"`)
- **JSON mode** — schema-guided decoding via [outlines](https://github.com/outlines-dev/outlines) when available; loose JSON parsing fallback with retry
- **Batched generation** — pass `requests: [...]` to fire multiple in one call (vLLM micro-batches them for free)
- **Usage accounting** — `prompt_tokens`, `completion_tokens`, `total_tokens` returned per response
- Chat template applied by the model's tokenizer (no hand-rolled prompts)

## Curated model allow-list

| Model | Context | VRAM | Tools |
|---|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` *(default)* | 32k | ~16 GB | ✓ |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8k | ~18 GB | ✓ |
| `Qwen/Qwen2.5-7B-Instruct` | 32k | ~16 GB | ✓ |
| `microsoft/Phi-3-mini-128k-instruct` | 128k | ~9 GB | — |
| `google/gemma-2-9b-it` | 8k | ~22 GB | — |

Set `LLM_ALLOW_ANY=true` in env to accept any HuggingFace repo id.

## Environment variables

| Var | Default | What it does |
|---|---|---|
| `LLM_MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | Model to load on worker boot |
| `MAX_MODEL_LEN` | `8192` | Maximum context length passed to `vllm.LLM(max_model_len=...)` |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of GPU VRAM vLLM may use |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs to shard across |
| `LLM_ALLOW_ANY` | `false` | When `true`, bypass the curated allow-list |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace cache directory |

## Input schema

### Single request (top-level `input`)
```json
{
  "mode": "chat",                       // "chat" | "complete" | "json"
  "model": "mistralai/Mistral-7B-Instruct-v0.3",   // optional; must match loaded model
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "..."}
  ],
  "prompt": "raw prompt for 'complete' mode",
  "json_schema": { "...JSON schema..." },          // for 'json' mode
  "tools": [{"type": "function", "function": {...}}],
  "tool_choice": "auto",                            // "auto" | "required" | "none"

  "temperature": 0.7,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 512,
  "stop": ["</s>"],
  "seed": 42,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "repetition_penalty": 1.0,
  "best_of": 1,
  "n": 1
}
```

### Batched (top-level `input.requests`)
```json
{
  "requests": [
    {"mode": "chat", "messages": [...], "max_tokens": 64},
    {"mode": "complete", "prompt": "Once upon", "max_tokens": 32},
    {"mode": "json", "messages": [...], "json_schema": {...}}
  ]
}
```

All requests in a batch must target the same loaded model (the worker only holds one).

## Output shape

Single-request response:
```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},   // chat / json
      "text": "...",                                        // complete
      "tool_calls": [{"id": "...", "type": "function", "function": {...}}],
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 19, "completion_tokens": 64, "total_tokens": 83}
}
```

Batched response wraps a `results` list:
```json
{
  "model": "...",
  "count": 3,
  "results": [ { ...single-request shape... }, ... ]
}
```

## Example requests

**Basic chat:**
```json
{
  "mode": "chat",
  "messages": [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Why is the sky blue?"}
  ],
  "max_tokens": 80,
  "temperature": 0.3
}
```

**Completion:**
```json
{
  "mode": "complete",
  "prompt": "The 3 best programming languages are:\n1.",
  "stop": ["\n\n"],
  "max_tokens": 60
}
```

**JSON mode with schema:**
```json
{
  "mode": "json",
  "messages": [
    {"role": "user", "content": "Extract: 'Iran is 1.6M km², ~85M people, capital Tehran'"}
  ],
  "json_schema": {
    "type": "object",
    "required": ["country", "area_km2", "population", "capital"],
    "properties": {
      "country": {"type": "string"},
      "area_km2": {"type": "integer"},
      "population": {"type": "integer"},
      "capital": {"type": "string"}
    }
  }
}
```

**Tool-calling shape:**
```json
{
  "mode": "chat",
  "messages": [{"role": "user", "content": "What's the weather in Tehran tomorrow?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}, "when": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "required"
}
```

**Batched:**
```json
{
  "requests": [
    {"mode": "chat", "messages": [{"role": "user", "content": "Hi"}],   "max_tokens": 8},
    {"mode": "chat", "messages": [{"role": "user", "content": "Bye"}],  "max_tokens": 8},
    {"mode": "complete", "prompt": "Hello", "max_tokens": 16}
  ]
}
```

## Local testing

```bash
pip install -r requirements.txt   # (vLLM is heavy; tests don't require it)
python3 test_handler.py
```

`test_handler.py` stubs out `vllm`, `transformers`, `torch`, and `outlines` via `sys.modules` injection — it runs without a GPU and without the real model. 15 tests cover sampling-param construction, chat template application, JSON loose-parsing, stop-sequence forwarding, batched dispatch, usage accounting, and per-request error capture.

## Deployment

1. **Build:** `docker build -t your-org/runpod-llm:latest .`
2. **Push or link the repo** on the RunPod Hub at https://console.runpod.io/hub.
3. **Set env vars** — at minimum `LLM_MODEL` and `MAX_MODEL_LEN`.
4. **Pick a GPU** matching the model's VRAM column. Cold start = model download → first-request latency; subsequent invocations reuse the loaded model.

## Performance notes

- vLLM uses **continuous batching** under the hood: multiple `requests` in one call cost roughly the same as one — pack them when you can.
- The first request after worker boot pays the model load cost (10–60 s depending on size). After that, you're amortized.
- `MAX_MODEL_LEN` higher than the model's native context just wastes KV cache memory — match it to the model.
- For multi-GPU sharding set `TENSOR_PARALLEL_SIZE=2` (or higher) and pick a multi-GPU pod.

## Notes

- The worker holds **one** loaded model per process. `LLM_MODEL` env determines which. Per-request `model` is validated but cannot trigger a swap (returns an error if it doesn't match).
- For JSON mode, schema-guided decoding via [outlines](https://github.com/outlines-dev/outlines) is preferred when available; otherwise the worker returns the model's free-form completion and runs a loose JSON parser with one retry.
- Tool-calling here is **shape-level** — the worker doesn't execute tools. When `tool_choice="required"`, the system prompt is augmented to force a JSON tool-call shape, which the worker parses and returns in `tool_calls`.
