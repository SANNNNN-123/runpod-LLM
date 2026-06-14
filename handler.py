"""Runpod serverless LLM handler (vLLM-backed).

Provides an OpenAI-compatible request shape for:
  * chat completions   (mode == "chat")
  * text completions   (mode == "complete")
  * guided JSON output (mode == "json")

A single vLLM ``LLM`` engine is created per worker process and reused. Switching
the loaded model at runtime is **not** supported because vLLM holds GPU state;
the worker must be restarted with a new ``LLM_MODEL`` env value to load a
different model.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import runpod
except Exception:
    runpod = None

try:
    import requests
except Exception:
    requests = None

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None

try:
    from vllm.outputs import RequestOutput
except Exception:
    RequestOutput = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    import outlines
except Exception:
    outlines = None




ALLOWED_MODELS: Dict[str, Dict[str, Any]] = {
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "label": "Mistral 7B Instruct v0.3",
        "vram_gb": 16,
        "context": 32768,
        "supports_tools": True,
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "label": "Llama 3 8B Instruct",
        "vram_gb": 18,
        "context": 8192,
        "supports_tools": True,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "label": "Qwen 2.5 7B Instruct",
        "vram_gb": 16,
        "context": 32768,
        "supports_tools": True,
    },
    "microsoft/Phi-3-mini-128k-instruct": {
        "label": "Phi-3 Mini 128k Instruct",
        "vram_gb": 9,
        "context": 131072,
        "supports_tools": False,
    },
    "google/gemma-2-9b-it": {
        "label": "Gemma 2 9B Instruct",
        "vram_gb": 22,
        "context": 8192,
        "supports_tools": False,
    },
}


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"




_LLM_INSTANCE: Optional[Tuple[str, Any, Any]] = None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _resolve_model_name(requested: Optional[str]) -> str:
    """Resolve which model to load.

    Priority: explicit request (must match curated registry) > env > default.
    """
    if requested:
        if requested not in ALLOWED_MODELS:
            raise ValueError(
                f"model '{requested}' is not in the curated allow-list. "
                f"Allowed: {sorted(ALLOWED_MODELS)}"
            )
        return requested
    env_model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    if env_model not in ALLOWED_MODELS:
        raise ValueError(
            f"LLM_MODEL env '{env_model}' is not in the allow-list. "
            f"Allowed: {sorted(ALLOWED_MODELS)}"
        )
    return env_model


def get_llm(requested_model: Optional[str] = None) -> Tuple[str, Any, Any]:
    """Return the cached ``(name, llm, tokenizer)`` triple.

    Raises if a different model is requested than the one already loaded.
    """
    global _LLM_INSTANCE
    name = _resolve_model_name(requested_model)

    if _LLM_INSTANCE is not None:
        loaded_name = _LLM_INSTANCE[0]
        if name != loaded_name:
            raise RuntimeError(
                f"This worker is bound to model '{loaded_name}'. "
                f"vLLM workers cannot swap models at runtime — restart the "
                f"worker with LLM_MODEL='{name}' to load a different model."
            )
        return _LLM_INSTANCE

    if LLM is None:
        raise RuntimeError(
            "vllm is not installed; cannot construct an LLM engine."
        )
    if AutoTokenizer is None:
        raise RuntimeError(
            "transformers is not installed; cannot load tokenizer."
        )

    max_model_len = _env_int("MAX_MODEL_LEN", 8192)
    gpu_util = _env_float("GPU_MEMORY_UTILIZATION", 0.9)
    tp_size = _env_int("TENSOR_PARALLEL_SIZE", 1)
    dtype = os.getenv("LLM_DTYPE", "auto")
    trust_remote = os.getenv("LLM_TRUST_REMOTE_CODE", "true").lower() not in (
        "0",
        "false",
        "no",
    )

    llm = LLM(
        model=name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        tensor_parallel_size=tp_size,
        dtype=dtype,
        trust_remote_code=trust_remote,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        name, trust_remote_code=trust_remote
    )
    _LLM_INSTANCE = (name, llm, tokenizer)
    return _LLM_INSTANCE




_DEFAULT_SAMPLING: Dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 512,
    "stop": None,
    "seed": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "best_of": 1,
    "n": 1,
}


def _coerce_stop(stop: Any) -> Optional[List[str]]:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        out = [str(s) for s in stop if s is not None]
        return out or None
    return None


def build_sampling_params(
    spec: Dict[str, Any],
    guided_json: Optional[Any] = None,
    guided_regex: Optional[str] = None,
) -> Any:
    """Build a ``SamplingParams`` from a user-supplied dict.

    Unknown keys are ignored; the dict is merged onto the defaults. Guided
    decoding is best-effort: vLLM 0.5+ accepts ``guided_json`` / ``guided_regex``
    kwargs directly, while older versions ignore them. We pass them through
    when set and silently fall back on TypeError.
    """
    if SamplingParams is None:
        raise RuntimeError("vllm.SamplingParams not available.")

    merged: Dict[str, Any] = {}
    for k, default in _DEFAULT_SAMPLING.items():
        merged[k] = spec.get(k, default) if isinstance(spec, dict) else default
    merged["stop"] = _coerce_stop(merged.get("stop"))

    if int(merged.get("best_of") or 1) < int(merged.get("n") or 1):
        merged["best_of"] = int(merged["n"])

    kwargs: Dict[str, Any] = {
        "temperature": float(merged["temperature"]),
        "top_p": float(merged["top_p"]),
        "top_k": int(merged["top_k"]),
        "max_tokens": int(merged["max_tokens"]),
        "presence_penalty": float(merged["presence_penalty"]),
        "frequency_penalty": float(merged["frequency_penalty"]),
        "repetition_penalty": float(merged["repetition_penalty"]),
        "n": int(merged["n"]),
        "best_of": int(merged["best_of"]),
    }
    if merged.get("seed") is not None:
        try:
            kwargs["seed"] = int(merged["seed"])
        except Exception:
            pass
    if merged.get("stop"):
        kwargs["stop"] = merged["stop"]

    if guided_json is not None:
        kwargs["guided_json"] = guided_json
    if guided_regex is not None:
        kwargs["guided_regex"] = guided_regex

    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("guided_json", None)
        kwargs.pop("guided_regex", None)
        return SamplingParams(**kwargs)




def _validate_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty list")
    out: List[Dict[str, str]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"messages[{i}] is not an object")
        role = m.get("role")
        content = m.get("content", "")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"messages[{i}].role invalid: {role!r}")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        out.append({"role": role, "content": content})
    return out


def apply_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Render messages to a prompt string using the tokenizer's chat template.

    Falls back to a simple OpenAI-style concatenation if no template is set.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=bool(add_generation_prompt),
        )
    except Exception:
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)


def _tool_choice_system_addendum(
    tools: List[Dict[str, Any]],
    tool_choice: Any,
) -> Optional[str]:
    """Build a system-prompt addendum instructing the model to emit a tool call.

    We do not invoke any tools — this just shapes the output as JSON matching
    the requested tool schema so the caller can route it.
    """
    if tool_choice is None or tool_choice == "none":
        return None
    if not tools:
        return None
    if tool_choice == "auto":
        return (
            "You may optionally call one of the provided tools. If you choose "
            "to call a tool, respond with a single JSON object of the form "
            '{"tool_calls": [{"name": "<tool>", "arguments": {...}}]} and no '
            "other text. Otherwise, answer normally."
        )
    chosen_name: Optional[str] = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        chosen_name = fn.get("name") if isinstance(fn, dict) else None
    if chosen_name:
        schema_for = next(
            (
                t for t in tools
                if isinstance(t, dict)
                and (t.get("function") or {}).get("name") == chosen_name
            ),
            None,
        )
        schema_text = json.dumps(
            schema_for or {"name": chosen_name}, ensure_ascii=False
        )
        return (
            f"You must call the tool '{chosen_name}'. Respond with exactly "
            "one JSON object of the form "
            '{"tool_calls": [{"name": "' + chosen_name + '", "arguments": '
            "{...}}]} and no other text. Tool schema: " + schema_text
        )
    schema_text = json.dumps(tools, ensure_ascii=False)
    return (
        "You must call one of the provided tools. Respond with exactly one "
        "JSON object of the form "
        '{"tool_calls": [{"name": "<tool>", "arguments": {...}}]} '
        "and no other text. Available tools: " + schema_text
    )


def _prepend_system(
    messages: List[Dict[str, str]], addendum: str
) -> List[Dict[str, str]]:
    """Prepend ``addendum`` to an existing system message or insert a new one."""
    out = list(messages)
    if out and out[0].get("role") == "system":
        merged = (out[0].get("content") or "") + "\n\n" + addendum
        out[0] = {"role": "system", "content": merged.strip()}
        return out
    return [{"role": "system", "content": addendum}] + out




_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
_JSON_ARR_RE = re.compile(r"\[[\s\S]*\]", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    """Remove a leading/trailing markdown code fence around JSON."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def parse_json_loose(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """Best-effort JSON parse — strip fences then look for first {...} or [...]."""
    if not isinstance(text, str):
        return None, "non-string output"
    candidate = _strip_code_fence(text)
    try:
        return json.loads(candidate), None
    except Exception as e:
        first_err = str(e)
    m = _JSON_OBJ_RE.search(candidate) or _JSON_ARR_RE.search(candidate)
    if m:
        try:
            return json.loads(m.group(0)), None
        except Exception as e:
            return None, f"{first_err}; substring parse: {e}"
    return None, first_err




def _generate(
    llm: Any,
    prompts: List[str],
    sampling: Any,
) -> List[Any]:
    """Single batched call into vLLM. Returns the raw outputs list."""
    out = llm.generate(prompts, sampling)
    if not isinstance(out, list):
        out = [out]
    return out


def _completion_from_output(req_output: Any, index: int = 0) -> Dict[str, Any]:
    """Extract the first CompletionOutput of a vLLM RequestOutput."""
    outputs = getattr(req_output, "outputs", None) or []
    if not outputs:
        return {"text": "", "finish_reason": "error", "token_ids": []}
    co = outputs[0]
    return {
        "text": getattr(co, "text", "") or "",
        "finish_reason": getattr(co, "finish_reason", "stop") or "stop",
        "token_ids": list(getattr(co, "token_ids", []) or []),
        "index": index,
    }


def _usage_from_output(req_output: Any, completion: Dict[str, Any]) -> Dict[str, int]:
    prompt_ids = list(getattr(req_output, "prompt_token_ids", []) or [])
    prompt_tokens = len(prompt_ids)
    completion_tokens = len(completion.get("token_ids") or [])
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
    }




def _build_chat_prompt(
    tokenizer: Any,
    req: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Return (prompt_string, parse_hints) for chat mode."""
    messages = _validate_messages(req.get("messages"))
    tools = req.get("tools") or []
    tool_choice = req.get("tool_choice")
    parse_hints: Dict[str, Any] = {"expect_tool_call": False}
    if tool_choice and tool_choice != "none" and tools:
        addendum = _tool_choice_system_addendum(tools, tool_choice)
        if addendum is not None:
            messages = _prepend_system(messages, addendum)
            parse_hints["expect_tool_call"] = tool_choice in (
                "required",
            ) or isinstance(tool_choice, dict)
            parse_hints["tool_choice"] = tool_choice
            parse_hints["tools"] = tools
    prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    return prompt, parse_hints


def _build_completion_prompt(req: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("'prompt' (string) is required in completion mode")
    return prompt, {}


def _build_json_prompt(
    tokenizer: Any,
    req: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Optional[Any], Optional[str]]:
    """JSON mode supports either messages or a single prompt.

    Returns (prompt, parse_hints, guided_json, guided_regex).
    """
    schema = req.get("schema") or req.get("json_schema")
    pattern = req.get("regex") or req.get("guided_regex")
    parse_hints: Dict[str, Any] = {"expect_json": True, "schema": schema}

    if "messages" in req and req["messages"]:
        messages = _validate_messages(req["messages"])
        guidance = (
            "Respond with valid JSON only. No prose, no markdown, no code "
            "fences."
        )
        if schema:
            guidance += (
                " The output must match this JSON schema: "
                + json.dumps(schema, ensure_ascii=False)
            )
        messages = _prepend_system(messages, guidance)
        prompt = apply_chat_template(
            tokenizer, messages, add_generation_prompt=True
        )
    else:
        raw = req.get("prompt")
        if not isinstance(raw, str) or not raw:
            raise ValueError(
                "json mode requires 'messages' or 'prompt' as input"
            )
        prefix = "Respond with valid JSON only.\n"
        if schema:
            prefix += (
                "Schema: "
                + json.dumps(schema, ensure_ascii=False)
                + "\n"
            )
        prompt = prefix + raw
    return prompt, parse_hints, schema, pattern




def _assemble_chat_choice(
    completion: Dict[str, Any],
    parse_hints: Dict[str, Any],
) -> Dict[str, Any]:
    text = completion["text"]
    choice: Dict[str, Any] = {
        "index": int(completion.get("index", 0)),
        "finish_reason": completion["finish_reason"],
    }
    if parse_hints.get("expect_tool_call"):
        obj, err = parse_json_loose(text)
        tool_calls: List[Dict[str, Any]] = []
        if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
            for i, tc in enumerate(obj["tool_calls"]):
                if not isinstance(tc, dict):
                    continue
                tool_calls.append(
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": str(tc.get("name", "")),
                            "arguments": json.dumps(
                                tc.get("arguments") or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
        elif isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            tool_calls.append(
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": str(obj.get("name", "")),
                        "arguments": json.dumps(
                            obj.get("arguments") or {}, ensure_ascii=False
                        ),
                    },
                }
            )
        message: Dict[str, Any] = {"role": "assistant", "content": None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        else:
            message["content"] = text
            if err:
                choice["tool_call_parse_error"] = err
        choice["message"] = message
    else:
        choice["message"] = {"role": "assistant", "content": text}
    return choice


def _assemble_completion_choice(completion: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": int(completion.get("index", 0)),
        "text": completion["text"],
        "finish_reason": completion["finish_reason"],
    }


def _assemble_json_choice(
    completion: Dict[str, Any],
    parse_hints: Dict[str, Any],
) -> Dict[str, Any]:
    text = completion["text"]
    obj, err = parse_json_loose(text)
    choice: Dict[str, Any] = {
        "index": int(completion.get("index", 0)),
        "finish_reason": completion["finish_reason"],
        "message": {
            "role": "assistant",
            "content": text,
        },
    }
    if obj is not None:
        choice["json"] = obj
    if err:
        choice["json_parse_error"] = err
    return choice




def _prepare_one(
    tokenizer: Any,
    req: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any], Any]:
    """Return (mode, prompt, parse_hints, sampling_params) for one request.

    Sampling-param construction is done here so guided-decoding kwargs can be
    plumbed in mode-specifically (JSON mode adds them).
    """
    mode = (req.get("mode") or "chat").lower()
    if mode not in ("chat", "complete", "completion", "json"):
        raise ValueError(
            f"mode must be one of chat|complete|json (got {mode!r})"
        )
    if mode == "completion":
        mode = "complete"

    sampling_spec: Dict[str, Any] = {}
    for k in _DEFAULT_SAMPLING:
        if k in req:
            sampling_spec[k] = req[k]

    if mode == "chat":
        prompt, parse_hints = _build_chat_prompt(tokenizer, req)
        sampling = build_sampling_params(sampling_spec)
    elif mode == "complete":
        prompt, parse_hints = _build_completion_prompt(req)
        sampling = build_sampling_params(sampling_spec)
    else:
        prompt, parse_hints, guided_json, guided_regex = _build_json_prompt(
            tokenizer, req
        )
        sampling = build_sampling_params(
            sampling_spec,
            guided_json=guided_json,
            guided_regex=guided_regex,
        )
    return mode, prompt, parse_hints, sampling


def _assemble_one(
    mode: str,
    req_output: Any,
    parse_hints: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    completion = _completion_from_output(req_output)
    usage = _usage_from_output(req_output, completion)
    if mode == "chat":
        choice = _assemble_chat_choice(completion, parse_hints)
    elif mode == "complete":
        choice = _assemble_completion_choice(completion)
    else:
        choice = _assemble_json_choice(completion, parse_hints)
    return {
        "choices": [choice],
        "usage": usage,
        "model": model_name,
        "mode": mode,
        "id": getattr(req_output, "request_id", None) or f"cmpl-{int(time.time() * 1000)}",
    }




def _coerce_request_list(inp: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """Return (requests_list, is_batched). ``is_batched`` controls response shape."""
    if isinstance(inp.get("requests"), list) and inp["requests"]:
        items: List[Dict[str, Any]] = []
        for r in inp["requests"]:
            if not isinstance(r, dict):
                raise ValueError("each item in 'requests' must be an object")
            items.append(r)
        return items, True
    return [inp], False


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    inp = event.get("input") or {}

    try:
        requests_list, is_batched = _coerce_request_list(inp)
    except Exception as e:
        return {"error": str(e)}

    requested_models = {r.get("model") for r in requests_list if r.get("model")}
    if len(requested_models) > 1:
        return {
            "error": (
                "All requests in a batch must target the same model; "
                f"received: {sorted(requested_models)}."
            )
        }
    requested_model = next(iter(requested_models), None)

    try:
        model_name, llm, tokenizer = get_llm(requested_model)
    except Exception as e:
        return {"error": str(e)}

    prepared: List[Tuple[str, str, Dict[str, Any], Any]] = []
    errors: List[Optional[str]] = []
    for r in requests_list:
        try:
            prepared.append(_prepare_one(tokenizer, r))
            errors.append(None)
        except Exception as e:
            prepared.append(("error", "", {}, None))
            errors.append(str(e))

    prompts: List[str] = []
    sampling_params: List[Any] = []
    request_index: List[int] = []
    for i, (mode, prompt, _hints, sampling) in enumerate(prepared):
        if mode == "error":
            continue
        prompts.append(prompt)
        sampling_params.append(sampling)
        request_index.append(i)

    raw_outputs: List[Any] = []
    if prompts:
        try:
            if len(prompts) == 1:
                raw_outputs = _generate(llm, prompts, sampling_params[0])
            else:
                raw_outputs = _generate(llm, prompts, sampling_params)
        except Exception as e:
            return {"error": f"generation failed: {e}"}

    by_index: Dict[int, Any] = {}
    for j, idx in enumerate(request_index):
        if j < len(raw_outputs):
            by_index[idx] = raw_outputs[j]

    results: List[Dict[str, Any]] = []
    for i, (mode, _prompt, hints, _sp) in enumerate(prepared):
        if mode == "error":
            results.append({"error": errors[i] or "unknown prepare error"})
            continue
        req_output = by_index.get(i)
        if req_output is None:
            results.append({"error": "no output produced"})
            continue
        results.append(_assemble_one(mode, req_output, hints, model_name))

    if is_batched:
        return {
            "model": model_name,
            "results": results,
            "count": len(results),
        }
    return results[0]




if __name__ == "__main__":
    if runpod is not None:
        runpod.serverless.start({"handler": handler})
