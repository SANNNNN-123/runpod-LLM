"""CPU-runnable test for handler.py.

All heavyweight imports (`vllm`, `transformers`, `torch`, `outlines`) are stubbed
via ``sys.modules`` BEFORE we import the handler module so it loads cleanly on
machines without a GPU. The fake vLLM ``LLM`` returns a deterministic echo of
the last user message (or of the raw prompt), capped at ``max_tokens`` tokens.
"""

import json
import os
import sys
import types
import unittest
from typing import Any, Dict, List, Optional




def _install_stubs() -> Dict[str, Any]:
    """Build and install the fake modules. Returns a dict of refs for tests."""
    refs: Dict[str, Any] = {}

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    sys.modules["torch"] = torch_mod

    outlines_mod = types.ModuleType("outlines")
    sys.modules["outlines"] = outlines_mod

    class FakeTokenizer:
        """Renders chat messages to a simple deterministic prompt and tokenizes
        by whitespace."""

        def __init__(self, name: str):
            self.name_or_path = name
            self.apply_chat_template_calls: List[Dict[str, Any]] = []

        def apply_chat_template(
            self,
            messages: List[Dict[str, str]],
            tokenize: bool = False,
            add_generation_prompt: bool = True,
            **_: Any,
        ) -> str:
            self.apply_chat_template_calls.append(
                {
                    "messages": list(messages),
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                }
            )
            parts: List[str] = []
            for m in messages:
                parts.append(
                    "<|{role}|>\n{content}".format(
                        role=m.get("role", "user"),
                        content=m.get("content", ""),
                    )
                )
            if add_generation_prompt:
                parts.append("<|assistant|>\n")
            text = "\n".join(parts)
            return text

        def encode(self, text: str, **_: Any) -> List[int]:
            return [hash(w) & 0xFFFF for w in text.split()]

    transformers_mod = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **_: Any) -> FakeTokenizer:
            tok = FakeTokenizer(name)
            refs["last_tokenizer"] = tok
            return tok

    transformers_mod.AutoTokenizer = _AutoTokenizer
    sys.modules["transformers"] = transformers_mod
    refs["transformers"] = transformers_mod

    class FakeCompletionOutput:
        def __init__(self, text: str, token_ids: List[int], finish_reason: str):
            self.text = text
            self.token_ids = token_ids
            self.finish_reason = finish_reason
            self.index = 0

    class FakeRequestOutput:
        def __init__(self, prompt: str, prompt_token_ids: List[int],
                     outputs: List[FakeCompletionOutput], request_id: str):
            self.prompt = prompt
            self.prompt_token_ids = prompt_token_ids
            self.outputs = outputs
            self.request_id = request_id

    class FakeSamplingParams:
        """Records constructor kwargs verbatim so tests can assert on them."""

        _ACCEPTED = {
            "temperature", "top_p", "top_k", "max_tokens", "stop", "seed",
            "presence_penalty", "frequency_penalty", "repetition_penalty",
            "best_of", "n", "guided_json", "guided_regex",
        }

        def __init__(self, **kwargs: Any):
            unknown = set(kwargs) - self._ACCEPTED
            if unknown:
                raise TypeError(
                    f"unexpected kwargs: {sorted(unknown)}"
                )
            for k, v in kwargs.items():
                setattr(self, k, v)
            self._kwargs = dict(kwargs)
            refs.setdefault("sampling_history", []).append(dict(kwargs))

    class FakeLLM:
        """Deterministic echo LLM.

        For each prompt:
          * Locate the last user message in the chat-template prompt (if any)
          * Otherwise, echo the prompt verbatim
          * Honor ``max_tokens`` (whitespace tokens) and ``stop`` sequences
          * If ``guided_json`` is set, return a JSON object echoing the request
        """

        construction_count = 0

        def __init__(self, **kwargs: Any):
            FakeLLM.construction_count += 1
            self.kwargs = kwargs
            refs["last_llm_kwargs"] = kwargs

        @staticmethod
        def _extract_user(prompt: str) -> str:
            marker = "<|user|>\n"
            if marker in prompt:
                last = prompt.rsplit(marker, 1)[-1]
                return last.split("\n<|", 1)[0]
            return prompt

        def generate(
            self,
            prompts: Any,
            sampling_params: Any,
        ) -> List[FakeRequestOutput]:
            single = isinstance(prompts, str)
            prompt_list: List[str] = [prompts] if single else list(prompts)
            if isinstance(sampling_params, list):
                sp_list = sampling_params
            else:
                sp_list = [sampling_params] * len(prompt_list)

            outs: List[FakeRequestOutput] = []
            for i, prompt in enumerate(prompt_list):
                sp = sp_list[i]
                max_tokens = int(getattr(sp, "max_tokens", 64))
                stop = getattr(sp, "stop", None) or []
                guided_json = getattr(sp, "guided_json", None)

                if guided_json is not None:
                    obj = {"echo": self._extract_user(prompt), "ok": True}
                    text = json.dumps(obj)
                    finish = "stop"
                else:
                    user = self._extract_user(prompt)
                    words = user.strip().split()
                    truncated_by_max = False
                    if len(words) > max_tokens:
                        words = words[:max_tokens]
                        truncated_by_max = True
                    text = " ".join(words)
                    finish = "length" if truncated_by_max else "stop"
                    for s in stop:
                        idx = text.find(s)
                        if idx >= 0:
                            text = text[:idx]
                            finish = "stop"

                token_ids = list(range(len(text.split())))
                prompt_tokens = list(range(len(prompt.split())))
                outs.append(
                    FakeRequestOutput(
                        prompt=prompt,
                        prompt_token_ids=prompt_tokens,
                        outputs=[FakeCompletionOutput(text, token_ids, finish)],
                        request_id=f"req-{i}",
                    )
                )
            return outs

    vllm_mod = types.ModuleType("vllm")
    vllm_mod.LLM = FakeLLM
    vllm_mod.SamplingParams = FakeSamplingParams
    sys.modules["vllm"] = vllm_mod

    vllm_outputs_mod = types.ModuleType("vllm.outputs")
    vllm_outputs_mod.RequestOutput = FakeRequestOutput
    vllm_outputs_mod.CompletionOutput = FakeCompletionOutput
    sys.modules["vllm.outputs"] = vllm_outputs_mod

    refs["FakeLLM"] = FakeLLM
    refs["FakeSamplingParams"] = FakeSamplingParams
    refs["FakeRequestOutput"] = FakeRequestOutput
    refs["FakeCompletionOutput"] = FakeCompletionOutput
    return refs


REFS = _install_stubs()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handler as h


def _reset_engine() -> None:
    """Tests must run independently — clear cached LLM instance between tests."""
    h._LLM_INSTANCE = None
    REFS["FakeLLM"].construction_count = 0
    REFS["sampling_history"] = []




class SamplingParamsTests(unittest.TestCase):
    def test_defaults_passed_through(self):
        _reset_engine()
        sp = h.build_sampling_params({})
        self.assertEqual(sp.temperature, 0.7)
        self.assertEqual(sp.top_p, 1.0)
        self.assertEqual(sp.top_k, -1)
        self.assertEqual(sp.max_tokens, 512)
        self.assertFalse(hasattr(sp, "stop"))

    def test_overrides_and_stop_coercion(self):
        _reset_engine()
        sp = h.build_sampling_params(
            {"temperature": 0.1, "max_tokens": 32, "stop": "STOP"}
        )
        self.assertAlmostEqual(sp.temperature, 0.1)
        self.assertEqual(sp.max_tokens, 32)
        self.assertEqual(sp.stop, ["STOP"])

    def test_best_of_lifted_to_n(self):
        _reset_engine()
        sp = h.build_sampling_params({"n": 3, "best_of": 1})
        self.assertEqual(sp.n, 3)
        self.assertGreaterEqual(sp.best_of, sp.n)

    def test_guided_json_forwarded(self):
        _reset_engine()
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        sp = h.build_sampling_params({}, guided_json=schema)
        self.assertEqual(getattr(sp, "guided_json", None), schema)


class ChatTemplateTests(unittest.TestCase):
    def test_chat_uses_template_and_returns_message(self):
        _reset_engine()
        event = {
            "input": {
                "mode": "chat",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hello world how are you"},
                ],
                "max_tokens": 16,
            }
        }
        out = h.handler(event)
        self.assertNotIn("error", out)
        self.assertIn("choices", out)
        self.assertEqual(out["choices"][0]["message"]["role"], "assistant")
        self.assertIn("hello", out["choices"][0]["message"]["content"])
        tok = REFS["last_tokenizer"]
        self.assertGreaterEqual(len(tok.apply_chat_template_calls), 1)
        call = tok.apply_chat_template_calls[-1]
        self.assertTrue(call["add_generation_prompt"])
        roles = [m["role"] for m in call["messages"]]
        self.assertIn("system", roles)
        self.assertIn("user", roles)

    def test_multi_turn_keeps_history(self):
        _reset_engine()
        event = {
            "input": {
                "mode": "chat",
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "second message"},
                ],
                "max_tokens": 8,
            }
        }
        out = h.handler(event)
        self.assertNotIn("error", out)
        self.assertIn("second", out["choices"][0]["message"]["content"])


class CompletionModeTests(unittest.TestCase):
    def test_completion_returns_text_field(self):
        _reset_engine()
        event = {
            "input": {
                "mode": "complete",
                "prompt": "Once upon a time there was",
                "max_tokens": 4,
            }
        }
        out = h.handler(event)
        self.assertNotIn("error", out)
        choice = out["choices"][0]
        self.assertIn("text", choice)
        self.assertNotIn("message", choice)
        self.assertEqual(len(choice["text"].split()), 4)
        self.assertEqual(choice["finish_reason"], "length")


class JsonModeTests(unittest.TestCase):
    def test_json_mode_parses_output(self):
        _reset_engine()
        schema = {
            "type": "object",
            "properties": {
                "echo": {"type": "string"},
                "ok": {"type": "boolean"},
            },
        }
        event = {
            "input": {
                "mode": "json",
                "messages": [{"role": "user", "content": "give me JSON"}],
                "schema": schema,
                "max_tokens": 64,
            }
        }
        out = h.handler(event)
        self.assertNotIn("error", out)
        self.assertIn("json", out["choices"][0])
        self.assertEqual(out["choices"][0]["json"]["ok"], True)
        last_sp = REFS["sampling_history"][-1]
        self.assertEqual(last_sp.get("guided_json"), schema)

    def test_json_loose_parser_strips_fences(self):
        obj, err = h.parse_json_loose("```json\n{\"a\": 1}\n```")
        self.assertIsNone(err)
        self.assertEqual(obj, {"a": 1})


class StopSequenceTests(unittest.TestCase):
    def test_stop_truncates_output(self):
        _reset_engine()
        event = {
            "input": {
                "mode": "complete",
                "prompt": "alpha beta STOP gamma delta",
                "max_tokens": 32,
                "stop": ["STOP"],
            }
        }
        out = h.handler(event)
        self.assertNotIn("error", out)
        text = out["choices"][0]["text"]
        self.assertIn("alpha", text)
        self.assertNotIn("STOP", text)
        self.assertNotIn("gamma", text)
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")


class BatchedRequestTests(unittest.TestCase):
    def test_batched_requests_list(self):
        _reset_engine()
        event = {
            "input": {
                "requests": [
                    {
                        "mode": "chat",
                        "messages": [{"role": "user", "content": "one"}],
                        "max_tokens": 4,
                    },
                    {
                        "mode": "complete",
                        "prompt": "two two two",
                        "max_tokens": 2,
                    },
                    {
                        "mode": "json",
                        "messages": [{"role": "user", "content": "three"}],
                        "schema": {"type": "object"},
                        "max_tokens": 32,
                    },
                ]
            }
        }
        out = h.handler(event)
        self.assertNotIn("error", out)
        self.assertEqual(out["count"], 3)
        self.assertEqual(len(out["results"]), 3)
        self.assertIn("message", out["results"][0]["choices"][0])
        self.assertIn("text", out["results"][1]["choices"][0])
        self.assertIn("json", out["results"][2]["choices"][0])


class UsageCountingTests(unittest.TestCase):
    def test_usage_counts_tokens(self):
        _reset_engine()
        event = {
            "input": {
                "mode": "complete",
                "prompt": "alpha beta gamma delta",
                "max_tokens": 3,
            }
        }
        out = h.handler(event)
        self.assertNotIn("error", out)
        usage = out["usage"]
        self.assertEqual(usage["completion_tokens"], 3)
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertEqual(
            usage["total_tokens"],
            usage["prompt_tokens"] + usage["completion_tokens"],
        )


class ToolChoiceTests(unittest.TestCase):
    def test_tool_choice_required_returns_tool_calls(self):
        _reset_engine()
        original_generate = REFS["FakeLLM"].generate

        def _tool_generate(self, prompts, sampling_params):
            outs = original_generate(self, prompts, sampling_params)
            payload = json.dumps(
                {
                    "tool_calls": [
                        {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        }
                    ]
                }
            )
            outs[0].outputs[0].text = payload
            outs[0].outputs[0].token_ids = list(range(len(payload.split())))
            return outs

        REFS["FakeLLM"].generate = _tool_generate
        try:
            event = {
                "input": {
                    "mode": "chat",
                    "messages": [
                        {"role": "user", "content": "what's the weather in Paris?"}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Look up weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "city": {"type": "string"}
                                    },
                                    "required": ["city"],
                                },
                            },
                        }
                    ],
                    "tool_choice": "required",
                    "max_tokens": 64,
                }
            }
            out = h.handler(event)
            self.assertNotIn("error", out)
            msg = out["choices"][0]["message"]
            self.assertIn("tool_calls", msg)
            self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")
            args_obj = json.loads(msg["tool_calls"][0]["function"]["arguments"])
            self.assertEqual(args_obj["city"], "Paris")
        finally:
            REFS["FakeLLM"].generate = original_generate


class ModelSelectionTests(unittest.TestCase):
    def test_disallowed_model_raises(self):
        _reset_engine()
        event = {
            "input": {
                "model": "not-a-real-model",
                "mode": "chat",
                "messages": [{"role": "user", "content": "hi"}],
            }
        }
        out = h.handler(event)
        self.assertIn("error", out)
        self.assertIn("not in the curated allow-list", out["error"])

    def test_refuses_model_swap_at_runtime(self):
        _reset_engine()
        h.handler(
            {
                "input": {
                    "mode": "chat",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            }
        )
        event = {
            "input": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "mode": "chat",
                "messages": [{"role": "user", "content": "again"}],
            }
        }
        out = h.handler(event)
        self.assertIn("error", out)
        self.assertIn("bound to model", out["error"])


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("TESTS FAILED")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
