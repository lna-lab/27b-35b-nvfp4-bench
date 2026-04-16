#!/usr/bin/env python3
"""Dense 27B/31B NVFP4 Comparative Benchmark — Lna-Lab

4 models: Qwen3.5-27B, Qwopus3.5-27B, Gemma4-31B (Lna-Lab), Gemma4-31B (lyf baseline)
Tests: English, Japanese, Math, Coding, Design + throughput scaling

Usage:
    python bench.py                              # full run
    python bench.py --tests coding math          # specific tests
    python bench.py --models qwen qwopus         # specific models
"""

import argparse
import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Any

import aiohttp

# ── Models ───────────────────────────────────────────────────────────────────

MODELS = {
    "qwen": {
        "name": "Huihui-Qwen3.5-27B-abliterated-NVFP4",
        "url": "http://localhost:8016",
        "served_name": "current",
        "desc": "Qwen3.5 27B Dense + MTP (Lna-Lab, TQ-ready)",
    },
    "qwopus": {
        "name": "Huihui-Qwopus3.5-27B-v3-abliterated-NVFP4",
        "url": "http://localhost:8017",
        "served_name": "current",
        "desc": "Qwopus 27B Opus-distilled (Lna-Lab, TQ-ready)",
    },
    "gemma31": {
        "name": "Huihui-gemma-4-31B-it-abliterated-v2-NVFP4",
        "url": "http://localhost:8018",
        "served_name": "current",
        "desc": "Gemma4 31B Dense (Lna-Lab, TQ-ready)",
    },
    "gemma31_lyf": {
        "name": "lyf-Huihui-gemma-4-31B-it-abliterated-v2-NVFP4",
        "url": "http://localhost:8019",
        "served_name": "current",
        "desc": "Gemma4 31B Dense (lyf, no-TQ baseline)",
    },
}

# ── Test Prompts ─────────────────────────────────────────────────────────────

TESTS = {
    "english_critique": {
        "label": "English Long-Context Critique",
        "max_tokens": 4096,
        "prompts": [
            "Write a 2000-word critical essay analyzing the philosophical implications of artificial general intelligence. "
            "Cover: (1) The alignment problem and why it's harder than it appears, "
            "(2) The economic disruption thesis vs. the augmentation thesis, "
            "(3) Consciousness and moral status of AI systems, "
            "(4) Historical parallels with previous technological revolutions. "
            "Use clear academic prose with proper paragraph structure, counterarguments, and a nuanced conclusion.",

            "Compose a detailed literary analysis (1500+ words) comparing the narrative techniques in "
            "George Orwell's '1984' and Aldous Huxley's 'Brave New World'. "
            "Examine: surveillance vs. pleasure as control mechanisms, the role of language manipulation, "
            "prophetic accuracy in the modern era, and which dystopia more closely resembles our present. "
            "Include direct textual references and scholarly perspective.",
        ],
    },
    "japanese": {
        "label": "Japanese Expression (多言語表現力)",
        "max_tokens": 2048,
        "prompts": [
            "以下のテーマで、文学的な日本語で800字以上のエッセイを書いてください。\n\n"
            "テーマ：「AIと人間の共創は、新しい文化を生むか」\n\n"
            "条件：\n"
            "- 具体的な事例を2つ以上含める\n"
            "- 季語や比喩表現を自然に織り交ぜる\n"
            "- 結論は断定せず、余韻を残す形で終える\n"
            "- 敬体ではなく常体（だ・である調）で書く",

            "日本語で以下の技術文書を書いてください。\n\n"
            "題目：「大規模言語モデルの量子化技術：NVFP4の原理と実践」\n\n"
            "対象読者：機械学習エンジニア（日本語話者）\n"
            "構成：1.背景 2.FP4量子化の数学的原理 3.MoEモデルへの適用課題 4.実装手順 5.ベンチマーク結果の読み方\n"
            "分量：各セクション200字以上、専門用語には簡潔な説明を添える",
        ],
    },
    "math": {
        "label": "Mathematical Reasoning",
        "max_tokens": 2048,
        "prompts": [
            "Solve this step by step, showing all work:\n\n"
            "A factory produces widgets with a 3% defect rate. An inspector checks widgets using a test "
            "that correctly identifies 95% of defective widgets (sensitivity) and correctly passes 98% "
            "of good widgets (specificity).\n\n"
            "(a) If a widget fails the test, what is the probability it is actually defective? (Bayes' theorem)\n"
            "(b) If the factory improves to a 0.5% defect rate, how does this probability change?\n"
            "(c) The factory wants P(defective | fails test) >= 0.80. What defect rate threshold achieves this?\n"
            "Show the algebraic setup and numerical computation for each part.",

            "Prove that for any positive integer n:\n"
            "sum_{k=0}^{n} C(n,k)^2 = C(2n, n)\n\n"
            "where C(n,k) is the binomial coefficient.\n\n"
            "Provide two different proofs:\n"
            "(1) A combinatorial/counting argument\n"
            "(2) Using generating functions\n"
            "Then compute the exact value for n=10 and verify.",
        ],
    },
    "coding": {
        "label": "Coding Ability",
        "max_tokens": 4096,
        "prompts": [
            "Implement a complete Python module for a concurrent task scheduler with the following requirements:\n\n"
            "1. `TaskScheduler` class with `submit(task, priority, dependencies)` method\n"
            "2. Tasks execute in priority order, respecting dependency DAG\n"
            "3. Configurable max concurrency via asyncio Semaphore\n"
            "4. Cycle detection in dependency graph\n"
            "5. Progress callback support\n"
            "6. Proper error propagation (if a dependency fails, dependents are cancelled)\n\n"
            "Include: type hints, docstrings, and at least 3 test cases using pytest-style assertions.\n"
            "The code should be production-ready, not a sketch.",

            "Write a Python implementation of a B-Tree (order m=4) with:\n"
            "- insert(key, value)\n"
            "- search(key) -> value\n"
            "- delete(key)\n"
            "- range_query(low, high) -> list of (key, value)\n\n"
            "Include proper node splitting, merging, and rebalancing.\n"
            "Add type hints and test with at least 5 assertions covering edge cases.",
        ],
    },
    "design": {
        "label": "System Design Ability",
        "max_tokens": 4096,
        "prompts": [
            "Design a real-time collaborative document editing system (like Google Docs) that supports:\n\n"
            "- 1000 concurrent users per document\n"
            "- Operational Transform or CRDT-based conflict resolution\n"
            "- Offline editing with sync on reconnect\n"
            "- Cursor presence and selection highlighting\n"
            "- Version history with branching\n\n"
            "Provide: (1) High-level architecture diagram in ASCII, (2) Data model, "
            "(3) Conflict resolution algorithm choice with justification, "
            "(4) Scaling strategy, (5) Failure modes and mitigation.\n"
            "Be specific about technology choices and trade-offs.",

            "Design an inference serving platform for hosting 50+ LLM variants with:\n\n"
            "- Dynamic model loading/unloading based on demand\n"
            "- Request routing with model-specific SLOs\n"
            "- GPU memory management across models\n"
            "- A/B testing and canary deployments\n"
            "- Cost optimization (spot instances, batching, quantization selection)\n\n"
            "Provide architecture, API design, scheduling algorithm, and capacity planning approach.\n"
            "Include concrete numbers for a 100-GPU cluster serving 10K req/s.",
        ],
    },
    "tool_call": {
        "label": "Tool-Call Accuracy",
        "max_tokens": 1024,
        "prompts": "SPECIAL",  # handled separately
    },
}

TOOL_CALL_SCENARIOS = [
    {
        "messages": [
            {"role": "system", "content": (
                "You are a helpful assistant. You have access to the following tools. "
                "When appropriate, call a tool by responding with a JSON tool call."
            )},
            {"role": "user", "content": "What's the current weather in San Francisco and New York?"},
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                }, "required": ["city"]},
            }},
        ],
        "eval": lambda tc, c: (
            2 if len(tc) >= 2 else (1 if len(tc) == 1 else 0),
            "multi_call" if len(tc) >= 2 else ("single_call" if len(tc) == 1 else "no_call")
        ),
    },
    {
        "messages": [
            {"role": "system", "content": "You have tools: calculator, web_search, send_email. Only use what's needed."},
            {"role": "user", "content": "Calculate the compound interest on $10,000 at 5% for 10 years, compounded monthly."},
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "calculator", "description": "Evaluate a math expression",
                "parameters": {"type": "object", "properties": {
                    "expression": {"type": "string"}
                }, "required": ["expression"]},
            }},
            {"type": "function", "function": {
                "name": "web_search", "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            }},
            {"type": "function", "function": {
                "name": "send_email", "description": "Send an email",
                "parameters": {"type": "object", "properties": {
                    "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"},
                }, "required": ["to", "subject", "body"]},
            }},
        ],
        "eval": lambda tc, c: (
            2 if (tc and tc[0].get("function", {}).get("name") == "calculator") else
            (1 if tc else (0.5 if "calculator" in (c or "").lower() or "10000" in (c or "") else 0)),
            f"fn:{tc[0]['function']['name']}" if tc else ("text_answer" if c else "empty"),
        ),
    },
    {
        "messages": [
            {"role": "system", "content": (
                "You are a coding assistant with access to tools. "
                "Use run_code to execute Python, and file_write to save files."
            )},
            {"role": "user", "content": (
                "Write a Python script that generates a Fibonacci sequence up to 1000, "
                "save it to fib.py, then run it and show the output."
            )},
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "run_code", "description": "Execute Python code and return stdout/stderr",
                "parameters": {"type": "object", "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                }, "required": ["code"]},
            }},
            {"type": "function", "function": {
                "name": "file_write", "description": "Write content to a file",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"},
                }, "required": ["path", "content"]},
            }},
        ],
        "eval": lambda tc, c: (
            2 if len(tc) >= 2 else (1.5 if len(tc) == 1 else (0.5 if "fibonacci" in (c or "").lower() else 0)),
            f"calls:{len(tc)}" if tc else "text_only",
        ),
    },
]


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass
class Result:
    model: str
    test: str
    prompt_idx: int
    concurrency: int
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0
    tok_per_sec: float = 0
    content: str = ""
    tool_calls: list = None
    error: str = ""
    quality_score: float = 0.0
    quality_notes: str = ""


# ── API ──────────────────────────────────────────────────────────────────────

async def call_api(
    session: aiohttp.ClientSession, base_url: str, model_name: str,
    messages: list, max_tokens: int, tools: list | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "model": model_name, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0.3,
    }
    if tools:
        payload["tools"] = tools

    start = time.monotonic()
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            data = await resp.json()
            latency = (time.monotonic() - start) * 1000
            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            msg = choice.get("message", {})
            return {
                "content": msg.get("content", "") or "",
                "tool_calls": msg.get("tool_calls") or [],
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "latency_ms": latency,
                "error": "",
            }
    except Exception as e:
        return {"content": "", "tool_calls": [], "tokens_in": 0, "tokens_out": 0,
                "latency_ms": (time.monotonic() - start) * 1000, "error": str(e)}


# ── Quality evaluators ───────────────────────────────────────────────────────

def eval_text(content: str, min_words: int = 200) -> tuple[float, str]:
    """Generic text quality: length, structure, vocabulary, coherence."""
    if not content.strip():
        return 0.0, "empty"
    words = content.split()
    paragraphs = [p for p in content.split("\n\n") if p.strip()]
    unique = len(set(w.lower() for w in words)) / max(len(words), 1)

    s = 0.0
    n = []
    # Length (0-3)
    ratio = min(len(words) / max(min_words, 1), 1.5)
    s += min(ratio * 2, 3); n.append(f"words:{len(words)}")
    # Structure (0-3)
    s += min(len(paragraphs) * 0.75, 3); n.append(f"para:{len(paragraphs)}")
    # Vocab (0-2)
    s += min(unique * 4, 2); n.append(f"uniq:{unique:.2f}")
    # Coherence — penalize repetition (0-2)
    lines = [l.strip().lower() for l in content.split("\n") if l.strip()]
    dups = len(lines) - len(set(lines))
    s += max(2 - dups * 0.5, 0); n.append(f"dups:{dups}")
    return s / 10, "|".join(n)


def eval_code_quality(content: str) -> tuple[float, str]:
    if not content.strip():
        return 0.0, "empty"
    s = 0.0; n = []
    if "def " in content or "class " in content:
        s += 2; n.append("has_def")
    if "->" in content or ": " in content:
        s += 1.5; n.append("typed")
    if '"""' in content or "'''" in content:
        s += 1; n.append("docstring")
    if "assert " in content or "test" in content.lower():
        s += 1.5; n.append("tests")
    if "try:" in content or "raise " in content:
        s += 1; n.append("error_handling")
    if "async " in content or "await " in content:
        s += 1; n.append("async")
    if len(content.split("\n")) >= 30:
        s += 1; n.append("substantial")
    if "import " in content:
        s += 1; n.append("imports")
    return min(s / 10, 1.0), "|".join(n)


def eval_math(content: str) -> tuple[float, str]:
    if not content.strip():
        return 0.0, "empty"
    s = 0.0; n = []
    indicators = {
        "step": ("step" in content.lower(), 2),
        "formula": (any(x in content for x in ["=", "P(", "C(", "sum", "∑"]), 2),
        "numbers": (any(c.isdigit() for c in content), 1),
        "therefore": (any(x in content.lower() for x in ["therefore", "thus", "hence", "so the answer"]), 1.5),
        "proof": (any(x in content.lower() for x in ["proof", "qed", "proven", "we show"]), 2),
        "structured": (len([p for p in content.split("\n\n") if p.strip()]) >= 2, 1.5),
    }
    for k, (hit, pts) in indicators.items():
        if hit:
            s += pts; n.append(k)
    return min(s / 10, 1.0), "|".join(n)


def eval_tool(tool_calls: list, content: str, scenario: dict) -> tuple[float, str]:
    score, note = scenario["eval"](tool_calls, content)
    return score / 2, note  # normalize to 0-1


# ── VRAM monitor ─────────────────────────────────────────────────────────────

def get_vram_usage() -> dict[int, int]:
    """Returns {gpu_id: used_mb}"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        result = {}
        for line in out.strip().split("\n"):
            parts = line.split(",")
            result[int(parts[0].strip())] = int(parts[1].strip())
        return result
    except Exception:
        return {}


# ── Runner ───────────────────────────────────────────────────────────────────

async def run_test(
    session: aiohttp.ClientSession, model_key: str,
    test_name: str, prompt_idx: int, concurrency: int,
) -> Result:
    m = MODELS[model_key]
    r = Result(model=model_key, test=test_name, prompt_idx=prompt_idx, concurrency=concurrency)

    if test_name == "tool_call":
        sc = TOOL_CALL_SCENARIOS[prompt_idx]
        resp = await call_api(session, m["url"], m["served_name"],
                              sc["messages"], 1024, sc["tools"])
        r.tool_calls = resp["tool_calls"]
        r.quality_score, r.quality_notes = eval_tool(resp["tool_calls"], resp["content"], sc)
    else:
        test = TESTS[test_name]
        messages = [{"role": "user", "content": test["prompts"][prompt_idx]}]
        resp = await call_api(session, m["url"], m["served_name"],
                              messages, test["max_tokens"])

        if test_name == "english_critique":
            r.quality_score, r.quality_notes = eval_text(resp["content"], 400)
        elif test_name == "japanese":
            r.quality_score, r.quality_notes = eval_text(resp["content"], 200)
        elif test_name == "math":
            r.quality_score, r.quality_notes = eval_math(resp["content"])
        elif test_name == "coding":
            r.quality_score, r.quality_notes = eval_code_quality(resp["content"])
        elif test_name == "design":
            r.quality_score, r.quality_notes = eval_text(resp["content"], 300)

    r.content = resp["content"]
    r.tokens_in = resp["tokens_in"]
    r.tokens_out = resp["tokens_out"]
    r.latency_ms = resp["latency_ms"]
    r.tok_per_sec = (resp["tokens_out"] / (resp["latency_ms"] / 1000)
                     if resp["latency_ms"] > 0 and resp["tokens_out"] > 0 else 0)
    r.error = resp["error"]
    return r


async def run_batch(model_key: str, test_name: str, concurrency: int) -> list[Result]:
    num_prompts = len(TOOL_CALL_SCENARIOS) if test_name == "tool_call" else len(TESTS[test_name]["prompts"])

    async with aiohttp.ClientSession() as session:
        # Warmup
        m = MODELS[model_key]
        await call_api(session, m["url"], m["served_name"],
                       [{"role": "user", "content": "Hi"}], 8)

        tasks = []
        for pidx in range(num_prompts):
            for _ in range(concurrency):
                tasks.append(run_test(session, model_key, test_name, pidx, concurrency))
        return list(await asyncio.gather(*tasks))


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen", "qwopus", "gemma31", "gemma31_lyf"])
    parser.add_argument("--tests", nargs="+",
                        default=["english_critique", "japanese", "math", "coding", "design", "tool_call"])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--output", default="results/benchmark_v2.json")
    args = parser.parse_args()

    test_names = {
        "english_critique": "English Critique",
        "japanese": "Japanese Expression",
        "math": "Math Reasoning",
        "coding": "Coding",
        "design": "System Design",
        "tool_call": "Tool Calls",
    }

    print("=" * 70)
    print("  Dense 27B/31B NVFP4 Comparative Benchmark — Lna-Lab")
    print("  Context: 128K | Single NVIDIA RTX PRO 6000 Blackwell per model")
    print("=" * 70)

    # Health check
    async with aiohttp.ClientSession() as session:
        for mk in args.models:
            try:
                async with session.get(
                    f"{MODELS[mk]['url']}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    print(f"  [{mk}] {'UP' if r.status == 200 else f'ERR {r.status}'}")
            except Exception as e:
                print(f"  [{mk}] DOWN — {e}")
                return

    # VRAM baseline
    vram_before = get_vram_usage()
    print(f"\n  VRAM: GPU0={vram_before.get(0,0)}MB  GPU1={vram_before.get(1,0)}MB  GPU2={vram_before.get(2,0)}MB")
    print("=" * 70)

    all_results = []
    summary_rows = []

    for model_key in args.models:
        for test_name in args.tests:
            for conc in args.concurrency:
                label = f"[{model_key}] {test_names.get(test_name, test_name)} x{conc}"
                print(f"\n{label} ...", end="", flush=True)

                results = await run_batch(model_key, test_name, conc)

                scores = [r.quality_score for r in results if not r.error]
                speeds = [r.tok_per_sec for r in results if r.tok_per_sec > 0]
                tokens_out = [r.tokens_out for r in results if r.tokens_out > 0]
                errors = [r for r in results if r.error]

                avg_score = sum(scores) / len(scores) if scores else 0
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
                total_tps = sum(speeds)
                avg_tokens = sum(tokens_out) / len(tokens_out) if tokens_out else 0

                print(f"  Q={avg_score:.2f}  {avg_speed:.0f}tps  Σ{total_tps:.0f}tps  "
                      f"~{avg_tokens:.0f}tok  err={len(errors)}")

                summary_rows.append({
                    "model": model_key, "test": test_name, "concurrency": conc,
                    "avg_quality": round(avg_score, 3),
                    "avg_tok_per_sec": round(avg_speed, 1),
                    "total_throughput": round(total_tps, 1),
                    "avg_tokens_out": round(avg_tokens),
                    "errors": len(errors),
                })

                for r in results:
                    d = asdict(r)
                    d.pop("tool_calls", None)  # not JSON-safe
                    d["content"] = d["content"][:500]  # truncate for storage
                    all_results.append(d)

    # VRAM after
    vram_after = get_vram_usage()

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "gpu": "NVIDIA RTX PRO 6000 Blackwell (96GB)",
            "context_length": 131072,
            "cuda_graph_mode": "piecewise",
            "models": {k: MODELS[k] for k in args.models},
            "vram_mb": {
                "before": vram_before,
                "after": vram_after,
            },
        },
        "summary": summary_rows,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'Test':<20} {'Conc':>4} {'Quality':>8} {'tok/s':>7} {'Total':>8} {'Tokens':>7}")
    print("-" * 70)
    for row in summary_rows:
        print(f"{row['model']:<12} {row['test']:<20} {row['concurrency']:>4} "
              f"{row['avg_quality']:>8.3f} {row['avg_tok_per_sec']:>7.1f} "
              f"{row['total_throughput']:>8.1f} {row['avg_tokens_out']:>7}")
    print("=" * 70)
    print(f"VRAM: GPU0={vram_after.get(0,0)}MB  GPU1={vram_after.get(1,0)}MB  GPU2={vram_after.get(2,0)}MB")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
