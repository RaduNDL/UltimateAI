"""Run repeatable English quality checks against the currently running API."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DEFAULT_URL = "http://127.0.0.1:8000/chat"
DEFAULT_REPORT = ROOT / "evaluation-report.json"
BAD_PATTERNS = (
    "i do not have enough information",
    "as an ai language model",
    "i cannot assist",
    "i am unable to assist",
)
CASES = [
    {
        "name": "english-greeting",
        "payload": {"prompt": "Hello", "response_mode": "auto"},
        "expected_any": ["hello", "ultimateai", "software"],
    },
    {
        "name": "sql-injection",
        "payload": {
            "prompt": "How do parameterized SQL queries prevent SQL injection?",
            "response_mode": "auto",
            "max_new_tokens": 160,
        },
        "expected_any": ["parameter", "bind", "concatenat"],
    },
    {
        "name": "ecommerce-schema",
        "payload": {
            "prompt": "Design the core database tables for a small online store.",
            "response_mode": "auto",
            "max_new_tokens": 224,
        },
        "expected_any": ["product", "order", "user", "customer"],
    },
    {
        "name": "fastapi-cors",
        "payload": {
            "prompt": "My Angular app cannot call FastAPI because of CORS. What should I check?",
            "response_mode": "auto",
            "max_new_tokens": 160,
        },
        "expected_any": ["cors", "origin", "middleware"],
    },
    {
        "name": "typescript-undefined",
        "payload": {
            "prompt": "TypeScript says object is possibly undefined. What is the safe fix?",
            "response_mode": "auto",
            "max_new_tokens": 160,
        },
        "expected_any": ["undefined", "guard", "check", "null"],
    },
    {
        "name": "project-context",
        "payload": {
            "prompt": "In this project, which endpoint receives chat prompts?",
            "project": "ai-backend",
            "use_project_context": True,
            "response_mode": "auto",
            "max_new_tokens": 160,
        },
        "expected_any": ["chat", "endpoint", "app.py"],
    },
]


def assess(text: str, expected_any: list[str]) -> tuple[bool, list[str]]:
    normalized = text.casefold()
    problems: list[str] = []
    if len(text.strip()) < 40:
        problems.append("response is too short")
    if any(pattern in normalized for pattern in BAD_PATTERNS):
        problems.append("generic refusal")
    if len(set(re.findall(r"[a-z]{4,}", normalized))) < 5:
        problems.append("response is repetitive or low-information")
    if not any(term in normalized for term in expected_any):
        problems.append("missing expected technical concept")
    return not problems, problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a running base model or candidate adapter before activation."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Chat endpoint URL")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--minimum-passes", type=int, default=5)
    args = parser.parse_args()

    results = []
    passed = 0
    for case in CASES:
        try:
            response = requests.post(args.url, json=case["payload"], timeout=args.timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            raise SystemExit(
                f"Cannot reach {args.url}. Start the backend first, then retry."
            ) from error
        payload = response.json()
        answer = str(payload.get("response", ""))
        ok, problems = assess(answer, case["expected_any"])
        passed += int(ok)
        results.append(
            {
                "name": case["name"],
                "passed": ok,
                "problems": problems,
                "mode": payload.get("mode"),
                "sources": payload.get("sources", []),
                "response": answer,
            }
        )
        print(f"{'PASS' if ok else 'FAIL'}: {case['name']}")

    report = {
        "passed": passed,
        "total": len(CASES),
        "minimum_passes": args.minimum_passes,
        "approved": passed >= args.minimum_passes,
        "cases": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "total", "approved")}, indent=2))
    print(f"Saved report to {args.output}")
    if not report["approved"]:
        raise SystemExit("Quality gate failed. Do not activate this adapter.")


if __name__ == "__main__":
    main()
