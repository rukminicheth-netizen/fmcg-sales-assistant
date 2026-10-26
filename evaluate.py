"""
TESTING – runs the 10 test questions and saves the results.

Run it with:
    python evaluate.py

It writes eval/results.md (a table you can read and paste into the README).
You then fill in the "Verdict" and "Why" columns yourself after reading each answer:
the point is an honest human judgement, not an automatic score.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from rag import Generator, Retriever, answer_question

load_dotenv()
EVAL_DIR = Path(__file__).parent / "eval"
PAUSE_SECONDS = 7  # the free Gemini tier allows only a few requests per minute


def short(text, n=350):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Add GEMINI_API_KEY to your .env file first.")

    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    retriever = Retriever()
    generator = Generator(api_key)

    rows = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['question']}")
        result = answer_question(q["question"], retriever, generator)
        cited = [result["chunks"][n - 1] for n in result["cited"]]
        rows.append({
            "id": i,
            "question": q["question"],
            "type": q["type"],
            "expected": q["expected"],
            "answer": result["answer"],
            "found": result["found"],
            "model": result.get("model"),
            "cited_sources": [f"{c['title']} p.{c['page']}" for c in cited],
            "retrieved_sources": [f"{c['title']} p.{c['page']}" for c in result["chunks"]],
        })
        if i < len(questions):
            time.sleep(PAUSE_SECONDS)

    (EVAL_DIR / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "| # | Question | Type | Answer (short) | Cited sources | Verdict | Why |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        answer = short(r["answer"]).replace("|", "/").replace("\n", " ")
        cites = "; ".join(r["cited_sources"]) or "—"
        lines.append(f"| {r['id']} | {r['question']} | {r['type']} | {answer} | {cites} |  |  |")
    (EVAL_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {len(rows)} results to eval/results.md and eval/results.json")


if __name__ == "__main__":
    main()
