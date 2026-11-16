"""Offline retrieval evaluation. Does not assert LLM correctness."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from campusguide_rag import VectorStore


def evaluate(store: VectorStore, examples: list[dict], k: int, threshold: float) -> dict:
    positive = 0
    hits = 0
    negative = 0
    negative_refusals = 0
    for case in examples:
        results = store.search(case["question"], top_k=k, min_score=threshold)
        expected = case.get("expected_source")
        if expected:
            positive += 1
            matched = any(hit["source"] == expected for hit in results)
            hits += int(matched)
        else:
            negative += 1
            negative_refusals += int(not results)
        print(f"{'HIT' if expected and matched else 'MISS' if expected else 'REFUSE' if not results else 'RETRIEVED'}: {case['question']}")
    return {
        "evaluated_questions": len(examples),
        "positive_questions": positive,
        f"source_hit_at_{k}": round(hits / positive, 4) if positive else None,
        "negative_questions": negative,
        "retrieval_refusal_rate": round(negative_refusals / negative, 4) if negative else None,
        "note": "Measures retrieved source matching and empty retrieval only; not answer quality or hallucination rate.",
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation/demo_cases.json")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--index", default=os.getenv("VECTOR_STORE_PATH", "data/processed"))
    parser.add_argument("--backend", choices=["faiss", "numpy"], default=os.getenv("VECTOR_BACKEND", "faiss"))
    args = parser.parse_args()
    store = VectorStore(args.index, model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"), backend=args.backend)
    if not store.load():
        raise SystemExit("No index present. Run python scripts/ingest.py first.")
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    print(json.dumps(evaluate(store, cases, args.k, args.threshold), indent=2))


if __name__ == "__main__":
    main()
