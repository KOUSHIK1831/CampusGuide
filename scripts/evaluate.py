"""Compare offline evidence retrieval on a versioned benchmark manifest."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campusguide_rag import VectorStore, chunk_documents
from campusguide_rag.ingestion import MAX_FILE_BYTES, load_document_bytes
from campusguide_rag.retrieval import CrossEncoderReranker, DEFAULT_RERANKER_MODEL, Reranker


CASE_KINDS = {"answerable", "negative", "follow_up", "conflict"}
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _text(data: dict, key: str) -> bool:
    value = data.get(key)
    return isinstance(value, str) and bool(value.strip())


def _valid_evidence(evidence: object) -> bool:
    if not isinstance(evidence, dict) or not all(_text(evidence, key) for key in ("source", "contains")):
        return False
    if "page" not in evidence:
        return True
    page = evidence["page"]
    return type(page) is int and page > 0


def load_manifest(data: bytes) -> dict:
    """Reject incomplete labels and incompatible schemas before embedding."""
    manifest = json.loads(data)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Expected an evaluation manifest with schema_version 1.")
    if not _text(manifest, "description"):
        raise ValueError("The manifest needs a description with case provenance.")
    corpus = manifest.get("corpus")
    if not isinstance(corpus, list) or not all(
        isinstance(entry, dict) and _text(entry, "path") and _text(entry, "provenance")
        for entry in corpus
    ):
        raise ValueError("The corpus must list paths and their provenance.")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("The manifest needs at least one case.")

    case_ids = set()
    for case in cases:
        if not isinstance(case, dict) or not all(_text(case, key) for key in ("id", "kind", "question")):
            raise ValueError("Every case needs a string id, kind, and question.")
        case_id = case["id"]
        if case_id in case_ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        case_ids.add(case_id)
        kind = case["kind"]
        if kind not in CASE_KINDS:
            raise ValueError(f"{case_id}: unexpected case kind {kind!r}.")
        evidence = case.get("evidence")
        if not isinstance(evidence, list) or not all(_valid_evidence(entry) for entry in evidence):
            raise ValueError(f"{case_id}: evidence needs a source, excerpt text, and optional page.")
        if bool(evidence) != (kind != "negative"):
            raise ValueError(f"{case_id}: only negative cases must have empty evidence.")
        if kind == "conflict" and len({entry["source"] for entry in evidence}) < 2:
            raise ValueError(f"{case_id}: conflict cases must label evidence from at least two sources.")
        if kind == "follow_up":
            history = case.get("history")
            if not _text(case, "standalone_query") or not isinstance(history, list) or not history:
                raise ValueError(f"{case_id}: follow-ups need history and a supplied standalone query.")
            if len(history) % 2 or any(
                not isinstance(turn, dict) or not _text(turn, "content")
                or turn.get("role") != ("user" if index % 2 == 0 else "assistant")
                for index, turn in enumerate(history)
            ):
                raise ValueError(f"{case_id}: history must contain complete user and assistant turns.")
    return manifest


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _evidence_rank(evidence: dict, results: list[dict]) -> int | None:
    snippet = _normalized(evidence["contains"])
    for rank, hit in enumerate(results, 1):
        if hit["source"] != evidence["source"]:
            continue
        if "page" in evidence and hit.get("page") != evidence["page"]:
            continue
        if snippet in _normalized(hit["text"]):
            return rank
    return None


def load_corpus(manifest: dict, directory: Path) -> tuple[list[dict], list[dict]]:
    """Build regular ingestion chunks and record the exact fixture bytes used."""
    documents = []
    files = []
    source_names = set()
    for entry in manifest["corpus"]:
        path = directory / entry["path"]
        if path.name in source_names:
            raise ValueError(f"Duplicate corpus filename: {path.name}")
        source_names.add(path.name)
        with path.open("rb") as file:
            data = file.read(MAX_FILE_BYTES + 1)
        documents.extend(load_document_bytes(path.name, data))
        files.append({**entry, "sha256": sha256(data).hexdigest()})
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("The corpus produced no chunks. Provide corpus files or pass --index.")
    for case in manifest["cases"]:
        for evidence in case["evidence"]:
            if _evidence_rank(evidence, chunks) is None:
                raise ValueError(f"{case['id']}: labelled evidence does not occur together in a corpus chunk.")
    return chunks, files


def _rate(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _metrics(rows: list[dict], k: int) -> dict:
    positive = [row for row in rows if row["kind"] != "negative"]
    negative = [row for row in rows if row["kind"] == "negative"]
    timings = sorted(row["retrieval_ms"] for row in rows)
    return {
        "questions": len(rows),
        "positive_questions": len(positive),
        f"evidence_hit_at_{k}": _rate([row["first_evidence_rank"] is not None for row in positive]),
        f"mrr_at_{k}": _rate([1 / row["first_evidence_rank"] if row["first_evidence_rank"] else 0
                            for row in positive]),
        f"full_evidence_at_{k}": _rate([all(row["evidence_ranks"]) for row in positive]),
        "negative_questions": len(negative),
        "negative_empty_retrieval_rate": _rate([not row["retrieved_count"] for row in negative]),
        "retrieval_ms_p50": round(median(timings), 2) if timings else None,
        "retrieval_ms_p95": round(timings[math.ceil(0.95 * len(timings)) - 1], 2) if timings else None,
    }


def evaluate(
    store: VectorStore, cases: list[dict], k: int, threshold: float, *,
    mode: str = "dense", reranker: Reranker | None = None, details: bool = False,
) -> dict:
    """Score source-plus-excerpt matching on originals and supplied standalone follow-ups."""
    def run_case(case: dict, query: str, variant: str) -> dict:
        started = perf_counter()
        hits = store.search(query, top_k=k, min_score=threshold, mode=mode, reranker=reranker)
        elapsed_ms = (perf_counter() - started) * 1000
        ranks = [_evidence_rank(evidence, hits) for evidence in case["evidence"]]
        return {
            "id": case["id"],
            "kind": case["kind"],
            "query_variant": variant,
            "query": query,
            "first_evidence_rank": min((rank for rank in ranks if rank is not None), default=None),
            "evidence_ranks": ranks,
            "retrieved_count": len(hits),
            "retrieved_chunk_ids": [hit.get("id") for hit in hits],
            "rerank_status": hits[0].get("rerank_status", "not_run") if hits else "no_candidates",
            "retrieval_ms": elapsed_ms,
        }

    original_rows = [run_case(case, case["question"], "original") for case in cases]
    supplied_rows = [
        run_case(case, case["standalone_query"], "supplied_standalone")
        for case in cases if case["kind"] == "follow_up"
    ]
    rows = original_rows + supplied_rows
    result = {
        "original_questions": _metrics(original_rows, k),
        "conflict_questions": _metrics([row for row in original_rows if row["kind"] == "conflict"], k),
        "follow_ups": {
            "original": _metrics([row for row in original_rows if row["kind"] == "follow_up"], k),
            "supplied_standalone": _metrics(supplied_rows, k),
        },
    }
    if reranker is not None:
        result["rerank_status_counts"] = dict(Counter(row["rerank_status"] for row in rows))
    if details:
        result["cases"] = rows
    return result


def _file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _warm_search(store: VectorStore, cases: list[dict], k: int, threshold: float, mode: str,
                 reranker: Reranker | None) -> int:
    # A one-candidate search exercises the encoder but defers reranker initialization.
    for count, case in enumerate(cases, 1):
        hits = store.search(case["question"], top_k=k, min_score=threshold, mode=mode, reranker=reranker)
        if reranker is None or (hits and hits[0]["rerank_status"] in {"applied", "fallback"}):
            return count
    return len(cases)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/demo_cases.json")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.30, help="Dense cosine cutoff, also used by hybrid")
    parser.add_argument("--index", type=Path, help="Use an existing saved index instead of the manifest corpus")
    parser.add_argument("--backend", choices=["faiss", "numpy"], default=os.getenv("VECTOR_BACKEND", "faiss"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--mode", choices=["dense", "hybrid", "compare"], default="compare")
    parser.add_argument("--reranker-model", nargs="?", const=DEFAULT_RERANKER_MODEL,
                        help="Add a hybrid cross-encoder run; model weights may be downloaded")
    parser.add_argument("--details", action="store_true", help="Include case-level query and rank diagnostics")
    args = parser.parse_args()
    if args.k < 1 or not -1 <= args.threshold <= 1:
        parser.error("--k must be positive and --threshold must be between -1 and 1.")
    if args.mode == "dense" and args.reranker_model:
        parser.error("--reranker-model needs --mode hybrid or compare.")

    store = VectorStore(directory=args.index or "data/processed", model_name=args.model, backend=args.backend)
    try:
        manifest_bytes = args.cases.read_bytes()
        manifest = load_manifest(manifest_bytes)
        if args.index is None:
            chunks, files = load_corpus(manifest, args.cases.parent)
            store.build(chunks)
            corpus = {"kind": "manifest", "files": files}
        else:
            if not store.load():
                raise ValueError("No saved index found. Run python scripts/ingest.py with matching settings.")
            filenames = ("chunks.json", "index.faiss" if args.backend == "faiss" else "index.npy")
            corpus = {
                "kind": "saved_index",
                "files": [{"path": name, "sha256": _file_sha256(args.index / name)}
                          for name in filenames],
            }
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    strategies = {}
    if args.mode in {"compare", "dense"}:
        strategies["dense"] = ("dense", None)
    if args.mode in {"compare", "hybrid"}:
        strategies["hybrid"] = ("hybrid", None)
    if args.reranker_model:
        strategies["hybrid_reranked"] = ("hybrid", CrossEncoderReranker(args.reranker_model))
    cases = manifest["cases"]
    runs = {}
    for name, (mode, reranker) in strategies.items():
        warmup = _warm_search(store, cases, args.k, args.threshold, mode, reranker)
        runs[name] = {
            "warmup_queries": warmup,
            **evaluate(store, cases, args.k, args.threshold, mode=mode, reranker=reranker, details=args.details),
        }

    packages = ["numpy", "sentence-transformers", "torch", "transformers"]
    if args.backend == "faiss":
        packages.append("faiss-cpu")
    report = {
        "schema_version": manifest["schema_version"],
        "description": manifest["description"],
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "corpus": {**corpus, **store.stats()},
        "embedding_model": args.model,
        "reranker_model": args.reranker_model,
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "processor": platform.processor() or platform.machine(),
            "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
            "packages": {name: version(name) for name in packages},
        },
        "k": args.k,
        "dense_cosine_cutoff": args.threshold,
        "latency": "Per-query search including query encoding, after warming each strategy.",
        "scope": "Retrieval only. Follow-up standalone queries are supplied labels. No runtime rewrite, policy resolution or answer generation is evaluated.",
        "runs": runs,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
