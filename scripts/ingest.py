"""Rebuild a local vector index from a directory of source documents."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from campusguide_rag import VectorStore, load_directory, chunk_documents


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build the CampusGuide RAG index")
    parser.add_argument("--source", default=os.getenv("DATA_RAW_PATH", "data/raw"))
    parser.add_argument("--output", default=os.getenv("VECTOR_STORE_PATH", "data/processed"))
    parser.add_argument("--backend", choices=["faiss", "numpy"], default=os.getenv("VECTOR_BACKEND", "faiss"))
    args = parser.parse_args()
    docs, errors = load_directory(args.source)
    for message in errors:
        print(f"Skipped {message}", file=sys.stderr)
    chunks = chunk_documents(docs)
    if not chunks:
        print("No valid text chunks found; check your input documents.", file=sys.stderr)
        return 1
    store = VectorStore(args.output, model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"), backend=args.backend)
    store.build(chunks)
    store.save()
    print(f"Indexed {store.stats()['documents']} documents into {len(chunks)} chunks at {args.output} ({args.backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
