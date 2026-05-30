#!/usr/bin/env python3
"""Inspect chunks stored in the local ChromaDB profile collection.

Usage:
    uv run python runners/inspect_chroma.py
    uv run python runners/inspect_chroma.py --limit 10 --full
    uv run python runners/inspect_chroma.py --chroma-dir .chroma --collection profile --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _default_chroma_dir() -> Path:
    try:
        from jobpilot.config import get_settings
    except Exception:
        return Path(".chroma")
    return get_settings().chroma_dir


def _collection_names(client: Any) -> list[str]:
    names: list[str] = []
    for item in client.list_collections():
        if isinstance(item, str):
            names.append(item)
        else:
            names.append(str(getattr(item, "name", item)))
    return names


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n..."


def _embedding_values(embedding: Any) -> list[float]:
    if embedding is None:
        return []
    return [float(value) for value in embedding]


def _embedding_summary(embedding: Any) -> str:
    values = _embedding_values(embedding)
    if not values:
        return "embedding: dim=0 []"
    head = ", ".join(f"{value:.5g}" for value in values[:8])
    suffix = ", ..." if len(values) > 8 else ""
    return f"embedding: dim={len(values)} [{head}{suffix}]"


def _format_chunk(
    index: int,
    chunk_id: str,
    document: str,
    metadata: dict[str, Any] | None,
    *,
    preview_chars: int,
    embedding: Any | None = None,
) -> str:
    metadata = metadata or {}
    lines = [
        f"[{index}] {chunk_id}",
        f"source: {metadata.get('source', '')}",
        f"heading: {metadata.get('heading_path', '')}",
    ]
    if embedding is not None:
        lines.append(_embedding_summary(embedding))
    lines.append(_truncate_text(document, preview_chars))
    return "\n".join(lines)


def _records_from_result(
    result: dict[str, Any], *, include_embeddings: bool
) -> list[dict[str, Any]]:
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings") if include_embeddings else None
    records: list[dict[str, Any]] = []

    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        record = {
            "id": chunk_id,
            "source": metadata.get("source", ""),
            "heading_path": metadata.get("heading_path", ""),
            "text": documents[index] if index < len(documents) else "",
        }
        if embeddings is not None:
            record["embedding"] = _embedding_values(embeddings[index])
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print documents and metadata from a local ChromaDB collection."
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=None,
        help="Chroma persist directory. Defaults to CHROMA_DIR from .env, then .chroma.",
    )
    parser.add_argument(
        "--collection",
        default="profile",
        help="Chroma collection name. Default: profile.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum chunks to print. Use 0 for all chunks. Default: 20.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1200,
        help="Maximum characters per chunk. Ignored by --full. Default: 1200.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full chunk text instead of a preview.",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Include embedding summaries in text output, or full embeddings in JSON output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print chunks as JSON records.",
    )
    args = parser.parse_args(argv)

    chroma_dir = args.chroma_dir or _default_chroma_dir()
    if not chroma_dir.exists():
        print(f"error: Chroma directory not found: {chroma_dir}", file=sys.stderr)
        return 2

    try:
        import chromadb
    except ImportError as exc:
        print(f"error: chromadb is not installed: {exc}", file=sys.stderr)
        return 2

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection(args.collection)
    except Exception as exc:
        known = ", ".join(_collection_names(client)) or "(none)"
        print(
            f"error: collection {args.collection!r} not found in {chroma_dir}: {exc}",
            file=sys.stderr,
        )
        print(f"known collections: {known}", file=sys.stderr)
        return 2

    include = ["documents", "metadatas"]
    if args.embeddings:
        include.append("embeddings")
    limit = None if args.limit == 0 else args.limit
    result = collection.get(limit=limit, include=include)

    if args.json:
        print(
            json.dumps(_records_from_result(result, include_embeddings=args.embeddings), indent=2)
        )
        return 0

    print(f"collection: {args.collection}")
    print(f"chroma_dir: {chroma_dir}")
    print(f"count: {collection.count()}")
    print(f"shown: {len(result.get('ids') or [])}")

    preview_chars = 0 if args.full else args.preview_chars
    embeddings = result.get("embeddings") if args.embeddings else None
    for index, chunk_id in enumerate(result.get("ids") or [], start=1):
        document = (result.get("documents") or [])[index - 1]
        metadata = (result.get("metadatas") or [])[index - 1]
        embedding = embeddings[index - 1] if embeddings is not None else None
        print()
        print(
            _format_chunk(
                index,
                chunk_id,
                document,
                metadata,
                preview_chars=preview_chars,
                embedding=embedding,
            )
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
