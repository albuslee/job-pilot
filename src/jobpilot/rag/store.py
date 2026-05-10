"""Persistent ChromaDB-backed vector store. Embeddings injected via the Embedder protocol."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import chromadb

from jobpilot.models.schemas import ProfileChunk

if TYPE_CHECKING:
    from jobpilot.rag.embeddings import Embedder


class RagStore:
    def __init__(
        self, *, persist_dir: Path, embedder: Embedder, collection: str = "profile"
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection)
        self._embedder = embedder

    def add(self, chunks: list[ProfileChunk]) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"source": c.source, "heading_path": " > ".join(c.heading_path)} for c in chunks
        ]
        embeddings = self._embedder.embed(documents)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
            embeddings=embeddings,  # type: ignore[arg-type]
        )

    def query(self, text: str, k: int) -> list[ProfileChunk]:
        if k <= 0 or self._collection.count() == 0:
            return []
        embedding = self._embedder.embed([text])[0]
        result = self._collection.query(query_embeddings=[embedding], n_results=k)  # type: ignore[arg-type]
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        out: list[ProfileChunk] = []
        for cid, doc, meta in zip(ids, docs, metas, strict=False):
            heading = str(meta.get("heading_path") or "")
            out.append(
                ProfileChunk(
                    id=cid,
                    source=str(meta.get("source") or "unknown"),
                    heading_path=[h for h in heading.split(" > ") if h],
                    text=doc,
                )
            )
        return out
