from __future__ import annotations

from pathlib import Path

from jobpilot.models.schemas import ProfileChunk
from jobpilot.rag.store import RagStore


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        # deterministic 4-d vector based on length so similar texts cluster
        return [[len(t) / 100, 0.0, 0.0, 0.0] for t in texts]


def test_add_then_query_round_trip(tmp_path: Path) -> None:
    embedder = _FakeEmbedder()
    store = RagStore(persist_dir=tmp_path / "chroma", embedder=embedder, collection="test")

    chunks = [
        ProfileChunk(id="cv-1", source="cv.docx", heading_path=["A"], text="alpha bravo"),
        ProfileChunk(
            id="cv-2", source="cv.docx", heading_path=["A"], text="charlie delta echo foxtrot"
        ),
    ]
    store.add(chunks)
    hits = store.query("alpha", k=1)
    assert len(hits) == 1
    assert hits[0].id in {"cv-1", "cv-2"}


def test_query_returns_at_most_k(tmp_path: Path) -> None:
    embedder = _FakeEmbedder()
    store = RagStore(persist_dir=tmp_path / "chroma", embedder=embedder, collection="test")
    chunks = [
        ProfileChunk(id=f"cv-{i}", source="cv.docx", heading_path=[], text=f"text-{i}")
        for i in range(5)
    ]
    store.add(chunks)
    assert len(store.query("text", k=3)) == 3


def test_add_empty_is_noop(tmp_path: Path) -> None:
    embedder = _FakeEmbedder()
    store = RagStore(persist_dir=tmp_path / "chroma", embedder=embedder, collection="test")
    store.add([])
    assert store.query("anything", k=1) == []
