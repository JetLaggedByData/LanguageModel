"""
v3_agentic/memory/chapter_store.py
FAISS-backed vector memory for written chapters.

The Writer agent calls:
  - add_chapter()         after generating each chapter
  - get_relevant_context() to retrieve semantically relevant past
                           chapter excerpts before writing the next one

Embeddings run on CPU (sentence-transformers/all-MiniLM-L6-v2, ~80MB).
FAISS index is persisted to disk per story so a crashed run can resume.

One ChapterMemory instance per story run, keyed by story_id.
"""

import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


STORIES_DIR   = Path(__file__).resolve().parent.parent.parent / "data/stories"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_FILENAME = "faiss_docs.json"   # public-API shadow of FAISS docstore

# How many sentences to extract per chapter for embedding
CHUNK_SENTENCES = 8
# Character limit per FAISS document chunk
CHUNK_MAX_CHARS = 600


class ChapterMemory:
    """
    Stores and retrieves chapter content via FAISS semantic search.

    Usage:
        memory = ChapterMemory(story_id="the-last-colony-20241201")
        memory.add_chapter(1, chapter_content)
        context = memory.get_relevant_context("What happened on the ship?")
    """

    def __init__(self, story_id: str) -> None:
        self.story_id  = story_id
        self.index_dir = STORIES_DIR / story_id / "faiss_index"
        self._store: FAISS | None = None
        self._documents: list[Document] = []   # public-API shadow; avoids ._dict
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},   # always CPU — never waste VRAM on embeddings
            encode_kwargs={"normalize_embeddings": True},
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _chunk_chapter(self, chapter_num: int, content: str) -> list[Document]:
        """
        Split chapter content into overlapping sentence chunks for FAISS.
        Returns list of LangChain Documents with chapter metadata.
        """
        sentences = [s.strip() for s in content.replace("\n", " ").split(". ") if s.strip()]
        chunks: list[Document] = []

        for i in range(0, len(sentences), CHUNK_SENTENCES // 2):  # 50% overlap
            chunk_text = ". ".join(sentences[i: i + CHUNK_SENTENCES])
            if not chunk_text:
                continue
            chunk_text = chunk_text[:CHUNK_MAX_CHARS]
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "chapter_num": chapter_num,
                    "chunk_index": i,
                    "story_id":    self.story_id,
                },
            ))

        return chunks

    def _persist(self) -> None:
        """Save FAISS index and document shadow list to disk."""
        if self._store is not None:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self.index_dir))
            docs_data = [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self._documents
            ]
            (self.index_dir / DOCS_FILENAME).write_text(json.dumps(docs_data))

    def _load_from_disk(self) -> bool:
        """Load existing FAISS index and document shadow list. Returns True if found."""
        if (self.index_dir / "index.faiss").exists():
            self._store = FAISS.load_local(
                str(self.index_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            docs_path = self.index_dir / DOCS_FILENAME
            if docs_path.exists():
                raw = json.loads(docs_path.read_text(encoding="utf-8"))
                self._documents = [
                    Document(page_content=d["page_content"], metadata=d["metadata"])
                    for d in raw
                ]
            return True
        return False

    # ── Public interface ──────────────────────────────────────────────────

    def add_chapter(self, chapter_num: int, content: str) -> None:
        """
        Embed and store a completed chapter in FAISS.
        Persists index to disk after every chapter for crash recovery.

        Args:
            chapter_num: 1-indexed chapter number
            content:     Full chapter text
        """
        if not content.strip():
            return

        docs = self._chunk_chapter(chapter_num, content)
        if not docs:
            return

        if self._store is None:
            if not _load_from_disk_safe(self):
                self._store = FAISS.from_documents(docs, self._embeddings)
        else:
            self._store.add_documents(docs)

        self._documents.extend(docs)
        self._persist()

    def get_relevant_context(
        self,
        query: str,
        k: int = 3,
        exclude_chapter: int | None = None,
    ) -> str:
        """
        Retrieve the k most semantically relevant chapter excerpts for a query.
        Used by the Writer to maintain narrative consistency.

        Args:
            query:           Typically the chapter brief or story so far summary
            k:               Number of chunks to retrieve
            exclude_chapter: Skip chunks from this chapter (avoid self-retrieval)

        Returns:
            Concatenated excerpt string, or empty string if store is empty.
        """
        if self._store is None:
            if not _load_from_disk_safe(self):
                return ""

        try:
            docs = self._store.similarity_search(query, k=k + 2)  # fetch extra, then filter
        except Exception:
            return ""

        filtered = [
            d for d in docs
            if d.metadata.get("chapter_num") != exclude_chapter
        ][:k]

        if not filtered:
            return ""

        parts = []
        for doc in filtered:
            ch = doc.metadata.get("chapter_num", "?")
            parts.append(f"[Chapter {ch}] {doc.page_content}")

        return "\n\n".join(parts)

    def update_chapter(self, chapter_num: int, new_content: str) -> None:
        """
        Replace all FAISS chunks for chapter_num with newly embedded content.

        FAISS has no in-place update, so we rebuild the index: load existing
        documents, drop every chunk whose chapter_num matches, add the new
        chunks, then persist. This prevents stale pre-revision vectors from
        polluting future retrieval for subsequent chapters.

        Args:
            chapter_num: 1-indexed chapter number to replace
            new_content: Revised chapter text
        """
        if not new_content.strip():
            return

        # Ensure the index and shadow document list are loaded
        if self._store is None:
            if not _load_from_disk_safe(self):
                self.add_chapter(chapter_num, new_content)
                return

        # Use the shadow list — avoids the private FAISS ._dict attribute
        remaining_docs = [
            doc for doc in self._documents
            if doc.metadata.get("chapter_num") != chapter_num
        ]

        new_docs = self._chunk_chapter(chapter_num, new_content)
        all_docs = remaining_docs + new_docs

        if not all_docs:
            return

        # Rebuild index from scratch with the filtered + new documents
        self._store = FAISS.from_documents(all_docs, self._embeddings)
        self._documents = all_docs
        self._persist()

    def chapter_count(self) -> int:
        """Return number of unique chapters stored."""
        if not self._documents:
            _load_from_disk_safe(self)
        chapters = {d.metadata.get("chapter_num") for d in self._documents}
        return len(chapters)


# ── Module-level helper (avoids self._load inside __init__) ───────────────

def _load_from_disk_safe(memory: ChapterMemory) -> bool:
    """Load FAISS index, return False silently if not found or corrupt."""
    try:
        return memory._load_from_disk()
    except Exception:
        return False
