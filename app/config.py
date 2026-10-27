"""All tunable settings in one place. Override any field with an environment variable or a .env file."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # Models. The main model writes answers; the small router model rewrites queries and reranks.
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.5-flash")
    router_model: str = os.getenv("ROUTER_MODEL", "gemini-3.5-flash-lite")
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # local via fastembed; no API quota
    embedding_dim: int = 384

    # Paths
    pdf_path: Path = ROOT / "data" / "raw" / "GEP-Jun-2026.pdf"
    index_dir: Path = ROOT / "data" / "index"
    cache_dir: Path = ROOT / "data" / "cache"

    # Hierarchical chunking (characters; roughly 4 chars per token)
    parent_max_chars: int = 3600  # ~900 tokens: the section context given to the LLM
    child_max_chars: int = 1100  # ~280 tokens: the unit that is searched
    child_overlap_sentences: int = 1

    # Retrieval
    dense_k: int = 30
    bm25_k: int = 30
    rrf_k: int = 60  # the constant from the original Reciprocal Rank Fusion paper
    parent_candidates: int = 8  # parents passed to the reranker
    final_parents: int = 4  # parents passed to the answer model
    use_reranker: bool = field(default=False)  # off by default: measured on eval, see README


def get_settings() -> Settings:
    return Settings()
