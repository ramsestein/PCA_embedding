"""
Re-ranker híbrido que combina:
- Personalized PageRank (PPR) sobre grafo de chunks
- Query-Likelihood Model (QLM) con suavizado Dirichlet
- Markov Random Field (MRF) con dependencias secuenciales
- Fusión con embeddings mediante mezcla lineal normalizada
"""

from .types import Chunk, Query, ScoredChunk, CorpusStats
from .config import RerankConfig
from .fusion import rerank
from .graph import build_chunk_graph, personalized_pagerank
from .qlm import compute_corpus_stats, qlm_dirichlet_score
from .mrf import mrf_sd_score

__version__ = "1.0.0"
__all__ = [
    "Chunk", "Query", "ScoredChunk", "CorpusStats",
    "RerankConfig", "rerank",
    "build_chunk_graph", "personalized_pagerank",
    "compute_corpus_stats", "qlm_dirichlet_score",
    "mrf_sd_score"
]
