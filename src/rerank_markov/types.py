"""
Tipos de datos para el re-ranker híbrido.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class Chunk:
    """Chunk de documento con metadatos y embedding."""
    id: str
    text: str
    doc_id: str
    position: int  # índice del chunk dentro del documento
    embedding: Optional[np.ndarray] = None  # vector precomputado
    meta: Dict[str, Any] = field(default_factory=dict)  # enlaces, contigüidad, etc.


@dataclass
class Query:
    """Query de búsqueda con tokens preprocesados."""
    text: str
    tokens: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


@dataclass
class ScoredChunk:
    """Chunk con puntuaciones de todos los componentes del re-ranker."""
    chunk: Chunk
    total_score: float
    embedding_score: float
    ppr_score: float
    qlm_score: float
    mrf_score: float
    rank: Optional[int] = None


@dataclass
class CorpusStats:
    """Estadísticas globales del corpus para QLM."""
    term_freqs: Dict[str, int]  # c(t,C) - frecuencia de término en corpus
    doc_lengths: Dict[str, int]  # |d| - longitud de cada chunk
    total_tokens: int  # total de tokens en corpus
    vocab_size: int  # tamaño del vocabulario
    
    def get_term_prob(self, term: str) -> float:
        """Calcula p(t|C) = c(t,C) / total_tokens."""
        return self.term_freqs.get(term, 0) / max(self.total_tokens, 1)
    
    def get_doc_length(self, chunk_id: str) -> int:
        """Obtiene la longitud de un chunk específico."""
        return self.doc_lengths.get(chunk_id, 0)
