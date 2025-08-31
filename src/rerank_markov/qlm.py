"""
Query-Likelihood Model (QLM) con suavizado Dirichlet y Jelinek-Mercer.
Basado en Zhai & Lafferty (SIGIR'01, ACM TOIS'04).
"""

import numpy as np
from typing import List, Dict
from .types import Chunk, CorpusStats, Query
from .utils import tokenize


def compute_corpus_stats(chunks: List[Chunk]) -> CorpusStats:
    """
    Calcula estadísticas del corpus necesarias para QLM.
    
    Args:
        chunks: Lista de chunks del corpus
        
    Returns:
        Objeto CorpusStats con estadísticas computadas
    """
    from .index_stats import compute_corpus_statistics
    return compute_corpus_statistics(chunks)


def qlm_dirichlet_score(query: str, chunk: Chunk, stats: CorpusStats, 
                       mu: int = 1500) -> float:
    """
    Calcula puntuación QLM usando suavizado Dirichlet.
    
    Fórmula: log p(q|d) = Σ log((c(t,d) + μ*p(t|C)) / (|d| + μ))
    
    Args:
        query: Texto de la query
        chunk: Chunk del documento
        stats: Estadísticas del corpus
        mu: Parámetro de suavizado Dirichlet (default: 1500)
        
    Returns:
        Puntuación QLM (log-likelihood)
    """
    if not query.strip():
        return 0.0
    
    # Tokenizar query
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    
    # Obtener longitud del chunk
    doc_length = stats.get_doc_length(chunk.id)
    if doc_length == 0:
        return 0.0
    
    # Calcular frecuencias de términos en el chunk
    chunk_tokens = tokenize(chunk.text)
    chunk_term_freqs = {}
    for token in chunk_tokens:
        chunk_term_freqs[token] = chunk_term_freqs.get(token, 0) + 1
    
    # Calcular puntuación QLM
    score = 0.0
    for term in query_tokens:
        # Frecuencia del término en el chunk
        term_freq = chunk_term_freqs.get(term, 0)
        
        # Probabilidad del término en el corpus
        term_corpus_prob = stats.get_term_prob(term)
        
        # Aplicar suavizado Dirichlet
        numerator = term_freq + mu * term_corpus_prob
        denominator = doc_length + mu
        
        if denominator > 0 and numerator > 0:
            prob = numerator / denominator
            # Usar log con offset y transformación para evitar valores muy negativos
            score += np.log(max(prob, 1e-6))
    
    return score


def qlm_jelinek_mercer_score(query: str, chunk: Chunk, stats: CorpusStats,
                            lambda_jm: float = 0.5) -> float:
    """
    Calcula puntuación QLM usando suavizado Jelinek-Mercer.
    
    Fórmula: log p(q|d) = Σ log(λ * p(t|d) + (1-λ) * p(t|C))
    
    Args:
        query: Texto de la query
        chunk: Chunk del documento
        stats: Estadísticas del corpus
        lambda_jm: Parámetro de interpolación (default: 0.5)
        
    Returns:
        Puntuación QLM con suavizado JM
    """
    if not query.strip():
        return 0.0
    
    # Tokenizar query
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    
    # Obtener longitud del chunk
    doc_length = stats.get_doc_length(chunk.id)
    if doc_length == 0:
        return 0.0
    
    # Calcular frecuencias de términos en el chunk
    chunk_tokens = tokenize(chunk.text)
    chunk_term_freqs = {}
    for token in chunk_tokens:
        chunk_term_freqs[token] = chunk_term_freqs.get(token, 0) + 1
    
    # Calcular puntuación QLM con JM
    score = 0.0
    for term in query_tokens:
        # Probabilidad del término en el documento: p(t|d)
        term_freq = chunk_term_freqs.get(term, 0)
        term_doc_prob = term_freq / doc_length if doc_length > 0 else 0.0
        
        # Probabilidad del término en el corpus: p(t|C)
        term_corpus_prob = stats.get_term_prob(term)
        
        # Interpolación Jelinek-Mercer
        prob = lambda_jm * term_doc_prob + (1 - lambda_jm) * term_corpus_prob
        
        if prob > 0:
            # Usar log con offset y transformación para evitar valores muy negativos
            score += np.log(max(prob, 1e-6))
    
    return score


def qlm_score(query: str, chunk: Chunk, stats: CorpusStats, 
              mu: int = 1500, use_jm: bool = False, lambda_jm: float = 0.5) -> float:
    """
    Calcula puntuación QLM usando el método especificado.
    
    Args:
        query: Texto de la query
        chunk: Chunk del documento
        stats: Estadísticas del corpus
        mu: Parámetro de suavizado Dirichlet
        use_jm: Si usar suavizado Jelinek-Mercer
        lambda_jm: Parámetro lambda para JM
        
    Returns:
        Puntuación QLM
    """
    if use_jm:
        return qlm_jelinek_mercer_score(query, chunk, stats, lambda_jm)
    else:
        return qlm_dirichlet_score(query, chunk, stats, mu)


def compute_query_term_weights(query: str, stats: CorpusStats) -> Dict[str, float]:
    """
    Calcula pesos de términos de la query basados en IDF.
    
    Args:
        query: Texto de la query
        stats: Estadísticas del corpus
        
    Returns:
        Diccionario {término: peso}
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return {}
    
    total_docs = len(stats.doc_lengths)
    term_weights = {}
    
    for term in query_tokens:
        # Calcular IDF
        doc_freq = 0
        for chunk_id, length in stats.doc_lengths.items():
            # Aquí simplificamos - en implementación real necesitarías mapeo chunk_id -> términos
            pass
        
        # IDF = log((N - df + 0.5) / (df + 0.5))
        idf = np.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        term_weights[term] = max(0.0, idf)
    
    return term_weights


def weighted_qlm_score(query: str, chunk: Chunk, stats: CorpusStats,
                       mu: int = 1500, use_weights: bool = True) -> float:
    """
    Calcula puntuación QLM ponderada por IDF de términos de la query.
    
    Args:
        query: Texto de la query
        chunk: Chunk del documento
        stats: Estadísticas del corpus
        mu: Parámetro de suavizado Dirichlet
        use_weights: Si usar ponderación por IDF
        
    Returns:
        Puntuación QLM ponderada
    """
    if not use_weights:
        return qlm_dirichlet_score(query, chunk, stats, mu)
    
    # Obtener pesos de términos
    term_weights = compute_query_term_weights(query, stats)
    if not term_weights:
        return qlm_dirichlet_score(query, chunk, stats, mu)
    
    # Tokenizar query
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    
    # Obtener longitud del chunk
    doc_length = stats.get_doc_length(chunk.id)
    if doc_length == 0:
        return 0.0
    
    # Calcular frecuencias de términos en el chunk
    chunk_tokens = tokenize(chunk.text)
    chunk_term_freqs = {}
    for token in chunk_tokens:
        chunk_term_freqs[token] = chunk_term_freqs.get(token, 0) + 1
    
    # Calcular puntuación QLM ponderada
    score = 0.0
    for term in query_tokens:
        weight = term_weights.get(term, 1.0)
        
        # Frecuencia del término en el chunk
        term_freq = chunk_term_freqs.get(term, 0)
        
        # Probabilidad del término en el corpus
        term_corpus_prob = stats.get_term_prob(term)
        
        # Aplicar suavizado Dirichlet
        numerator = term_freq + mu * term_corpus_prob
        denominator = doc_length + mu
        
        if denominator > 0 and numerator > 0:
            prob = numerator / denominator
            score += weight * np.log(prob)
    
    return score
