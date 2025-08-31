"""
Cómputo de estadísticas del corpus para Query-Likelihood Model.
"""

from typing import List, Dict, Iterable
from collections import Counter, defaultdict
from .types import Chunk, CorpusStats
from .utils import tokenize, compute_term_freqs, compute_doc_freqs, compute_chunk_lengths


def compute_corpus_statistics(chunks: Iterable[Chunk], 
                             tokenize_func=tokenize) -> CorpusStats:
    """
    Calcula estadísticas globales del corpus para QLM.
    
    Args:
        chunks: Iterable de chunks del corpus
        tokenize_func: Función de tokenización
        
    Returns:
        Objeto CorpusStats con estadísticas computadas
    """
    # Convertir a lista si es necesario
    chunks_list = list(chunks)
    
    if not chunks_list:
        return CorpusStats(
            term_freqs={},
            doc_lengths={},
            total_tokens=0,
            vocab_size=0
        )
    
    # Calcular frecuencias de términos en el corpus
    term_freqs = compute_term_freqs(chunks_list, tokenize_func)
    
    # Calcular longitudes de chunks
    doc_lengths = compute_chunk_lengths(chunks_list, tokenize_func)
    
    # Calcular total de tokens
    total_tokens = sum(doc_lengths.values())
    
    # Tamaño del vocabulario
    vocab_size = len(term_freqs)
    
    return CorpusStats(
        term_freqs=term_freqs,
        doc_lengths=doc_lengths,
        total_tokens=total_tokens,
        vocab_size=vocab_size
    )


def compute_chunk_term_freqs(chunk: Chunk, tokenize_func=tokenize) -> Dict[str, int]:
    """
    Calcula frecuencias de términos en un chunk específico.
    
    Args:
        chunk: Chunk del documento
        tokenize_func: Función de tokenización
        
    Returns:
        Diccionario {término: frecuencia en el chunk}
    """
    tokens = tokenize_func(chunk.text)
    return Counter(tokens)


def compute_avg_doc_length(chunks: Iterable[Chunk], tokenize_func=tokenize) -> float:
    """
    Calcula la longitud promedio de chunks en tokens.
    
    Args:
        chunks: Iterable de chunks
        tokenize_func: Función de tokenización
        
    Returns:
        Longitud promedio
    """
    chunks_list = list(chunks)
    if not chunks_list:
        return 0.0
    
    total_length = sum(len(tokenize_func(chunk.text)) for chunk in chunks_list)
    return total_length / len(chunks_list)


def compute_idf_scores(chunks: Iterable[Chunk], tokenize_func=tokenize) -> Dict[str, float]:
    """
    Calcula puntuaciones IDF para todos los términos del corpus.
    
    Args:
        chunks: Iterable de chunks
        tokenize_func: Función de tokenización
        
    Returns:
        Diccionario {término: puntuación IDF}
    """
    chunks_list = list(chunks)
    if not chunks_list:
        return {}
    
    doc_freqs = compute_doc_freqs(chunks_list, tokenize_func)
    total_docs = len(chunks_list)
    
    idf_scores = {}
    for term, doc_freq in doc_freqs.items():
        # IDF = log((N - df + 0.5) / (df + 0.5))
        idf = (total_docs - doc_freq + 0.5) / (doc_freq + 0.5)
        idf_scores[term] = idf if idf > 0 else 0.0
    
    return idf_scores


def compute_tfidf_scores(chunk: Chunk, idf_scores: Dict[str, float], 
                         tokenize_func=tokenize) -> Dict[str, float]:
    """
    Calcula puntuaciones TF-IDF para un chunk.
    
    Args:
        chunk: Chunk del documento
        idf_scores: Puntuaciones IDF precomputadas
        tokenize_func: Función de tokenización
        
    Returns:
        Diccionario {término: puntuación TF-IDF}
    """
    tokens = tokenize_func(chunk.text)
    term_freqs = Counter(tokens)
    
    tfidf_scores = {}
    for term, freq in term_freqs.items():
        tf = freq
        idf = idf_scores.get(term, 0.0)
        tfidf_scores[term] = tf * idf
    
    return tfidf_scores


def get_corpus_vocabulary(chunks: Iterable[Chunk], tokenize_func=tokenize) -> set:
    """
    Obtiene el vocabulario completo del corpus.
    
    Args:
        chunks: Iterable de chunks
        tokenize_func: Función de tokenización
        
    Returns:
        Conjunto de términos únicos
    """
    vocabulary = set()
    for chunk in chunks:
        tokens = tokenize_func(chunk.text)
        vocabulary.update(tokens)
    
    return vocabulary


def compute_term_corpus_probabilities(chunks: Iterable[Chunk], 
                                    tokenize_func=tokenize) -> Dict[str, float]:
    """
    Calcula probabilidades p(t|C) para todos los términos del corpus.
    
    Args:
        chunks: Iterable de chunks
        tokenize_func: Función de tokenización
        
    Returns:
        Diccionario {término: p(t|C)}
    """
    stats = compute_corpus_statistics(chunks, tokenize_func)
    
    if stats.total_tokens == 0:
        return {}
    
    term_probs = {}
    for term, freq in stats.term_freqs.items():
        term_probs[term] = freq / stats.total_tokens
    
    return term_probs


def compute_chunk_specific_stats(chunk: Chunk, corpus_stats: CorpusStats,
                                tokenize_func=tokenize) -> Dict[str, any]:
    """
    Calcula estadísticas específicas de un chunk usando estadísticas del corpus.
    
    Args:
        chunk: Chunk del documento
        corpus_stats: Estadísticas del corpus
        tokenize_func: Función de tokenización
        
    Returns:
        Diccionario con estadísticas del chunk
    """
    tokens = tokenize_func(chunk.text)
    chunk_length = len(tokens)
    
    # Frecuencias de términos en el chunk
    term_freqs = Counter(tokens)
    
    # Probabilidades p(t|C) para términos del chunk
    term_probs = {}
    for term in term_freqs:
        term_probs[term] = corpus_stats.get_term_prob(term)
    
    # Términos únicos del chunk
    unique_terms = set(tokens)
    
    # Términos que no están en el corpus (nuevos)
    new_terms = unique_terms - set(corpus_stats.term_freqs.keys())
    
    return {
        'chunk_length': chunk_length,
        'term_freqs': dict(term_freqs),
        'term_probs': term_probs,
        'unique_terms': len(unique_terms),
        'new_terms': len(new_terms),
        'avg_term_prob': sum(term_probs.values()) / len(term_probs) if term_probs else 0.0
    }
