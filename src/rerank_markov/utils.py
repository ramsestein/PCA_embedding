"""
Utilidades para el re-ranker híbrido: tokenización, conteos y helpers.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokeniza texto usando regex \W+ y convierte a minúsculas.
    
    Args:
        text: Texto a tokenizar
        lowercase: Si convertir a minúsculas
        
    Returns:
        Lista de tokens
    """
    if lowercase:
        text = text.lower()
    
    # Usar regex \W+ para dividir por caracteres no alfanuméricos
    tokens = re.split(r'\W+', text)
    
    # Filtrar tokens vacíos
    tokens = [token.strip() for token in tokens if token.strip()]
    
    return tokens


def compute_term_freqs(chunks: List, tokenize_func=tokenize) -> Dict[str, int]:
    """
    Calcula frecuencias de términos en el corpus.
    
    Args:
        chunks: Lista de chunks
        tokenize_func: Función de tokenización
        
    Returns:
        Diccionario {término: frecuencia}
    """
    term_freqs = Counter()
    
    for chunk in chunks:
        tokens = tokenize_func(chunk.text)
        term_freqs.update(tokens)
    
    return dict(term_freqs)


def compute_doc_freqs(chunks: List, tokenize_func=tokenize) -> Dict[str, int]:
    """
    Calcula frecuencias de documento (en cuántos chunks aparece cada término).
    
    Args:
        chunks: Lista de chunks
        tokenize_func: Función de tokenización
        
    Returns:
        Diccionario {término: frecuencia de documento}
    """
    doc_freqs = defaultdict(set)
    
    for chunk in chunks:
        tokens = set(tokenize_func(chunk.text))  # Usar set para términos únicos
        for token in tokens:
            doc_freqs[token].add(chunk.id)
    
    return {term: len(docs) for term, docs in doc_freqs.items()}


def compute_chunk_lengths(chunks: List, tokenize_func=tokenize) -> Dict[str, int]:
    """
    Calcula longitudes de chunks en tokens.
    
    Args:
        chunks: Lista de chunks
        tokenize_func: Función de tokenización
        
    Returns:
        Diccionario {chunk_id: longitud}
    """
    return {chunk.id: len(tokenize_func(chunk.text)) for chunk in chunks}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula similitud coseno entre dos vectores.
    
    Args:
        a, b: Vectores numpy
        
    Returns:
        Similitud coseno en [0, 1]
    """
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def normalize_scores(scores: List[float], method: str = 'zscore', 
                    clip_sigma: float = 3.0) -> List[float]:
    """
    Normaliza puntuaciones usando diferentes métodos.
    
    Args:
        scores: Lista de puntuaciones
        method: Método de normalización ('zscore', 'minmax', 'none')
        clip_sigma: Para z-score, número de desviaciones estándar para clipping
        
    Returns:
        Lista de puntuaciones normalizadas
    """
    if not scores:
        return []
    
    if method == 'none':
        return scores
    
    scores = np.array(scores)
    
    if method == 'zscore':
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return [0.0] * len(scores)
        
        normalized = (scores - mean) / std
        
        # Clipping a ±clip_sigma desviaciones estándar
        normalized = np.clip(normalized, -clip_sigma, clip_sigma)
        
        return normalized.tolist()
    
    elif method == 'minmax':
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        if max_val == min_val:
            return [0.5] * len(scores)
        
        normalized = (scores - min_val) / (max_val - min_val)
        return normalized.tolist()
    
    else:
        raise ValueError(f"Método de normalización no soportado: {method}")


def create_term_positions(chunk, tokenize_func=tokenize) -> Dict[str, List[int]]:
    """
    Crea índice de posiciones de términos en un chunk.
    
    Args:
        chunk: Chunk del documento
        tokenize_func: Función de tokenización
        
    Returns:
        Diccionario {término: [posiciones]}
    """
    tokens = tokenize_func(chunk.text)
    term_positions = defaultdict(list)
    
    for pos, token in enumerate(tokens):
        term_positions[token].append(pos)
    
    return dict(term_positions)


def extract_bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Extrae bigramas consecutivos de una lista de tokens.
    
    Args:
        tokens: Lista de tokens
        
    Returns:
        Lista de bigramas (tuplas de 2 términos)
    """
    if len(tokens) < 2:
        return []
    
    return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]


def extract_unordered_bigrams(tokens: List[str], window: int = 8) -> List[Tuple[str, str]]:
    """
    Extrae bigramas no ordenados dentro de una ventana.
    
    Args:
        tokens: Lista de tokens
        window: Tamaño de ventana
        
    Returns:
        Lista de bigramas no ordenados
    """
    bigrams = set()
    
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window + 1, len(tokens))):
            # Ordenar términos para bigramas no ordenados
            term1, term2 = sorted([tokens[i], tokens[j]])
            bigrams.add((term1, term2))
    
    return list(bigrams)


def bm25_score(term: str, chunk, doc_freqs: Dict[str, int], 
               avg_doc_length: float, k1: float = 1.2, b: float = 0.75) -> float:
    """
    Calcula puntuación BM25 para un término en un chunk.
    
    Args:
        term: Término a evaluar
        chunk: Chunk del documento
        doc_freqs: Frecuencias de documento
        avg_doc_length: Longitud promedio de documentos
        k1, b: Parámetros BM25
        
    Returns:
        Puntuación BM25
    """
    tokens = tokenize(chunk.text)
    term_freq = tokens.count(term)
    
    if term_freq == 0:
        return 0.0
    
    doc_length = len(tokens)
    doc_freq = doc_freqs.get(term, 0)
    
    # Fórmula BM25
    numerator = term_freq * (k1 + 1)
    denominator = term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length))
    
    # IDF component (log del número total de documentos)
    total_docs = len(set(chunk.doc_id for chunk in [chunk]))  # Simplificado
    idf = np.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    
    return idf * (numerator / denominator)
