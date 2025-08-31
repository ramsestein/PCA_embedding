"""
Markov Random Field (MRF) con dependencias secuenciales.
Basado en Metzler & Croft (SIGIR'05).
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from .types import Chunk
from .utils import tokenize, extract_bigrams, extract_unordered_bigrams, create_term_positions


def mrf_sd_score(query: str, chunk: Chunk, window: int = 8, 
                 w_unigram: float = 0.8, w_ordered: float = 0.1, 
                 w_unordered: float = 0.1) -> float:
    """
    Calcula puntuación MRF con dependencias secuenciales.
    
    Fórmula: score = w_u * f_unigram + w_o * f_ordered + w_w * f_unordered
    
    Args:
        query: Texto de la query
        chunk: Chunk del documento
        window: Tamaño de ventana para bigramas no ordenados
        w_unigram: Peso para características unigram
        w_ordered: Peso para bigramas ordenados
        w_unigram: Peso para bigramas no ordenados
        
    Returns:
        Puntuación MRF combinada
    """
    if not query.strip():
        return 0.0
    
    # Tokenizar query y chunk
    query_tokens = tokenize(query)
    chunk_tokens = tokenize(chunk.text)
    
    if not query_tokens or not chunk_tokens:
        return 0.0
    
    # Calcular características individuales
    unigram_score = _compute_unigram_score(query_tokens, chunk_tokens)
    ordered_score = _compute_ordered_bigram_score(query_tokens, chunk_tokens)
    unordered_score = _compute_unordered_bigram_score(query_tokens, chunk_tokens, window)
    
    # Combinar puntuaciones con pesos
    total_score = (w_unigram * unigram_score + 
                   w_ordered * ordered_score + 
                   w_unordered * unordered_score)
    
    return total_score


def _compute_unigram_score(query_tokens: List[str], chunk_tokens: List[str]) -> float:
    """
    Calcula puntuación unigram basada en presencia y frecuencia de términos.
    
    Args:
        query_tokens: Tokens de la query
        chunk_tokens: Tokens del chunk
        
    Returns:
        Puntuación unigram
    """
    if not query_tokens:
        return 0.0
    
    # Crear conjunto de términos únicos del chunk para búsqueda eficiente
    chunk_terms = set(chunk_tokens)
    
    # Contar términos de la query que aparecen en el chunk
    matches = 0
    total_freq = 0
    
    for term in query_tokens:
        if term in chunk_terms:
            matches += 1
            # Contar frecuencia del término en el chunk
            freq = chunk_tokens.count(term)
            total_freq += freq
    
    # Puntuación basada en cobertura y frecuencia
    coverage = matches / len(query_tokens) if query_tokens else 0.0
    frequency_bonus = min(0.1, np.log(1 + total_freq) / 10) if total_freq > 0 else 0.0
    
    # Normalizar para que esté en [0, 1]
    score = coverage + frequency_bonus
    return min(1.0, max(0.0, score))


def _compute_ordered_bigram_score(query_tokens: List[str], chunk_tokens: List[str]) -> float:
    """
    Calcula puntuación para bigramas ordenados (términos consecutivos).
    
    Args:
        query_tokens: Tokens de la query
        chunk_tokens: Tokens del chunk
        
    Returns:
        Puntuación para bigramas ordenados
    """
    if len(query_tokens) < 2:
        return 0.0
    
    # Extraer bigramas de la query
    query_bigrams = extract_bigrams(query_tokens)
    if not query_bigrams:
        return 0.0
    
    # Extraer bigramas del chunk
    chunk_bigrams = extract_bigrams(chunk_tokens)
    
    # Contar coincidencias exactas
    matches = 0
    for bigram in query_bigrams:
        if bigram in chunk_bigrams:
            matches += 1
    
    # Puntuación basada en proporción de bigramas coincidentes
    return matches / len(query_bigrams) if query_bigrams else 0.0


def _compute_unordered_bigram_score(query_tokens: List[str], chunk_tokens: List[str], 
                                   window: int = 8) -> float:
    """
    Calcula puntuación para bigramas no ordenados dentro de una ventana.
    
    Args:
        query_tokens: Tokens de la query
        chunk_tokens: Tokens del chunk
        window: Tamaño de ventana para co-ocurrencia
        
    Returns:
        Puntuación para bigramas no ordenados
    """
    if len(query_tokens) < 2:
        return 0.0
    
    # Extraer bigramas no ordenados de la query
    query_bigrams = extract_unordered_bigrams(query_tokens, window)
    if not query_bigrams:
        return 0.0
    
    # Extraer bigramas no ordenados del chunk
    chunk_bigrams = extract_unordered_bigrams(chunk_tokens, window)
    
    # Contar coincidencias
    matches = 0
    for bigram in query_bigrams:
        if bigram in chunk_bigrams:
            matches += 1
    
    # Puntuación basada en proporción de bigramas coincidentes
    return matches / len(query_bigrams) if query_bigrams else 0.0


def mrf_window_score(query: str, chunk: Chunk, window_sizes: List[int] = None) -> Dict[str, float]:
    """
    Calcula puntuaciones MRF para diferentes tamaños de ventana.
    
    Args:
        query: Texto de la query
        chunk: Chunk del documento
        window_sizes: Lista de tamaños de ventana a probar
        
    Returns:
        Diccionario con puntuaciones para cada tamaño de ventana
    """
    if window_sizes is None:
        window_sizes = [4, 8, 16, 32]
    
    scores = {}
    for window in window_sizes:
        scores[f'window_{window}'] = _compute_unordered_bigram_score(
            tokenize(query), tokenize(chunk.text), window
        )
    
    return scores


def mrf_position_aware_score(query: str, chunk: Chunk, window: int = 8) -> float:
    """
    Calcula puntuación MRF considerando posiciones de términos.
    
    Args:
        query: Texto de la query
        chunk: Chunk del documento
        window: Tamaño de ventana
        
    Returns:
        Puntuación MRF con información de posiciones
    """
    query_tokens = tokenize(query)
    chunk_tokens = tokenize(chunk.text)
    
    if len(query_tokens) < 2 or len(chunk_tokens) < 2:
        return 0.0
    
    # Crear índice de posiciones de términos en el chunk
    term_positions = create_term_positions(chunk)
    
    # Calcular puntuación considerando proximidad
    score = 0.0
    for i in range(len(query_tokens) - 1):
        term1, term2 = query_tokens[i], query_tokens[i + 1]
        
        if term1 in term_positions and term2 in term_positions:
            # Encontrar la distancia mínima entre posiciones
            min_distance = float('inf')
            for pos1 in term_positions[term1]:
                for pos2 in term_positions[term2]:
                    distance = abs(pos2 - pos1)
                    if distance <= window:
                        min_distance = min(min_distance, distance)
            
            if min_distance != float('inf'):
                # Puntuación inversamente proporcional a la distancia
                score += 1.0 / (1.0 + min_distance)
    
    # Normalizar por número de bigramas
    num_bigrams = len(query_tokens) - 1
    return score / num_bigrams if num_bigrams > 0 else 0.0


def mrf_adaptive_weights(query: str, chunk: Chunk, 
                         base_weights: Tuple[float, float, float] = None) -> Tuple[float, float, float]:
    """
    Calcula pesos adaptativos para MRF basados en características de la query y chunk.
    
    Args:
        query: Texto de la query
        chunk: Chunk del documento
        base_weights: Pesos base (w_unigram, w_ordered, w_unordered)
        
    Returns:
        Tupla con pesos adaptativos
    """
    if base_weights is None:
        base_weights = (0.8, 0.1, 0.1)
    
    w_u, w_o, w_w = base_weights
    
    query_tokens = tokenize(query)
    chunk_tokens = tokenize(chunk.text)
    
    # Ajustar pesos basado en longitud de query
    if len(query_tokens) == 1:
        # Solo un término, enfatizar unigram
        w_u, w_o, w_w = 1.0, 0.0, 0.0
    elif len(query_tokens) == 2:
        # Dos términos, enfatizar bigramas
        w_u, w_o, w_w = 0.3, 0.6, 0.1
    else:
        # Query larga, mantener pesos base
        pass
    
    # Ajustar basado en densidad de términos en el chunk
    query_terms_in_chunk = sum(1 for term in query_tokens if term in set(chunk_tokens))
    coverage = query_terms_in_chunk / len(query_tokens) if query_tokens else 0.0
    
    if coverage < 0.5:
        # Baja cobertura, enfatizar unigram
        w_u *= 1.5
        w_o *= 0.5
        w_w *= 0.5
    
    # Normalizar pesos
    total = w_u + w_o + w_w
    if total > 0:
        w_u /= total
        w_o /= total
        w_w /= total
    
    return w_u, w_o, w_w


def mrf_enhanced_score(query: str, chunk: Chunk, window: int = 8, 
                      use_adaptive_weights: bool = True) -> float:
    """
    Calcula puntuación MRF mejorada con pesos adaptativos y consideración de posiciones.
    
    Args:
        query: Texto de la query
        chunk: Chunk del documento
        window: Tamaño de ventana
        use_adaptive_weights: Si usar pesos adaptativos
        
    Returns:
        Puntuación MRF mejorada
    """
    if use_adaptive_weights:
        w_u, w_o, w_w = mrf_adaptive_weights(query, chunk)
    else:
        w_u, w_o, w_w = 0.8, 0.1, 0.1
    
    # Calcular puntuaciones base
    unigram_score = _compute_unigram_score(tokenize(query), tokenize(chunk.text))
    ordered_score = _compute_ordered_bigram_score(tokenize(query), tokenize(chunk.text))
    unordered_score = _compute_unordered_bigram_score(tokenize(query), tokenize(chunk.text), window)
    
    # Puntuación posicional como bonus
    position_bonus = mrf_position_aware_score(query, chunk, window)
    
    # Combinar puntuaciones
    base_score = w_u * unigram_score + w_o * ordered_score + w_w * unordered_score
    enhanced_score = base_score + 0.1 * position_bonus
    
    return enhanced_score
