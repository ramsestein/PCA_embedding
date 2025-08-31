"""
Construcción del grafo de chunks y Personalized PageRank (PPR).
Implementa random walk con reinicio (RWR) para re-ranking.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from .types import Chunk
from .utils import cosine_similarity


def build_chunk_graph(candidates: List[Chunk], alpha: float = 0.6, 
                     beta: float = 0.25, gamma: float = 0.15) -> nx.DiGraph:
    """
    Construye grafo dirigido de chunks con aristas ponderadas.
    
    Las aristas combinan:
    - Similitud coseno de embeddings (alpha)
    - Contigüidad dentro del mismo documento (beta)
    - Enlaces/referencias cruzadas (gamma)
    
    Args:
        candidates: Lista de chunks candidatos
        alpha: Peso para similitud de embeddings
        beta: Peso para contigüidad
        gamma: Peso para enlaces
        
    Returns:
        Grafo dirigido NetworkX
    """
    if not candidates:
        return nx.DiGraph()
    
    G = nx.DiGraph()
    
    # Añadir nodos
    for chunk in candidates:
        G.add_node(chunk.id, chunk=chunk)
    
    # Calcular pesos de aristas
    edges = []
    
    for i, chunk1 in enumerate(candidates):
        for j, chunk2 in enumerate(candidates):
            if i == j:
                continue
            
            weight = 0.0
            
            # 1. Similitud de embeddings
            if (chunk1.embedding is not None and chunk2.embedding is not None and 
                chunk1.embedding.size > 0 and chunk2.embedding.size > 0):
                sim_emb = cosine_similarity(chunk1.embedding, chunk2.embedding)
                weight += alpha * sim_emb
            
            # 2. Contigüidad dentro del mismo documento
            if chunk1.doc_id == chunk2.doc_id:
                position_diff = abs(chunk1.position - chunk2.position)
                if position_diff == 1:  # Chunks consecutivos
                    contig_score = 1.0
                else:
                    # Decay exponencial con distancia
                    contig_score = np.exp(-position_diff / 5.0)
                weight += beta * contig_score
            
            # 3. Enlaces/referencias cruzadas
            if 'links' in chunk1.meta and chunk2.id in chunk1.meta['links']:
                weight += gamma * 1.0
            if 'links' in chunk2.meta and chunk1.id in chunk2.meta['links']:
                weight += gamma * 1.0
            
            # Solo añadir arista si hay peso significativo
            if weight > 0.01:
                edges.append((chunk1.id, chunk2.id, weight))
    
    # Añadir aristas al grafo
    G.add_weighted_edges_from(edges)
    
    return G


def personalized_pagerank(G: nx.DiGraph, seed_ids: List[str], lambda_: float = 0.85, 
                         tol: float = 1e-6, max_iter: int = 100) -> Dict[str, float]:
    """
    Calcula Personalized PageRank con random walk con reinicio.
    
    Fórmula: π = (1-λ) * e_S + λ * P * π
    
    Args:
        G: Grafo dirigido NetworkX
        seed_ids: IDs de chunks semilla (top-K por embeddings)
        lambda_: Factor de damping (probabilidad de continuar el paseo)
        tol: Tolerancia de convergencia
        max_iter: Máximo de iteraciones
        
    Returns:
        Diccionario {chunk_id: puntuación PPR}
    """
    if not G.nodes() or not seed_ids:
        return {}
    
    # Crear vector de reinicio uniforme sobre semillas
    restart_vector = {}
    for node in G.nodes():
        if node in seed_ids:
            restart_vector[node] = 1.0 / len(seed_ids)
        else:
            restart_vector[node] = 0.0
    
    # Crear matriz de transición P
    P = _create_transition_matrix(G)
    
    # Inicializar vector de estado π
    pi = restart_vector.copy()
    
    # Iterar hasta convergencia
    for iteration in range(max_iter):
        pi_old = pi.copy()
        
        # π = (1-λ) * e_S + λ * P * π
        pi_new = {}
        for node in G.nodes():
            # Componente de reinicio
            restart_component = (1 - lambda_) * restart_vector[node]
            
            # Componente de transición
            transition_component = 0.0
            for pred in G.predecessors(node):
                if pred in pi_old:
                    # Obtener peso de la arista
                    edge_weight = G[pred][node].get('weight', 0.0)
                    transition_component += lambda_ * edge_weight * pi_old[pred]
            
            pi_new[node] = restart_component + transition_component
        
        pi = pi_new
        
        # Normalizar para que sume 1.0
        total = sum(pi.values())
        if total > 0:
            for node in pi:
                pi[node] /= total
        
        # Verificar convergencia
        if _l1_norm_diff(pi_old, pi) < tol:
            break
    
    return pi


def _create_transition_matrix(G: nx.DiGraph) -> Dict[Tuple[str, str], float]:
    """
    Crea matriz de transición P normalizada por filas.
    
    Args:
        G: Grafo dirigido NetworkX
        
    Returns:
        Diccionario {(origen, destino): probabilidad}
    """
    P = {}
    
    for source in G.nodes():
        # Obtener pesos de aristas salientes
        out_edges = list(G.out_edges(source, data='weight'))
        
        if not out_edges:
            # Nodo sin aristas salientes - self-loop
            P[(source, source)] = 1.0
            continue
        
        # Calcular suma de pesos
        total_weight = sum(weight for _, _, weight in out_edges)
        
        if total_weight > 0:
            # Normalizar por fila
            for _, target, weight in out_edges:
                P[(source, target)] = weight / total_weight
        else:
            # Fila nula - self-loop
            P[(source, source)] = 1.0
    
    return P


def _l1_norm_diff(dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
    """
    Calcula diferencia L1 entre dos diccionarios.
    
    Args:
        dict1, dict2: Diccionarios a comparar
        
    Returns:
        Diferencia L1
    """
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    diff = 0.0
    for key in all_keys:
        val1 = dict1.get(key, 0.0)
        val2 = dict2.get(key, 0.0)
        diff += abs(val1 - val2)
    
    return diff


def build_enhanced_chunk_graph(candidates: List[Chunk], alpha: float = 0.6,
                              beta: float = 0.25, gamma: float = 0.15,
                              use_semantic_similarity: bool = True) -> nx.DiGraph:
    """
    Construye grafo mejorado con características adicionales.
    
    Args:
        candidates: Lista de chunks candidatos
        alpha: Peso para similitud de embeddings
        beta: Peso para contigüidad
        gamma: Peso para enlaces
        use_semantic_similarity: Si usar similitud semántica mejorada
        
    Returns:
        Grafo dirigido NetworkX mejorado
    """
    G = build_chunk_graph(candidates, alpha, beta, gamma)
    
    if use_semantic_similarity:
        _add_semantic_edges(G, candidates)
    
    return G


def _add_semantic_edges(G: nx.DiGraph, candidates: List[Chunk]) -> None:
    """
    Añade aristas semánticas adicionales basadas en similitud de contenido.
    
    Args:
        G: Grafo a mejorar
        candidates: Lista de chunks candidatos
    """
    # Agrupar chunks por documento
    doc_chunks = {}
    for chunk in candidates:
        if chunk.doc_id not in doc_chunks:
            doc_chunks[chunk.doc_id] = []
        doc_chunks[chunk.doc_id].append(chunk)
    
    # Añadir aristas semánticas entre chunks del mismo documento
    for doc_id, chunks in doc_chunks.items():
        if len(chunks) < 2:
            continue
        
        # Ordenar por posición
        chunks_sorted = sorted(chunks, key=lambda x: x.position)
        
        # Añadir aristas entre chunks cercanos del mismo documento
        for i, chunk1 in enumerate(chunks_sorted):
            for j in range(i + 1, min(i + 3, len(chunks_sorted))):  # Ventana de 3
                chunk2 = chunks_sorted[j]
                
                # Calcular similitud semántica
                if (chunk1.embedding is not None and chunk2.embedding is not None):
                    sim = cosine_similarity(chunk1.embedding, chunk2.embedding)
                    
                    # Solo añadir si no existe ya
                    if not G.has_edge(chunk1.id, chunk2.id):
                        G.add_edge(chunk1.id, chunk2.id, weight=sim * 0.3)
                    if not G.has_edge(chunk2.id, chunk1.id):
                        G.add_edge(chunk2.id, chunk1.id, weight=sim * 0.3)


def compute_graph_metrics(G: nx.DiGraph) -> Dict[str, float]:
    """
    Calcula métricas del grafo para análisis.
    
    Args:
        G: Grafo NetworkX
        
    Returns:
        Diccionario con métricas del grafo
    """
    if not G.nodes():
        return {}
    
    metrics = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G) if G.number_of_nodes() > 2 else 0.0,
        'avg_shortest_path': 0.0
    }
    
    # Calcular camino promedio solo si el grafo es fuertemente conectado
    try:
        if nx.is_strongly_connected(G):
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        pass
    
    return metrics


def personalized_pagerank_with_teleport(G: nx.DiGraph, seed_ids: List[str], 
                                       lambda_: float = 0.85, teleport_prob: float = 0.1,
                                       tol: float = 1e-6, max_iter: int = 100) -> Dict[str, float]:
    """
    PPR con teleportación adicional para mejorar exploración.
    
    Args:
        G: Grafo dirigido NetworkX
        seed_ids: IDs de chunks semilla
        lambda_: Factor de damping
        teleport_prob: Probabilidad de teleportación a nodos aleatorios
        tol: Tolerancia de convergencia
        max_iter: Máximo de iteraciones
        
    Returns:
        Diccionario {chunk_id: puntuación PPR}
    """
    if not G.nodes() or not seed_ids:
        return {}
    
    # Vector de reinicio con teleportación
    restart_vector = {}
    for node in G.nodes():
        if node in seed_ids:
            restart_vector[node] = (1 - teleport_prob) / len(seed_ids)
        else:
            restart_vector[node] = teleport_prob / (len(G.nodes()) - len(seed_ids))
    
    # Crear matriz de transición
    P = _create_transition_matrix(G)
    
    # Inicializar vector de estado
    pi = restart_vector.copy()
    
    # Iterar hasta convergencia
    for iteration in range(max_iter):
        pi_old = pi.copy()
        
        # π = (1-λ) * e_S + λ * P * π
        pi_new = {}
        for node in G.nodes():
            restart_component = (1 - lambda_) * restart_vector[node]
            
            transition_component = 0.0
            for pred in G.predecessors(node):
                if pred in pi_old:
                    edge_weight = G[pred][node].get('weight', 0.0)
                    transition_component += lambda_ * edge_weight * pi_old[pred]
            
            pi_new[node] = restart_component + transition_component
        
        pi = pi_new
        
        # Normalizar para que sume 1.0
        total = sum(pi.values())
        if total > 0:
            for node in pi:
                pi[node] /= total
        
        # Verificar convergencia
        if _l1_norm_diff(pi_old, pi) < tol:
            break
    
    return pi
