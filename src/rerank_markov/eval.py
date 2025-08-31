"""
Métricas de evaluación para el re-ranker híbrido.
Implementa Recall@k, nDCG@k, MRR y comparación con baseline.
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from .types import ScoredChunk, Chunk
from .fusion import rerank
from .config import RerankConfig


def evaluate_reranker(query: str, seed_chunks: List[Chunk], candidate_chunks: List[Chunk],
                     relevant_ids: Set[str], config: RerankConfig, k_values: List[int] = None) -> Dict:
    """
    Evalúa el re-ranker híbrido comparándolo con el baseline de embeddings.
    
    Args:
        query: Texto de la query
        seed_chunks: Chunks semilla
        candidate_chunks: Chunks candidatos
        relevant_ids: Conjunto de IDs de chunks relevantes
        config: Configuración del re-ranker
        k_values: Lista de valores k para evaluar
        
    Returns:
        Diccionario con métricas de evaluación
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 30]
    
    # 1. Baseline: solo embeddings (ordenar por similitud coseno)
    baseline_results = _baseline_embedding_ranking(query, candidate_chunks)
    
    # 2. Re-ranker híbrido
    hybrid_results = rerank(query, seed_chunks, candidate_chunks, config)
    
    # 3. Calcular métricas para ambos
    baseline_metrics = _calculate_metrics(baseline_results, relevant_ids, k_values)
    hybrid_metrics = _calculate_metrics(hybrid_results, relevant_ids, k_values)
    
    # 4. Calcular mejoras
    improvements = _calculate_improvements(baseline_metrics, hybrid_metrics)
    
    return {
        'query': query,
        'num_relevant': len(relevant_ids),
        'num_candidates': len(candidate_chunks),
        'baseline_metrics': baseline_metrics,
        'hybrid_metrics': hybrid_metrics,
        'improvements': improvements,
        'k_values': k_values
    }


def _baseline_embedding_ranking(query: str, chunks: List[Chunk]) -> List[ScoredChunk]:
    """
    Crea ranking baseline usando solo similitud de embeddings.
    
    Args:
        query: Texto de la query
        chunks: Lista de chunks
        
    Returns:
        Lista de ScoredChunk ordenados por similitud de embeddings
    """
    # Simular embedding de la query (fijo para consistencia)
    query_embedding = np.ones(384) * 0.5
    
    # Calcular similitudes
    scored_chunks = []
    for chunk in chunks:
        if chunk.embedding is not None and chunk.embedding.size > 0:
            embedding_score = _cosine_similarity(query_embedding, chunk.embedding)
        else:
            embedding_score = 0.1
        
        scored_chunk = ScoredChunk(
            chunk=chunk,
            total_score=embedding_score,
            embedding_score=embedding_score,
            ppr_score=0.0,
            qlm_score=0.0,
            mrf_score=0.0
        )
        scored_chunks.append(scored_chunk)
    
    # Ordenar por puntuación descendente
    scored_chunks.sort(key=lambda x: x.total_score, reverse=True)
    
    # Asignar rankings
    for i, scored_chunk in enumerate(scored_chunks):
        scored_chunk.rank = i + 1
    
    return scored_chunks


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula similitud coseno entre dos vectores."""
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def _calculate_metrics(results: List[ScoredChunk], relevant_ids: Set[str], 
                      k_values: List[int]) -> Dict:
    """
    Calcula métricas de evaluación para diferentes valores de k.
    
    Args:
        results: Lista de ScoredChunk ordenados
        relevant_ids: Conjunto de IDs relevantes
        k_values: Lista de valores k
        
    Returns:
        Diccionario con métricas
    """
    metrics = {}
    
    for k in k_values:
        top_k = results[:k]
        top_k_ids = {sc.chunk.id for sc in top_k}
        
        # Recall@k
        recall = len(top_k_ids & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0
        
        # Precision@k
        precision = len(top_k_ids & relevant_ids) / k if k > 0 else 0.0
        
        # nDCG@k
        ndcg = _calculate_ndcg(top_k, relevant_ids, k)
        
        # F1@k
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[f'k_{k}'] = {
            'recall': recall,
            'precision': precision,
            'ndcg': ndcg,
            'f1': f1
        }
    
    # MRR (Mean Reciprocal Rank)
    mrr = _calculate_mrr(results, relevant_ids)
    metrics['mrr'] = mrr
    
    return metrics


def _calculate_ndcg(results: List[ScoredChunk], relevant_ids: Set[str], k: int) -> float:
    """
    Calcula nDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        results: Lista de ScoredChunk ordenados
        relevant_ids: Conjunto de IDs relevantes
        k: Valor de k
        
    Returns:
        Valor de nDCG@k
    """
    if k == 0:
        return 0.0
    
    # DCG
    dcg = 0.0
    for i, scored_chunk in enumerate(results[:k]):
        relevance = 1.0 if scored_chunk.chunk.id in relevant_ids else 0.0
        dcg += relevance / np.log2(i + 2)  # i+2 porque log2(1) = 0
    
    # IDCG (ideal DCG)
    idcg = 0.0
    num_relevant = min(k, len(relevant_ids))
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)
    
    # nDCG
    return dcg / idcg if idcg > 0 else 0.0


def _calculate_mrr(results: List[ScoredChunk], relevant_ids: Set[str]) -> float:
    """
    Calcula MRR (Mean Reciprocal Rank).
    
    Args:
        results: Lista de ScoredChunk ordenados
        relevant_ids: Conjunto de IDs relevantes
        
    Returns:
        Valor de MRR
    """
    if not relevant_ids:
        return 0.0
    
    reciprocal_ranks = []
    
    for scored_chunk in results:
        if scored_chunk.chunk.id in relevant_ids:
            reciprocal_ranks.append(1.0 / scored_chunk.rank)
            # Solo considerar el primer resultado relevante
            break
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def _calculate_improvements(baseline_metrics: Dict, hybrid_metrics: Dict) -> Dict:
    """
    Calcula mejoras porcentuales del híbrido sobre el baseline.
    
    Args:
        baseline_metrics: Métricas del baseline
        hybrid_metrics: Métricas del híbrido
        
    Returns:
        Diccionario con mejoras porcentuales
    """
    improvements = {}
    
    # Métricas por k
    for k_key in baseline_metrics:
        if k_key == 'mrr':
            continue
        
        baseline = baseline_metrics[k_key]
        hybrid = hybrid_metrics[k_key]
        
        k_improvements = {}
        for metric in ['recall', 'precision', 'ndcg', 'f1']:
            baseline_val = baseline[metric]
            hybrid_val = hybrid[metric]
            
            if baseline_val > 0:
                improvement = ((hybrid_val - baseline_val) / baseline_val) * 100
            else:
                improvement = 0.0 if hybrid_val == 0 else float('inf')
            
            k_improvements[metric] = improvement
        
        improvements[k_key] = k_improvements
    
    # MRR
    baseline_mrr = baseline_metrics.get('mrr', 0.0)
    hybrid_mrr = hybrid_metrics.get('mrr', 0.0)
    
    if baseline_mrr > 0:
        mrr_improvement = ((hybrid_mrr - baseline_mrr) / baseline_mrr) * 100
    else:
        mrr_improvement = 0.0 if hybrid_mrr == 0 else float('inf')
    
    improvements['mrr'] = mrr_improvement
    
    return improvements


def evaluate_multiple_queries(queries_data: List[Dict], config: RerankConfig,
                             k_values: List[int] = None) -> Dict:
    """
    Evalúa múltiples queries y calcula métricas agregadas.
    
    Args:
        queries_data: Lista de diccionarios con datos de queries
        config: Configuración del re-ranker
        k_values: Lista de valores k
        
    Returns:
        Diccionario con métricas agregadas
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 30]
    
    all_results = []
    
    for query_data in queries_data:
        query = query_data['query']
        seed_chunks = query_data['seed_chunks']
        candidate_chunks = query_data['candidate_chunks']
        relevant_ids = set(query_data['relevant_ids'])
        
        result = evaluate_reranker(
            query, seed_chunks, candidate_chunks, relevant_ids, config, k_values
        )
        all_results.append(result)
    
    # Calcular métricas agregadas
    aggregated_metrics = _aggregate_metrics(all_results, k_values)
    
    return {
        'individual_results': all_results,
        'aggregated_metrics': aggregated_metrics,
        'num_queries': len(queries_data),
        'k_values': k_values
    }


def _aggregate_metrics(all_results: List[Dict], k_values: List[int]) -> Dict:
    """
    Agrega métricas de múltiples queries.
    
    Args:
        all_results: Lista de resultados de evaluación
        k_values: Lista de valores k
        
    Returns:
        Diccionario con métricas agregadas
    """
    if not all_results:
        return {}
    
    aggregated = {}
    
    # Métricas por k
    for k_key in [f'k_{k}' for k in k_values]:
        k_metrics = []
        for result in all_results:
            if k_key in result['hybrid_metrics']:
                k_metrics.append(result['hybrid_metrics'][k_key])
        
        if k_metrics:
            aggregated[k_key] = {}
            for metric in ['recall', 'precision', 'ndcg', 'f1']:
                values = [km[metric] for km in k_metrics]
                aggregated[k_key][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    # MRR agregado
    mrr_values = [result['hybrid_metrics'].get('mrr', 0.0) for result in all_results]
    aggregated['mrr'] = {
        'mean': np.mean(mrr_values),
        'std': np.std(mrr_values),
        'min': np.min(mrr_values),
        'max': np.max(mrr_values)
    }
    
    # Mejoras agregadas
    improvements = []
    for result in all_results:
        if 'improvements' in result:
            improvements.append(result['improvements'])
    
    if improvements:
        aggregated['improvements'] = _aggregate_improvements(improvements, k_values)
    
    return aggregated


def _aggregate_improvements(improvements: List[Dict], k_values: List[int]) -> Dict:
    """
    Agrega mejoras de múltiples queries.
    
    Args:
        improvements: Lista de diccionarios de mejoras
        k_values: Lista de valores k
        
    Returns:
        Diccionario con mejoras agregadas
    """
    aggregated_improvements = {}
    
    # Métricas por k
    for k_key in [f'k_{k}' for k in k_values]:
        k_improvements = []
        for imp in improvements:
            if k_key in imp:
                k_improvements.append(imp[k_key])
        
        if k_improvements:
            aggregated_improvements[k_key] = {}
            for metric in ['recall', 'precision', 'ndcg', 'f1']:
                values = [ki[metric] for ki in k_improvements if metric in ki]
                if values:
                    # Filtrar valores infinitos
                    finite_values = [v for v in values if not np.isinf(v)]
                    if finite_values:
                        aggregated_improvements[k_key][metric] = {
                            'mean': np.mean(finite_values),
                            'std': np.std(finite_values),
                            'min': np.min(finite_values),
                            'max': np.max(finite_values)
                        }
    
    # MRR
    mrr_improvements = [imp.get('mrr', 0.0) for imp in improvements]
    finite_mrr = [v for v in mrr_improvements if not np.isinf(v)]
    if finite_mrr:
        aggregated_improvements['mrr'] = {
            'mean': np.mean(finite_mrr),
            'std': np.std(finite_mrr),
            'min': np.min(finite_mrr),
            'max': np.max(finite_mrr)
        }
    
    return aggregated_improvements


def print_evaluation_summary(evaluation_results: Dict) -> None:
    """
    Imprime un resumen de la evaluación en formato tabular.
    
    Args:
        evaluation_results: Resultados de evaluación
    """
    print("\n" + "="*80)
    print("RESUMEN DE EVALUACIÓN DEL RE-RANKER HÍBRIDO")
    print("="*80)
    
    if 'individual_results' in evaluation_results:
        num_queries = len(evaluation_results['individual_results'])
        print(f"Número de queries evaluadas: {num_queries}")
        print()
    
    if 'aggregated_metrics' in evaluation_results:
        aggregated = evaluation_results['aggregated_metrics']
        
        # Tabla de métricas por k
        print("MÉTRICAS AGREGADAS:")
        print("-" * 80)
        print(f"{'k':<5} {'Recall':<12} {'Precision':<12} {'nDCG':<12} {'F1':<12}")
        print("-" * 80)
        
        for k_key in sorted(aggregated.keys()):
            if k_key.startswith('k_'):
                k = k_key[2:]  # Extraer número de k
                metrics = aggregated[k_key]
                
                recall = metrics['recall']['mean']
                precision = metrics['precision']['mean']
                ndcg = metrics['ndcg']['mean']
                f1 = metrics['f1']['mean']
                
                print(f"{k:<5} {recall:<12.4f} {precision:<12.4f} {ndcg:<12.4f} {f1:<12.4f}")
        
        print("-" * 80)
        
        # MRR
        if 'mrr' in aggregated:
            mrr = aggregated['mrr']['mean']
            print(f"MRR: {mrr:.4f}")
        
        # Mejoras
        if 'improvements' in aggregated:
            print("\nMEJORAS SOBRE BASELINE (EMBEDDINGS):")
            print("-" * 80)
            print(f"{'k':<5} {'Recall':<12} {'Precision':<12} {'nDCG':<12} {'F1':<12}")
            print("-" * 80)
            
            for k_key in sorted(aggregated['improvements'].keys()):
                if k_key.startswith('k_'):
                    k = k_key[2:]
                    improvements = aggregated['improvements'][k_key]
                    
                    recall_imp = improvements.get('recall', {}).get('mean', 0.0)
                    precision_imp = improvements.get('precision', {}).get('mean', 0.0)
                    ndcg_imp = improvements.get('ndcg', {}).get('mean', 0.0)
                    f1_imp = improvements.get('f1', {}).get('mean', 0.0)
                    
                    print(f"{k:<5} {recall_imp:<+11.1f}% {precision_imp:<+11.1f}% {ndcg_imp:<+11.1f}% {f1_imp:<+11.1f}%")
            
            print("-" * 80)
            
            # MRR improvement
            if 'mrr' in aggregated['improvements']:
                mrr_imp = aggregated['improvements']['mrr']['mean']
                print(f"MRR: {mrr_imp:+.1f}%")
    
    print("="*80)
