#!/usr/bin/env python3
"""
CLI para probar y perfilar el re-ranker híbrido.
"""

import argparse
import json
import pickle
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Añadir el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from rerank_markov import (
    Chunk, Query, ScoredChunk, RerankConfig,
    rerank
)
from rerank_markov.eval import evaluate_reranker, print_evaluation_summary


def load_chunks_from_jsonl(file_path: str) -> List[Chunk]:
    """
    Carga chunks desde archivo JSONL.
    
    Args:
        file_path: Ruta al archivo JSONL
        
    Returns:
        Lista de chunks
    """
    chunks = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Convertir embedding si existe
                embedding = None
                if 'embedding' in data and data['embedding']:
                    embedding = np.array(data['embedding'], dtype=np.float32)
                
                chunk = Chunk(
                    id=data['id'],
                    text=data['text'],
                    doc_id=data['doc_id'],
                    position=data.get('position', 0),
                    embedding=embedding,
                    meta=data.get('meta', {})
                )
                chunks.append(chunk)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error en línea {line_num}: {e}")
                continue
    
    return chunks


def load_chunks_from_pickle(file_path: str) -> List[Chunk]:
    """
    Carga chunks desde archivo pickle.
    
    Args:
        file_path: Ruta al archivo pickle
        
    Returns:
        Lista de chunks
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def create_synthetic_chunks(num_chunks: int = 100, embedding_dim: int = 384) -> List[Chunk]:
    """
    Crea chunks sintéticos para pruebas.
    
    Args:
        num_chunks: Número de chunks a crear
        embedding_dim: Dimensión de los embeddings
        
    Returns:
        Lista de chunks sintéticos
    """
    chunks = []
    
    # Textos de ejemplo médicos
    sample_texts = [
        "El paciente presenta síntomas de hipoxemia postoperatoria que requieren ajuste de PEEP.",
        "La administración de medicamentos vía inhalatoria debe realizarse con precaución.",
        "El abordaje integral de la mujer con cáncer ginecológico incluye múltiples especialidades.",
        "La crisis epiléptica generalizada requiere intervención inmediata del personal médico.",
        "La hospitalización a domicilio (HDOM) ofrece atención integral al paciente crónico.",
        "La técnica de magnetoterapia se aplica en casos de dolor crónico y rehabilitación.",
        "El trasplante hepático requiere seguimiento exhaustivo post-alta.",
        "La administración segura de ferroterapia intravenosa previene complicaciones.",
        "El aislamiento social y la soledad afectan la calidad de vida del paciente.",
        "La acogida electiva en pandemia requiere protocolos específicos de seguridad."
    ]
    
    for i in range(num_chunks):
        # Seleccionar texto aleatorio
        text = sample_texts[i % len(sample_texts)]
        
        # Crear embedding aleatorio
        embedding = np.random.rand(embedding_dim).astype(np.float32)
        
        # Crear chunk
        chunk = Chunk(
            id=f"chunk_{i:03d}",
            text=text,
            doc_id=f"doc_{i // 10:02d}",
            position=i % 10,
            embedding=embedding,
            meta={
                'length': len(text),
                'tokens': len(text.split()),
                'category': f"cat_{i % 5}"
            }
        )
        chunks.append(chunk)
    
    return chunks


def create_synthetic_qrels(chunks: List[Chunk], num_queries: int = 5) -> List[Dict]:
    """
    Crea qrels sintéticos para evaluación.
    
    Args:
        chunks: Lista de chunks
        num_queries: Número de queries a crear
        
    Returns:
        Lista de qrels sintéticos
    """
    queries = [
        "hipoxemia postoperatoria PEEP",
        "medicamentos vía inhalatoria",
        "cáncer ginecológico tratamiento",
        "crisis epiléptica emergencia",
        "hospitalización domicilio HDOM"
    ]
    
    qrels = []
    
    for i, query in enumerate(queries[:num_queries]):
        # Seleccionar chunks relevantes aleatoriamente (20-30% de los chunks)
        num_relevant = max(1, len(chunks) // 5)
        relevant_chunks = np.random.choice(chunks, num_relevant, replace=False)
        
        qrel = {
            'query_id': f"q_{i:02d}",
            'query': query,
            'seed_chunks': chunks[:10],  # Primeros 10 como semilla
            'candidate_chunks': chunks,
            'relevant_ids': [chunk.id for chunk in relevant_chunks]
        }
        qrels.append(qrel)
    
    return qrels


def print_ranking_table(scored_chunks: List[ScoredChunk], max_results: int = 20) -> None:
    """
    Imprime tabla de ranking con todas las puntuaciones.
    
    Args:
        scored_chunks: Lista de chunks puntuados
        max_results: Máximo número de resultados a mostrar
    """
    print("\n" + "="*120)
    print("RANKING DEL RE-RANKER HÍBRIDO")
    print("="*120)
    print(f"{'Rank':<5} {'ID':<12} {'Score Total':<12} {'Embedding':<12} {'PPR':<12} {'QLM':<12} {'MRF':<12}")
    print("-"*120)
    
    for i, scored_chunk in enumerate(scored_chunks[:max_results]):
        print(f"{scored_chunk.rank:<5} {scored_chunk.chunk.id:<12} "
              f"{scored_chunk.total_score:<12.4f} {scored_chunk.embedding_score:<12.4f} "
              f"{scored_chunk.ppr_score:<12.4f} {scored_chunk.qlm_score:<12.4f} "
              f"{scored_chunk.mrf_score:<12.4f}")
    
    if len(scored_chunks) > max_results:
        print(f"... y {len(scored_chunks) - max_results} resultados más")
    
    print("="*120)


def main():
    """Función principal del CLI."""
    parser = argparse.ArgumentParser(
        description="CLI para probar el re-ranker híbrido",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Re-ranking básico
  python rerank_cli.py --query "hipoxemia postoperatoria PEEP" --k 30 --kprime 100
  
  # Con archivo de chunks
  python rerank_cli.py --query "medicamentos inhalatorios" --chunks chunks.jsonl --k 20
  
  # Evaluación con qrels
  python rerank_cli.py --eval qrels.json --k 10 --kprime 50
  
  # Configuración personalizada
  python rerank_cli.py --query "crisis epiléptica" --mu 2000 --lambda_ppr 0.9 --a 0.5 --b 0.3 --c 0.15 --d 0.05
        """
    )
    
    # Argumentos básicos
    parser.add_argument('--query', type=str, help='Query de búsqueda')
    parser.add_argument('--chunks', type=str, help='Archivo con chunks (JSONL o pickle)')
    parser.add_argument('--eval', type=str, help='Archivo con qrels para evaluación')
    
    # Parámetros de configuración
    parser.add_argument('--k', type=int, default=30, help='Número de chunks a retornar')
    parser.add_argument('--kprime', type=int, default=100, help='Número de candidatos para PPR')
    parser.add_argument('--mu', type=int, default=1500, help='Parámetro de suavizado Dirichlet')
    parser.add_argument('--lambda_ppr', type=float, default=0.85, help='Factor de damping para PPR')
    
    # Pesos de fusión
    parser.add_argument('--a', type=float, default=0.45, help='Peso para embeddings')
    parser.add_argument('--b', type=float, default=0.25, help='Peso para PPR')
    parser.add_argument('--c', type=float, default=0.20, help='Peso para QLM')
    parser.add_argument('--d', type=float, default=0.10, help='Peso para MRF')
    
    # Parámetros del grafo
    parser.add_argument('--alpha', type=float, default=0.6, help='Peso para similitud de embeddings')
    parser.add_argument('--beta', type=float, default=0.25, help='Peso para contigüidad')
    parser.add_argument('--gamma', type=float, default=0.15, help='Peso para enlaces')
    
    # Parámetros MRF
    parser.add_argument('--window', type=int, default=8, help='Tamaño de ventana para MRF')
    parser.add_argument('--w_unigram', type=float, default=0.8, help='Peso unigram para MRF')
    parser.add_argument('--w_ordered', type=float, default=0.1, help='Peso bigram ordenado para MRF')
    parser.add_argument('--w_unordered', type=float, default=0.1, help='Peso bigram no ordenado para MRF')
    
    # Opciones adicionales
    parser.add_argument('--synthetic', action='store_true', help='Usar datos sintéticos')
    parser.add_argument('--num_chunks', type=int, default=100, help='Número de chunks sintéticos')
    parser.add_argument('--analysis', action='store_true', help='Mostrar análisis detallado')
    parser.add_argument('--verbose', action='store_true', help='Modo verboso')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.query and not args.eval:
        parser.error("Debe especificar --query o --eval")
    
    if args.synthetic and args.chunks:
        parser.error("No puede usar --synthetic y --chunks simultáneamente")
    
    # Crear configuración
    config = RerankConfig(
        k=args.k,
        k_prime=args.kprime,
        mu=args.mu,
        lambda_ppr=args.lambda_ppr,
        a=args.a,
        b=args.b,
        c=args.c,
        d=args.d,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        window_size=args.window,
        w_unigram=args.w_unigram,
        w_ordered=args.w_ordered,
        w_unordered=args.w_unordered
    )
    
    print("Configuración del re-ranker:")
    print(f"  k: {config.k}, k': {config.k_prime}")
    print(f"  μ: {config.mu}, λ_ppr: {config.lambda_ppr}")
    print(f"  Pesos: a={config.a}, b={config.b}, c={config.c}, d={config.d}")
    print(f"  Grafo: α={config.alpha}, β={config.beta}, γ={config.gamma}")
    print(f"  MRF: ventana={config.window_size}, w_u={config.w_unigram}, w_o={config.w_ordered}, w_w={config.w_unordered}")
    print()
    
    # Cargar o crear chunks
    if args.synthetic:
        print("Creando chunks sintéticos...")
        chunks = create_synthetic_chunks(args.num_chunks)
        print(f"Creados {len(chunks)} chunks sintéticos")
    elif args.chunks:
        print(f"Cargando chunks desde {args.chunks}...")
        if args.chunks.endswith('.jsonl'):
            chunks = load_chunks_from_jsonl(args.chunks)
        else:
            chunks = load_chunks_from_pickle(args.chunks)
        print(f"Cargados {len(chunks)} chunks")
    else:
        print("Creando chunks sintéticos por defecto...")
        chunks = create_synthetic_chunks(100)
        print(f"Creados {len(chunks)} chunks sintéticos")
    
    if not chunks:
        print("Error: No se pudieron cargar chunks")
        return 1
    
    # Ejecutar evaluación o re-ranking
    if args.eval:
        # Modo evaluación
        print(f"Ejecutando evaluación con {args.eval}...")
        
        if args.eval.endswith('.json'):
            with open(args.eval, 'r', encoding='utf-8') as f:
                qrels_data = json.load(f)
        elif args.eval.endswith('.jsonl'):
            # Cargar qrels desde JSONL
            qrels_data = []
            with open(args.eval, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        qrels_data.append(json.loads(line.strip()))
        else:
            # Crear qrels sintéticos
            qrels_data = create_synthetic_qrels(chunks, 5)
        
        # Ejecutar evaluación para cada query en los qrels
        start_time = time.time()
        all_evaluation_results = []
        
        for qrel in qrels_data:
            query = qrel['query']
            relevant_ids = set(qrel['relevant_ids'])
            
            # Usar chunks sintéticos como candidatos
            seed_chunks = chunks[:min(config.k, len(chunks))]
            candidate_chunks = chunks[:min(config.k_prime, len(chunks))]
            
            evaluation_result = evaluate_reranker(
                query, seed_chunks, candidate_chunks, relevant_ids, config
            )
            all_evaluation_results.append(evaluation_result)
        
        end_time = time.time()
        
        print(f"Evaluación completada en {end_time - start_time:.2f} segundos")
        
        # Calcular métricas agregadas
        from rerank_markov.eval import evaluate_multiple_queries
        aggregated_results = evaluate_multiple_queries(
            [{'query': qrel['query'], 'seed_chunks': chunks[:min(config.k, len(chunks))], 
              'candidate_chunks': chunks[:min(config.k_prime, len(chunks))], 
              'relevant_ids': qrel['relevant_ids']} for qrel in qrels_data],
            config
        )
        
        print_evaluation_summary(aggregated_results)
        
    else:
        # Modo re-ranking
        print(f"Ejecutando re-ranking para query: '{args.query}'")
        
        # Seleccionar chunks semilla y candidatos
        seed_chunks = chunks[:min(config.k, len(chunks))]
        candidate_chunks = chunks[:min(config.k_prime, len(chunks))]
        
        print(f"Chunks semilla: {len(seed_chunks)}")
        print(f"Chunks candidatos: {len(candidate_chunks)}")
        
        # Ejecutar re-ranking
        start_time = time.time()
        
        if args.analysis:
            results = rerank_with_analysis(args.query, seed_chunks, candidate_chunks, config)
            print(f"Análisis completado en {time.time() - start_time:.2f} segundos")
            
            # Mostrar análisis
            print(f"\nAnálisis del grafo:")
            if 'graph_metrics' in results:
                for key, value in results['graph_metrics'].items():
                    print(f"  {key}: {value}")
        else:
            scored_chunks = rerank(args.query, seed_chunks, candidate_chunks, config)
            print(f"Re-ranking completado en {time.time() - start_time:.2f} segundos")
            
            # Mostrar resultados
            print_ranking_table(scored_chunks, args.k)
            
            if args.verbose:
                print(f"\nPrimeros 3 resultados:")
                for i, sc in enumerate(scored_chunks[:3]):
                    print(f"\n{i+1}. {sc.chunk.id}")
                    print(f"   Texto: {sc.chunk.text[:100]}...")
                    print(f"   Documento: {sc.chunk.doc_id}, Posición: {sc.chunk.position}")
                    print(f"   Puntuaciones: Total={sc.total_score:.4f}, Emb={sc.embedding_score:.4f}, "
                          f"PPR={sc.ppr_score:.4f}, QLM={sc.qlm_score:.4f}, MRF={sc.mrf_score:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
