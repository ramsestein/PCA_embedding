#!/usr/bin/env python3
"""
Experimento 2: Evaluación Inicial con All-Mini Base
Evalúa el rendimiento del modelo all-mini-base como baseline.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rerank_markov.types import Chunk, ScoredChunk
from rerank_markov.utils import tokenize
from rerank_markov.index_stats import compute_corpus_statistics
from rerank_markov.qlm import qlm_score
from rerank_markov.mrf import mrf_sd_score
from rerank_markov.fusion import normalize_scores


class AllMiniBaselineExperiment:
    """Experimento que evalúa All-Mini Base como baseline."""
    
    def __init__(self):
        self.chunks = []
        self.benchmark_queries = []
        self.model = None
        self.corpus_stats = None
        
    def load_pnts_documents(self, pnts_dir: str = "PNTs") -> None:
        """Carga documentos PNTs y crea chunks."""
        print(f"Cargando documentos desde: {pnts_dir}")
        
        if not os.path.exists(pnts_dir):
            raise ValueError(f"Directorio {pnts_dir} no encontrado")
        
        chunks = []
        pnts_path = Path(pnts_dir)
        
        for txt_file in pnts_path.glob("*_limpio.txt"):
            print(f"\nProcesando: {txt_file.name}")
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                continue
                
            # Crear chunks del documento
            doc_chunks = self._create_chunks_from_text(
                content, 
                str(txt_file.stem), 
                chunk_size=512, 
                overlap=50
            )
            chunks.extend(doc_chunks)
            
        self.chunks = chunks
        print(f"Total de chunks creados: {len(chunks)}")
        
    def _create_chunks_from_text(self, text: str, doc_id: str, chunk_size: int = 512, overlap: int = 50) -> List[Chunk]:
        """Crea chunks de texto con overlap."""
        tokens = tokenize(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = " ".join(chunk_tokens)
            
            if len(chunk_text.strip()) < 50:  # Filtrar chunks muy pequeños
                continue
                
            chunk = Chunk(
                id=f"chunk_{i:03d}",
                text=chunk_text,
                doc_id=doc_id,
                position=i,
                embedding=None,  # Se calculará después
                meta={"source": doc_id}
            )
            chunks.append(chunk)
            
        return chunks
    
    def load_benchmark_queries(self) -> List[Dict[str, str]]:
        """Carga las queries del archivo de benchmark."""
        benchmark_file = "benchmark/preguntas_con_docs_es.json"
        
        if not os.path.exists(benchmark_file):
            print(f"Archivo de benchmark {benchmark_file} no encontrado")
            return []
            
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        queries = []
        for item in data:
            if 'query' in item and 'document_expected' in item:
                queries.append({
                    'query': item['query'],
                    'expected_doc': item['document_expected']
                })
                
        print(f"Queries cargadas: {len(queries)}")
        return queries
    
    def load_allmini_model(self):
        """Carga el modelo All-Mini Base."""
        print("Cargando modelo All-Mini Base...")
        model_path = "all-mini-base"
        self.model = SentenceTransformer(model_path)
        print("Modelo All-Mini Base cargado exitosamente!")
        
    def compute_embeddings(self):
        """Calcula embeddings para todos los chunks."""
        print("Calculando embeddings con All-Mini Base...")
        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Asignar embeddings
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]
            
        print("Embeddings calculados exitosamente!")
        
    def compute_corpus_statistics(self):
        """Calcula estadísticas del corpus para QLM."""
        print("Calculando estadísticas del corpus...")
        self.corpus_stats = compute_corpus_statistics(self.chunks)
        print("Estadísticas del corpus calculadas!")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def evaluate_solo_embeddings(self, query: str, expected_doc: str, top_k: int = 5) -> Dict[str, Any]:
        """Evalúa la estrategia de solo embeddings."""
        start_time = time.time()
        
        # Normalizar expected_doc para comparación
        expected_doc_normalized = expected_doc.replace('_limpio.txt', '_limpio')
        
        # Calcular embedding de la query
        query_embedding = self.model.encode([query])[0]
        
        # Calcular similitudes con todos los chunks
        similarities = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            else:
                similarity = 0.0
            similarities.append((similarity, chunk))
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Verificar Top1 y Top5
        top1_chunk = similarities[0][1]
        top1_correct = top1_chunk.doc_id == expected_doc_normalized
        
        top5_chunks = [chunk for _, chunk in similarities[:top_k]]
        top5_correct = any(chunk.doc_id == expected_doc_normalized for chunk in top5_chunks)
        
        # Encontrar ranking del documento esperado
        expected_rank = -1
        for i, (_, chunk) in enumerate(similarities):
            if chunk.doc_id == expected_doc_normalized:
                expected_rank = i + 1
                break
        
        search_time = time.time() - start_time
        
        return {
            'strategy': 'Solo Embeddings',
            'query': query,
            'expected_doc': expected_doc,
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'search_time': search_time,
            'expected_rank': expected_rank,
            'top_results': [
                {
                    'rank': i + 1,
                    'doc_id': chunk.doc_id,
                    'score': similarity,
                    'text_preview': chunk.text[:100] + "..."
                }
                for i, (similarity, chunk) in enumerate(similarities[:top_k])
            ]
        }
    
    def evaluate_solo_mrf(self, query: str, expected_doc: str, top_k: int = 5) -> Dict[str, Any]:
        """Evalúa la estrategia de solo MRF."""
        start_time = time.time()
        
        # Normalizar expected_doc para comparación
        expected_doc_normalized = expected_doc.replace('_limpio.txt', '_limpio')
        
        # Calcular puntuaciones MRF para todos los chunks
        scored_chunks = []
        for chunk in self.chunks:
            score = mrf_sd_score(query, chunk, w_unigram=0.7, w_ordered=0.2, w_unordered=0.1)
            scored_chunks.append((score, chunk))
        
        # Ordenar por puntuación
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Verificar Top1 y Top5
        top1_chunk = scored_chunks[0][1]
        top1_correct = top1_chunk.doc_id == expected_doc_normalized
        
        top5_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
        top5_correct = any(chunk.doc_id == expected_doc_normalized for chunk in top5_chunks)
        
        # Encontrar ranking del documento esperado
        expected_rank = -1
        for i, (_, chunk) in enumerate(scored_chunks):
            if chunk.doc_id == expected_doc_normalized:
                expected_rank = i + 1
                break
        
        search_time = time.time() - start_time
        
        return {
            'strategy': 'Solo MRF',
            'query': query,
            'expected_doc': expected_doc,
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'search_time': search_time,
            'expected_rank': expected_rank,
            'top_results': [
                {
                    'rank': i + 1,
                    'doc_id': chunk.doc_id,
                    'score': score,
                    'text_preview': chunk.text[:100] + "..."
                }
                for i, (score, chunk) in enumerate(scored_chunks[:top_k])
            ]
        }
    
    def evaluate_hybrid_mrf_embeddings(self, query: str, expected_doc: str, top_k: int = 5) -> Dict[str, Any]:
        """Evalúa la estrategia híbrida MRF + Embeddings."""
        start_time = time.time()
        
        # Normalizar expected_doc para comparación
        expected_doc_normalized = expected_doc.replace('_limpio.txt', '_limpio')
        
        # Calcular embedding de la query
        query_embedding = self.model.encode([query])[0]
        
        # Calcular puntuaciones híbridas para todos los chunks
        scored_chunks = []
        for chunk in self.chunks:
            # Puntuación de embeddings
            if chunk.embedding is not None:
                embedding_score = self._cosine_similarity(query_embedding, chunk.embedding)
            else:
                embedding_score = 0.0
            
            # Puntuación MRF
            mrf_score = mrf_sd_score(query, chunk, w_unigram=0.7, w_ordered=0.2, w_unordered=0.1)
            
            # Combinar puntuaciones (50% cada una)
            hybrid_score = 0.5 * embedding_score + 0.5 * mrf_score
            
            scored_chunks.append((hybrid_score, chunk, embedding_score, mrf_score))
        
        # Ordenar por puntuación híbrida
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Verificar Top1 y Top5
        top1_chunk = scored_chunks[0][1]
        top1_correct = top1_chunk.doc_id == expected_doc_normalized
        
        top5_chunks = [chunk for _, chunk, _, _ in scored_chunks[:top_k]]
        top5_correct = any(chunk.doc_id == expected_doc_normalized for chunk in top5_chunks)
        
        # Encontrar ranking del documento esperado
        expected_rank = -1
        for i, (_, chunk, _, _) in enumerate(scored_chunks):
            if chunk.doc_id == expected_doc_normalized:
                expected_rank = i + 1
                break
        
        search_time = time.time() - start_time
        
        return {
            'strategy': 'MRF + Embeddings',
            'query': query,
            'expected_doc': expected_doc,
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'search_time': search_time,
            'expected_rank': expected_rank,
            'top_results': [
                {
                    'rank': i + 1,
                    'doc_id': chunk.doc_id,
                    'hybrid_score': hybrid_score,
                    'embedding_score': embedding_score,
                    'mrf_score': mrf_score,
                    'text_preview': chunk.text[:100] + "..."
                }
                for i, (hybrid_score, chunk, embedding_score, mrf_score) in enumerate(scored_chunks[:top_k])
            ]
        }
    
    def run_experiment(self, top_k: int = 5) -> Dict[str, Any]:
        """Ejecuta el experimento completo."""
        print("=== INICIANDO EXPERIMENTO 2: EVALUACIÓN INICIAL CON ALL-MINI BASE ===")
        
        # Cargar documentos y queries
        self.load_pnts_documents()
        self.benchmark_queries = self.load_benchmark_queries()
        
        if not self.benchmark_queries:
            print("No se pudieron cargar las queries de benchmark")
            return {}
        
        # Cargar modelo y calcular embeddings
        self.load_allmini_model()
        self.compute_embeddings()
        self.compute_corpus_statistics()
        
        # Evaluar las tres estrategias
        strategies = ['solo_embeddings', 'solo_mrf', 'hybrid_mrf_embeddings']
        all_results = {}
        
        for strategy in strategies:
            print(f"\n=== Evaluando estrategia: {strategy.upper()} ===")
            
            results = []
            correct_top1 = 0
            correct_top5 = 0
            
            for i, query_data in enumerate(self.benchmark_queries):
                print(f"Evaluando query {i+1}/{len(self.benchmark_queries)}: {query_data['query'][:50]}...")
                
                if strategy == 'solo_embeddings':
                    result = self.evaluate_solo_embeddings(query_data['query'], query_data['expected_doc'], top_k)
                elif strategy == 'solo_mrf':
                    result = self.evaluate_solo_mrf(query_data['query'], query_data['expected_doc'], top_k)
                elif strategy == 'hybrid_mrf_embeddings':
                    result = self.evaluate_hybrid_mrf_embeddings(query_data['query'], query_data['expected_doc'], top_k)
                
                results.append(result)
                
                if result['top1_correct']:
                    correct_top1 += 1
                if result['top5_correct']:
                    correct_top5 += 1
            
            # Calcular métricas agregadas
            total_queries = len(self.benchmark_queries)
            top1_accuracy = correct_top1 / total_queries
            top5_accuracy = correct_top5 / total_queries
            
            # Calcular MRR
            mrr_scores = []
            for result in results:
                if result['expected_rank'] > 0:
                    mrr_scores.append(1.0 / result['expected_rank'])
                else:
                    mrr_scores.append(0.0)
            
            mrr = np.mean(mrr_scores) if mrr_scores else 0.0
            
            strategy_results = {
                'strategy_name': strategy,
                'total_queries': total_queries,
                'top1_correct': correct_top1,
                'top5_correct': correct_top5,
                'top1_accuracy': top1_accuracy,
                'top5_accuracy': top5_accuracy,
                'mrr': mrr,
                'query_results': results
            }
            
            all_results[strategy] = strategy_results
        
        experiment_results = {
            'experiment_name': 'Evaluación Inicial con All-Mini Base',
            'model': 'all-mini-base',
            'total_chunks': len(self.chunks),
            'strategies_evaluated': strategies,
            'results_by_strategy': all_results,
            'timestamp': time.time()
        }
        
        return experiment_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Guarda los resultados del experimento."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"results/exp2_allmini_baseline_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResultados guardados en: {filename}")
        
        # También guardar resumen en CSV
        csv_filename = filename.replace('.json', '.csv')
        self._save_csv_summary(results, csv_filename)
        
    def _save_csv_summary(self, results: Dict[str, Any], csv_filename: str):
        """Guarda un resumen de los resultados en CSV."""
        import csv
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Strategy', 'Top1_Accuracy', 'Top5_Accuracy', 'MRR', 
                'Top1_Correct', 'Top5_Correct', 'Total_Queries'
            ])
            
            # Datos por estrategia
            for strategy_name, strategy_data in results['results_by_strategy'].items():
                writer.writerow([
                    strategy_name,
                    f"{strategy_data['top1_accuracy']:.3f}",
                    f"{strategy_data['top5_accuracy']:.3f}",
                    f"{strategy_data['mrr']:.4f}",
                    strategy_data['top1_correct'],
                    strategy_data['top5_correct'],
                    strategy_data['total_queries']
                ])
        
        print(f"Resumen CSV guardado en: {csv_filename}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Imprime un resumen de los resultados."""
        print("\n" + "="*80)
        print("RESUMEN DEL EXPERIMENTO 2: EVALUACIÓN INICIAL CON ALL-MINI BASE")
        print("="*80)
        
        print(f"\n📊 INFORMACIÓN GENERAL:")
        print(f"   Modelo: {results['model']}")
        print(f"   Total de chunks: {results['total_chunks']}")
        print(f"   Estrategias evaluadas: {len(results['strategies_evaluated'])}")
        
        print(f"\n📈 RESULTADOS POR ESTRATEGIA:")
        print(f"{'Estrategia':<25} {'Top1 Acc':<10} {'Top5 Acc':<10} {'MRR':<8} {'Top1':<6} {'Top5':<6}")
        print("-" * 75)
        
        for strategy_name, strategy_data in results['results_by_strategy'].items():
            top1_acc = f"{strategy_data['top1_accuracy']:.3f}"
            top5_acc = f"{strategy_data['top5_accuracy']:.3f}"
            mrr = f"{strategy_data['mrr']:.4f}"
            top1_correct = strategy_data['top1_correct']
            top5_correct = strategy_data['top5_correct']
            
            print(f"{strategy_name:<25} {top1_acc:<10} {top5_acc:<10} {mrr:<8} {top1_correct:<6} {top5_correct:<6}")
        
        # Mostrar la mejor estrategia
        best_strategy = max(results['results_by_strategy'].items(), 
                           key=lambda x: x[1]['top1_accuracy'])
        
        print(f"\n🏆 MEJOR ESTRATEGIA: {best_strategy[0].upper()}")
        print(f"   Top1 Accuracy: {best_strategy[1]['top1_accuracy']:.3f}")
        print(f"   Top5 Accuracy: {best_strategy[1]['top5_accuracy']:.3f}")
        print(f"   MRR: {best_strategy[1]['mrr']:.4f}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Experimento 2: Evaluación Inicial con All-Mini Base")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados top-k a evaluar")
    parser.add_argument("--output", type=str, help="Nombre del archivo de salida")
    
    args = parser.parse_args()
    
    # Ejecutar experimento
    experiment = AllMiniBaselineExperiment()
    results = experiment.run_experiment(args.top_k)
    
    if results:
        # Guardar resultados
        experiment.save_results(results, args.output)
        
        # Mostrar resumen
        experiment.print_summary(results)
        
        print("\n¡Experimento completado exitosamente!")
    else:
        print("Error: No se pudieron obtener resultados del experimento")


if __name__ == "__main__":
    main()
