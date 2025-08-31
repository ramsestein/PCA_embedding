#!/usr/bin/env python3
"""
Experimento: SAPBERT + Markov + Detección de Palabras Clave con Regex
Combina embeddings, MRF y detección léxica para mejorar el rendimiento.
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rerank_markov.types import Chunk, Query, ScoredChunk
from rerank_markov.utils import tokenize
from rerank_markov.index_stats import compute_corpus_statistics
from rerank_markov.qlm import qlm_score
from rerank_markov.mrf import mrf_sd_score
from rerank_markov.graph import build_chunk_graph, personalized_pagerank
from rerank_markov.fusion import normalize_scores


class KeywordRegexExperiment:
    """Experimento que combina SAPBERT + Markov + Detección de palabras clave con regex."""
    
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
    
    def load_sapbert_model(self):
        """Carga el modelo SAPBERT-UMLS."""
        print("Cargando modelo SAPBERT-UMLS...")
        model_path = "sapbert-umls/model-0_0029"
        if not os.path.exists(model_path):
            raise ValueError(f"Modelo SAPBERT no encontrado en: {model_path}")
        
        self.model = SentenceTransformer(model_path)
        print("Modelo SAPBERT cargado exitosamente!")
        
    def compute_embeddings(self):
        """Calcula embeddings para todos los chunks."""
        print("Calculando embeddings con SAPBERT...")
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
        
    def extract_keywords_with_regex(self, query: str, chunk: Chunk) -> Dict[str, Any]:
        """
        Extrae palabras clave de la query y las busca en el chunk usando regex.
        """
        # Limpiar y tokenizar la query
        query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
        query_words = [word.strip() for word in query_clean.split() if len(word.strip()) > 2]
        
        # Buscar cada palabra clave en el chunk
        chunk_text_lower = chunk.text.lower()
        keyword_matches = {}
        
        for word in query_words:
            # Crear patrón regex para buscar la palabra completa
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = re.findall(pattern, chunk_text_lower)
            
            if matches:
                keyword_matches[word] = {
                    'count': len(matches),
                    'frequency': len(matches) / len(chunk_text_lower.split()),
                    'positions': [m.start() for m in re.finditer(pattern, chunk_text_lower)]
                }
        
        # Calcular puntuación de palabras clave
        total_matches = sum(match['count'] for match in keyword_matches.values())
        unique_keywords = len(keyword_matches)
        avg_frequency = np.mean([match['frequency'] for match in keyword_matches.values()]) if keyword_matches else 0
        
        # Puntuación compuesta
        keyword_score = (total_matches * 0.4 + unique_keywords * 0.4 + avg_frequency * 100 * 0.2)
        
        return {
            'keywords_found': keyword_matches,
            'total_matches': total_matches,
            'unique_keywords': unique_keywords,
            'avg_frequency': avg_frequency,
            'keyword_score': keyword_score,
            'coverage_ratio': unique_keywords / len(query_words) if query_words else 0
        }
    
    def calculate_hybrid_score(self, query: str, chunk: Chunk, 
                             weights: Dict[str, float] = None) -> float:
        """
        Calcula puntuación híbrida combinando SAPBERT + Markov + Palabras clave.
        """
        if weights is None:
            weights = {
                'embedding': 0.4,      # Similitud de embeddings
                'mrf': 0.3,            # Markov Random Field
                'qlm': 0.2,            # Query-Likelihood Model
                'keywords': 0.1        # Detección de palabras clave
            }
        
        # 1. Puntuación de embeddings (SAPBERT)
        query_embedding = self.model.encode([query])[0]
        if chunk.embedding is not None:
            embedding_similarity = self._cosine_similarity(query_embedding, chunk.embedding)
        else:
            embedding_similarity = 0.0
        
        # 2. Puntuación MRF
        mrf_score = mrf_sd_score(query, chunk, w_unigram=0.7, w_ordered=0.2, w_unordered=0.1)
        
        # 3. Puntuación QLM
        qlm_score_val = qlm_score(query, chunk, self.corpus_stats, mu=1500)
        
        # 4. Puntuación de palabras clave
        keyword_analysis = self.extract_keywords_with_regex(query, chunk)
        keyword_score = keyword_analysis['keyword_score']
        
        # Normalizar puntuaciones individualmente
        scores_list = [embedding_similarity, mrf_score, qlm_score_val, keyword_score]
        normalized_scores_list = normalize_scores(scores_list)
        
        # Crear diccionario de puntuaciones normalizadas
        normalized_scores = {
            'embedding': normalized_scores_list[0],
            'mrf': normalized_scores_list[1],
            'qlm': normalized_scores_list[2],
            'keywords': normalized_scores_list[3]
        }
        
        # Combinar con pesos
        final_score = sum(weights[component] * normalized_scores[component] 
                         for component in weights.keys())
        
        return final_score
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def evaluate_strategy(self, query: str, expected_doc: str, top_k: int = 5) -> Dict[str, Any]:
        """Evalúa la estrategia híbrida para una query específica."""
        start_time = time.time()
        
        # Normalizar expected_doc para comparación
        expected_doc_normalized = expected_doc.replace('_limpio.txt', '_limpio')
        
        # Calcular puntuaciones híbridas para todos los chunks
        scored_chunks = []
        for chunk in self.chunks:
            score = self.calculate_hybrid_score(query, chunk)
            scored_chunks.append({
                'chunk': chunk,
                'score': score,
                'doc_id': chunk.doc_id
            })
        
        # Ordenar por puntuación
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # Verificar Top1 y Top5
        top1_chunk = scored_chunks[0]
        top1_correct = top1_chunk['doc_id'] == expected_doc_normalized
        
        top5_chunks = scored_chunks[:top_k]
        top5_correct = any(chunk['doc_id'] == expected_doc_normalized for chunk in top5_chunks)
        
        # Encontrar ranking del documento esperado
        expected_rank = -1
        for i, chunk in enumerate(scored_chunks):
            if chunk['doc_id'] == expected_doc_normalized:
                expected_rank = i + 1
                break
        
        search_time = time.time() - start_time
        
        return {
            'query': query,
            'expected_doc': expected_doc,
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'search_time': search_time,
            'expected_rank': expected_rank,
            'top_results': [
                {
                    'rank': i + 1,
                    'doc_id': chunk['doc_id'],
                    'score': chunk['score'],
                    'text_preview': chunk['chunk'].text[:100] + "..."
                }
                for i, chunk in enumerate(top5_chunks)
            ]
        }
    
    def run_experiment(self, top_k: int = 5) -> Dict[str, Any]:
        """Ejecuta el experimento completo."""
        print("=== INICIANDO EXPERIMENTO: SAPBERT + MARKOV + PALABRAS CLAVE ===")
        
        # Cargar documentos y queries
        self.load_pnts_documents()
        self.benchmark_queries = self.load_benchmark_queries()
        
        if not self.benchmark_queries:
            print("No se pudieron cargar las queries de benchmark")
            return {}
        
        # Cargar modelo y calcular embeddings
        self.load_sapbert_model()
        self.compute_embeddings()
        self.compute_corpus_statistics()
        
        # Evaluar todas las queries
        results = []
        correct_top1 = 0
        correct_top5 = 0
        
        for i, query_data in enumerate(self.benchmark_queries):
            print(f"\nEvaluando query {i+1}/{len(self.benchmark_queries)}: {query_data['query'][:50]}...")
            
            result = self.evaluate_strategy(
                query_data['query'], 
                query_data['expected_doc'], 
                top_k
            )
            
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
        
        experiment_results = {
            'experiment_name': 'SAPBERT + Markov + Palabras Clave con Regex',
            'strategy': 'Hibridación con detección léxica',
            'total_queries': total_queries,
            'top1_correct': correct_top1,
            'top5_correct': correct_top5,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'mrr': mrr,
            'query_results': results,
            'weights_used': {
                'embedding': 0.4,
                'mrf': 0.3,
                'qlm': 0.2,
                'keywords': 0.1
            },
            'timestamp': time.time()
        }
        
        return experiment_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Guarda los resultados del experimento."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"results/experiment_keyword_regex_{timestamp}.json"
        
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
                'Query', 'Expected_Doc', 'Top1_Correct', 'Top5_Correct', 
                'Expected_Rank', 'Search_Time', 'Top1_Doc', 'Top1_Score'
            ])
            
            # Datos
            for result in results['query_results']:
                writer.writerow([
                    result['query'][:100],
                    result['expected_doc'],
                    result['top1_correct'],
                    result['top5_correct'],
                    result['expected_rank'],
                    f"{result['search_time']:.4f}",
                    result['top_results'][0]['doc_id'] if result['top_results'] else 'N/A',
                    f"{result['top_results'][0]['score']:.4f}" if result['top_results'] else 'N/A'
                ])
        
        print(f"Resumen CSV guardado en: {csv_filename}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Imprime un resumen de los resultados."""
        print("\n" + "="*80)
        print("RESUMEN DEL EXPERIMENTO: SAPBERT + MARKOV + PALABRAS CLAVE")
        print("="*80)
        
        print(f"\n📊 MÉTRICAS GENERALES:")
        print(f"   Total de queries: {results['total_queries']}")
        print(f"   Top1 correctas: {results['top1_correct']}/{results['total_queries']} ({results['top1_accuracy']*100:.1f}%)")
        print(f"   Top5 correctas: {results['top5_correct']}/{results['total_queries']} ({results['top5_accuracy']*100:.1f}%)")
        print(f"   MRR: {results['mrr']:.4f}")
        
        print(f"\n⚖️ PESOS UTILIZADOS:")
        for component, weight in results['weights_used'].items():
            print(f"   {component.capitalize()}: {weight}")
        
        print(f"\n🔍 ANÁLISIS DE RENDIMIENTO:")
        # Análisis de queries correctas vs incorrectas
        correct_queries = [r for r in results['query_results'] if r['top1_correct']]
        incorrect_queries = [r for r in results['query_results'] if not r['top1_correct']]
        
        print(f"   Queries Top1 correctas: {len(correct_queries)}")
        print(f"   Queries Top1 incorrectas: {len(incorrect_queries)}")
        
        if incorrect_queries:
            print(f"\n❌ EJEMPLOS DE QUERIES INCORRECTAS:")
            for i, result in enumerate(incorrect_queries[:3]):
                print(f"   {i+1}. {result['query'][:60]}...")
                print(f"      Esperado: {result['expected_doc']}")
                print(f"      Obtenido: {result['top_results'][0]['doc_id']}")
                print(f"      Ranking esperado: {result['expected_rank']}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Experimento SAPBERT + Markov + Palabras Clave")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados top-k a evaluar")
    parser.add_argument("--output", type=str, help="Nombre del archivo de salida")
    
    args = parser.parse_args()
    
    # Ejecutar experimento
    experiment = KeywordRegexExperiment()
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
