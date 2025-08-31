#!/usr/bin/env python3
"""
Experimento: SOLO Detección de Palabras Clave con Regex
Evalúa únicamente la componente léxica sin embeddings, MRF o QLM.
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

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rerank_markov.types import Chunk, Query, ScoredChunk
from rerank_markov.utils import tokenize


class RegexOnlyExperiment:
    """Experimento que evalúa SOLO la detección de palabras clave con regex."""
    
    def __init__(self):
        self.chunks = []
        self.benchmark_queries = []
        
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
                embedding=None,  # No necesitamos embeddings para este experimento
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
        
        # Puntuación compuesta (misma que en el experimento híbrido)
        keyword_score = (total_matches * 0.4 + unique_keywords * 0.4 + avg_frequency * 100 * 0.2)
        
        return {
            'keywords_found': keyword_matches,
            'total_matches': total_matches,
            'unique_keywords': unique_keywords,
            'avg_frequency': avg_frequency,
            'keyword_score': keyword_score,
            'coverage_ratio': unique_keywords / len(query_words) if query_words else 0
        }
    
    def calculate_regex_only_score(self, query: str, chunk: Chunk) -> float:
        """
        Calcula puntuación usando SOLO detección de palabras clave con regex.
        """
        keyword_analysis = self.extract_keywords_with_regex(query, chunk)
        return keyword_analysis['keyword_score']
    
    def evaluate_strategy(self, query: str, expected_doc: str, top_k: int = 5) -> Dict[str, Any]:
        """Evalúa la estrategia de solo regex para una query específica."""
        start_time = time.time()
        
        # Normalizar expected_doc para comparación
        expected_doc_normalized = expected_doc.replace('_limpio.txt', '_limpio')
        
        # Calcular puntuaciones de solo regex para todos los chunks
        scored_chunks = []
        for chunk in self.chunks:
            score = self.calculate_regex_only_score(query, chunk)
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
                    'text_preview': chunk['chunk'].text[:100] + "...",
                    'keyword_analysis': self.extract_keywords_with_regex(query, chunk['chunk'])
                }
                for i, chunk in enumerate(top5_chunks)
            ]
        }
    
    def run_experiment(self, top_k: int = 5) -> Dict[str, Any]:
        """Ejecuta el experimento completo."""
        print("=== INICIANDO EXPERIMENTO: SOLO DETECCIÓN DE PALABRAS CLAVE CON REGEX ===")
        
        # Cargar documentos y queries
        self.load_pnts_documents()
        self.benchmark_queries = self.load_benchmark_queries()
        
        if not self.benchmark_queries:
            print("No se pudieron cargar las queries de benchmark")
            return {}
        
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
            'experiment_name': 'SOLO Detección de Palabras Clave con Regex',
            'strategy': 'Componente léxica aislada',
            'total_queries': total_queries,
            'top1_correct': correct_top1,
            'top5_correct': correct_top5,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'mrr': mrr,
            'query_results': results,
            'methodology': {
                'approach': 'Regex-only keyword detection',
                'pattern': r'\bpalabra\b',
                'scoring': 'total_matches * 0.4 + unique_keywords * 0.4 + avg_frequency * 100 * 0.2',
                'no_embeddings': True,
                'no_mrf': True,
                'no_qlm': True
            },
            'timestamp': time.time()
        }
        
        return experiment_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Guarda los resultados del experimento."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"results/experiment_regex_only_{timestamp}.json"
        
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
                'Expected_Rank', 'Search_Time', 'Top1_Doc', 'Top1_Score',
                'Keywords_Found', 'Total_Matches', 'Unique_Keywords'
            ])
            
            # Datos
            for result in results['query_results']:
                top1_result = result['top_results'][0] if result['top_results'] else {}
                keyword_analysis = top1_result.get('keyword_analysis', {})
                
                writer.writerow([
                    result['query'][:100],
                    result['expected_doc'],
                    result['top1_correct'],
                    result['top5_correct'],
                    result['expected_rank'],
                    f"{result['search_time']:.4f}",
                    top1_result.get('doc_id', 'N/A'),
                    f"{top1_result.get('score', 0):.4f}",
                    keyword_analysis.get('unique_keywords', 0),
                    keyword_analysis.get('total_matches', 0),
                    keyword_analysis.get('unique_keywords', 0)
                ])
        
        print(f"Resumen CSV guardado en: {csv_filename}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Imprime un resumen de los resultados."""
        print("\n" + "="*80)
        print("RESUMEN DEL EXPERIMENTO: SOLO DETECCIÓN DE PALABRAS CLAVE CON REGEX")
        print("="*80)
        
        print(f"\n📊 MÉTRICAS GENERALES:")
        print(f"   Total de queries: {results['total_queries']}")
        print(f"   Top1 correctas: {results['top1_correct']}/{results['total_queries']} ({results['top1_accuracy']*100:.1f}%)")
        print(f"   Top5 correctas: {results['top5_correct']}/{results['total_queries']} ({results['top5_accuracy']*100:.1f}%)")
        print(f"   MRR: {results['mrr']:.4f}")
        
        print(f"\n🔍 METODOLOGÍA:")
        print(f"   Enfoque: {results['methodology']['approach']}")
        print(f"   Patrón regex: {results['methodology']['pattern']}")
        print(f"   Sin embeddings: {results['methodology']['no_embeddings']}")
        print(f"   Sin MRF: {results['methodology']['no_mrf']}")
        print(f"   Sin QLM: {results['methodology']['no_qlm']}")
        
        print(f"\n🔍 ANÁLISIS DE RENDIMIENTO:")
        # Análisis de queries correctas vs incorrectas
        correct_queries = [r for r in results['query_results'] if r['top1_correct']]
        incorrect_queries = [r for r in results['query_results'] if not r['top1_correct']]
        
        print(f"   Queries Top1 correctas: {len(correct_queries)}")
        print(f"   Queries Top1 incorrectas: {len(incorrect_queries)}")
        
        if correct_queries:
            print(f"\n✅ EJEMPLOS DE QUERIES CORRECTAS:")
            for i, result in enumerate(correct_queries[:3]):
                print(f"   {i+1}. {result['query'][:60]}...")
                print(f"      Documento: {result['expected_doc']}")
                print(f"      Ranking: {result['expected_rank']}")
                top1_analysis = result['top_results'][0]['keyword_analysis']
                print(f"      Keywords encontradas: {top1_analysis['unique_keywords']}")
                print(f"      Matches totales: {top1_analysis['total_matches']}")
        
        if incorrect_queries:
            print(f"\n❌ EJEMPLOS DE QUERIES INCORRECTAS:")
            for i, result in enumerate(incorrect_queries[:3]):
                print(f"   {i+1}. {result['query'][:60]}...")
                print(f"      Esperado: {result['expected_doc']}")
                print(f"      Obtenido: {result['top_results'][0]['doc_id']}")
                print(f"      Ranking esperado: {result['expected_rank']}")
                top1_analysis = result['top_results'][0]['keyword_analysis']
                print(f"      Keywords encontradas: {top1_analysis['unique_keywords']}")
                print(f"      Matches totales: {top1_analysis['total_matches']}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Experimento SOLO Detección de Palabras Clave con Regex")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados top-k a evaluar")
    parser.add_argument("--output", type=str, help="Nombre del archivo de salida")
    
    args = parser.parse_args()
    
    # Ejecutar experimento
    experiment = RegexOnlyExperiment()
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
