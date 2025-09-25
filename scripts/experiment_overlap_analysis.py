#!/usr/bin/env python3
"""
Experimento 12 y 13: Análisis de Solapamiento entre Modelos
Calcula el solapamiento entre SAPBERT y All-Mini Base.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
import numpy as np
from sentence_transformers import SentenceTransformer

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rerank_markov.types import Chunk
from rerank_markov.utils import tokenize


class OverlapAnalysisExperiment:
    """Experimento que analiza el solapamiento entre modelos."""
    
    def __init__(self):
        self.chunks = []
        self.benchmark_queries = []
        self.sapbert_model = None
        self.allmini_model = None
        
    def load_pnts_documents(self, pnts_dir: str = "PNTs") -> None:
        """Carga documentos PNTs y crea chunks."""
        print(f"Cargando documentos desde: {pnts_dir}")
        
        chunks = []
        pnts_path = Path(pnts_dir)
        
        for txt_file in pnts_path.glob("*_limpio.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                continue
                
            doc_chunks = self._create_chunks_from_text(
                content, str(txt_file.stem), chunk_size=512, overlap=50
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
            
            if len(chunk_text.strip()) < 50:
                continue
                
            chunk = Chunk(
                id=f"chunk_{i:03d}",
                text=chunk_text,
                doc_id=doc_id,
                position=i,
                embedding=None,
                meta={"source": doc_id}
            )
            chunks.append(chunk)
            
        return chunks
    
    def load_benchmark_queries(self) -> List[Dict[str, str]]:
        """Carga las queries del archivo de benchmark."""
        benchmark_file = "benchmark/preguntas_con_docs_es.json"
        
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
    
    def load_models(self):
        """Carga ambos modelos."""
        print("Cargando modelos...")
        
        # SAPBERT
        sapbert_path = "sapbert-umls-100/sapbert-umls/model-0_0003"
        self.sapbert_model = SentenceTransformer(sapbert_path)
        print("Modelo SAPBERT cargado!")
        
        # All-Mini
        self.allmini_model = SentenceTransformer("all-mini-base")
        print("Modelo All-Mini cargado!")
        
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def evaluate_model_top_results(self, model: SentenceTransformer, query: str, expected_doc: str, top_k: int = 5) -> Dict[str, Any]:
        """Evalúa un modelo y retorna resultados top-k."""
        expected_doc_normalized = expected_doc.replace('_limpio.txt', '_limpio')
        
        # Calcular embeddings
        texts = [chunk.text for chunk in self.chunks]
        chunk_embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
        query_embedding = model.encode([query])[0]
        
        # Calcular similitudes
        similarities = []
        for i, chunk in enumerate(self.chunks):
            similarity = self._cosine_similarity(query_embedding, chunk_embeddings[i])
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
        
        return {
            'query': query,
            'expected_doc': expected_doc,
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'expected_rank': expected_rank,
            'top1_doc': top1_chunk.doc_id,
            'top5_docs': [chunk.doc_id for chunk in top5_chunks]
        }
    
    def analyze_overlap(self, top_k: int = 5) -> Dict[str, Any]:
        """Analiza el solapamiento entre modelos."""
        print("=== INICIANDO ANÁLISIS DE SOLAPAMIENTO ===")
        
        # Cargar datos
        self.load_pnts_documents()
        self.benchmark_queries = self.load_benchmark_queries()
        self.load_models()
        
        sapbert_results = []
        allmini_results = []
        
        print("\nEvaluando SAPBERT...")
        for i, query_data in enumerate(self.benchmark_queries):
            print(f"Query {i+1}/{len(self.benchmark_queries)}: {query_data['query'][:50]}...")
            result = self.evaluate_model_top_results(
                self.sapbert_model, query_data['query'], query_data['expected_doc'], top_k
            )
            sapbert_results.append(result)
        
        print("\nEvaluando All-Mini...")
        for i, query_data in enumerate(self.benchmark_queries):
            print(f"Query {i+1}/{len(self.benchmark_queries)}: {query_data['query'][:50]}...")
            result = self.evaluate_model_top_results(
                self.allmini_model, query_data['query'], query_data['expected_doc'], top_k
            )
            allmini_results.append(result)
        
        # Análizar solapamiento
        overlap_analysis = self._calculate_overlap(sapbert_results, allmini_results)
        
        return {
            'experiment_name': 'Análisis de Solapamiento entre Modelos',
            'models_compared': ['SAPBERT model-0_0003', 'All-Mini Base'],
            'total_queries': len(self.benchmark_queries),
            'total_chunks': len(self.chunks),
            'sapbert_results': sapbert_results,
            'allmini_results': allmini_results,
            'overlap_analysis': overlap_analysis,
            'timestamp': time.time()
        }
    
    def _calculate_overlap(self, sapbert_results: List[Dict], allmini_results: List[Dict]) -> Dict[str, Any]:
        """Calcula métricas de solapamiento."""
        
        # Top1 Analysis
        sapbert_top1_correct = set()
        allmini_top1_correct = set()
        
        for i, (sap_result, mini_result) in enumerate(zip(sapbert_results, allmini_results)):
            if sap_result['top1_correct']:
                sapbert_top1_correct.add(i)
            if mini_result['top1_correct']:
                allmini_top1_correct.add(i)
        
        # Top5 Analysis
        sapbert_top5_correct = set()
        allmini_top5_correct = set()
        
        for i, (sap_result, mini_result) in enumerate(zip(sapbert_results, allmini_results)):
            if sap_result['top5_correct']:
                sapbert_top5_correct.add(i)
            if mini_result['top5_correct']:
                allmini_top5_correct.add(i)
        
        # Calcular solapamientos
        top1_overlap = len(sapbert_top1_correct & allmini_top1_correct)
        top1_only_sapbert = len(sapbert_top1_correct - allmini_top1_correct)
        top1_only_allmini = len(allmini_top1_correct - sapbert_top1_correct)
        top1_neither = len(sapbert_results) - len(sapbert_top1_correct | allmini_top1_correct)
        
        top5_overlap = len(sapbert_top5_correct & allmini_top5_correct)
        top5_only_sapbert = len(sapbert_top5_correct - allmini_top5_correct)
        top5_only_allmini = len(allmini_top5_correct - sapbert_top5_correct)
        top5_neither = len(sapbert_results) - len(sapbert_top5_correct | allmini_top5_correct)
        
        total_queries = len(sapbert_results)
        
        return {
            'top1_analysis': {
                'overlap': {'count': top1_overlap, 'percentage': top1_overlap / total_queries * 100},
                'only_sapbert': {'count': top1_only_sapbert, 'percentage': top1_only_sapbert / total_queries * 100},
                'only_allmini': {'count': top1_only_allmini, 'percentage': top1_only_allmini / total_queries * 100},
                'neither': {'count': top1_neither, 'percentage': top1_neither / total_queries * 100}
            },
            'top5_analysis': {
                'overlap': {'count': top5_overlap, 'percentage': top5_overlap / total_queries * 100},
                'only_sapbert': {'count': top5_only_sapbert, 'percentage': top5_only_sapbert / total_queries * 100},
                'only_allmini': {'count': top5_only_allmini, 'percentage': top5_only_allmini / total_queries * 100},
                'neither': {'count': top5_neither, 'percentage': top5_neither / total_queries * 100}
            },
            'sapbert_performance': {
                'top1_accuracy': len(sapbert_top1_correct) / total_queries,
                'top5_accuracy': len(sapbert_top5_correct) / total_queries,
                'top1_correct': len(sapbert_top1_correct),
                'top5_correct': len(sapbert_top5_correct)
            },
            'allmini_performance': {
                'top1_accuracy': len(allmini_top1_correct) / total_queries,
                'top5_accuracy': len(allmini_top5_correct) / total_queries,
                'top1_correct': len(allmini_top1_correct),
                'top5_correct': len(allmini_top5_correct)
            }
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Guarda los resultados del experimento."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"results/exp12_13_overlap_analysis_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResultados guardados en: {filename}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Imprime un resumen de los resultados."""
        print("\n" + "="*80)
        print("RESUMEN DEL ANÁLISIS DE SOLAPAMIENTO")
        print("="*80)
        
        overlap = results['overlap_analysis']
        
        print(f"\n📊 INFORMACIÓN GENERAL:")
        print(f"   Total de queries: {results['total_queries']}")
        print(f"   Total de chunks: {results['total_chunks']}")
        print(f"   Modelos comparados: {', '.join(results['models_compared'])}")
        
        print(f"\n📈 RENDIMIENTO INDIVIDUAL:")
        sap_perf = overlap['sapbert_performance']
        mini_perf = overlap['allmini_performance']
        
        print(f"   SAPBERT:")
        print(f"     Top1: {sap_perf['top1_correct']}/{results['total_queries']} ({sap_perf['top1_accuracy']*100:.1f}%)")
        print(f"     Top5: {sap_perf['top5_correct']}/{results['total_queries']} ({sap_perf['top5_accuracy']*100:.1f}%)")
        
        print(f"   All-Mini:")
        print(f"     Top1: {mini_perf['top1_correct']}/{results['total_queries']} ({mini_perf['top1_accuracy']*100:.1f}%)")
        print(f"     Top5: {mini_perf['top5_correct']}/{results['total_queries']} ({mini_perf['top5_accuracy']*100:.1f}%)")
        
        print(f"\n🔍 ANÁLISIS DE SOLAPAMIENTO TOP1:")
        top1 = overlap['top1_analysis']
        print(f"   Solapamiento: {top1['overlap']['count']}/{results['total_queries']} ({top1['overlap']['percentage']:.1f}%)")
        print(f"   Solo SAPBERT: {top1['only_sapbert']['count']}/{results['total_queries']} ({top1['only_sapbert']['percentage']:.1f}%)")
        print(f"   Solo All-Mini: {top1['only_allmini']['count']}/{results['total_queries']} ({top1['only_allmini']['percentage']:.1f}%)")
        print(f"   Ninguno: {top1['neither']['count']}/{results['total_queries']} ({top1['neither']['percentage']:.1f}%)")
        
        print(f"\n🔍 ANÁLISIS DE SOLAPAMIENTO TOP5:")
        top5 = overlap['top5_analysis']
        print(f"   Solapamiento: {top5['overlap']['count']}/{results['total_queries']} ({top5['overlap']['percentage']:.1f}%)")
        print(f"   Solo SAPBERT: {top5['only_sapbert']['count']}/{results['total_queries']} ({top5['only_sapbert']['percentage']:.1f}%)")
        print(f"   Solo All-Mini: {top5['only_allmini']['count']}/{results['total_queries']} ({top5['only_allmini']['percentage']:.1f}%)")
        print(f"   Ninguno: {top5['neither']['count']}/{results['total_queries']} ({top5['neither']['percentage']:.1f}%)")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Experimento 12-13: Análisis de Solapamiento")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados top-k a evaluar")
    parser.add_argument("--output", type=str, help="Nombre del archivo de salida")
    
    args = parser.parse_args()
    
    # Ejecutar experimento
    experiment = OverlapAnalysisExperiment()
    results = experiment.analyze_overlap(args.top_k)
    
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
