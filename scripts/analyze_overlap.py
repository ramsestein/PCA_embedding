#!/usr/bin/env python3
"""
Script para analizar el solapamiento de clasificación correcta entre SAPBERT y All-Mini.
Calcula qué queries cada modelo resuelve correctamente y cuántas son comunes vs únicas.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rerank_markov.types import Chunk
from rerank_markov.utils import tokenize


class OverlapAnalyzer:
    """Clase para analizar el solapamiento entre modelos de embeddings."""
    
    def __init__(self):
        self.chunks = []
        self.benchmark_queries = []
        self.sapbert_results = {}
        self.allmini_results = {}
        
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
    
    def test_sapbert_model(self, queries: List[Dict[str, str]], top_k: int = 5) -> Dict[str, Any]:
        """Prueba el modelo SAPBERT y retorna resultados detallados."""
        print("\n=== Probando modelo SAPBERT-UMLS ===")
        
        model_path = "sapbert-umls/model-0_0029"
        if not os.path.exists(model_path):
            raise ValueError(f"Modelo SAPBERT no encontrado en: {model_path}")
        
        # Cargar modelo
        start_time = time.time()
        model = SentenceTransformer(model_path)
        load_time = time.time() - start_time
        print(f"Modelo SAPBERT cargado en {load_time:.2f} segundos")
        
        # Calcular embeddings
        print("Calculando embeddings con SAPBERT...")
        start_time = time.time()
        texts = [chunk.text for chunk in self.chunks]
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Asignar embeddings
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]
            
        embedding_time = time.time() - start_time
        print(f"Embeddings calculados en {embedding_time:.2f} segundos")
        
        # Evaluar modelo
        results = self._evaluate_model_detailed(model, queries, top_k, "SAPBERT")
        results['load_time'] = load_time
        results['embedding_time'] = embedding_time
        
        return results
    
    def test_allmini_model(self, queries: List[Dict[str, str]], top_k: int = 5) -> Dict[str, Any]:
        """Prueba el modelo All-Mini y retorna resultados detallados."""
        print("\n=== Probando modelo All-Mini ===")
        
        # Usar modelo por defecto
        start_time = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        load_time = time.time() - start_time
        print(f"Modelo All-Mini cargado en {load_time:.2f} segundos")
        
        # Calcular embeddings
        print("Calculando embeddings con All-Mini...")
        start_time = time.time()
        texts = [chunk.text for chunk in self.chunks]
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Asignar embeddings
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]
            
        embedding_time = time.time() - start_time
        print(f"Embeddings calculados en {embedding_time:.2f} segundos")
        
        # Evaluar modelo
        results = self._evaluate_model_detailed(model, queries, top_k, "All-Mini")
        results['load_time'] = load_time
        results['embedding_time'] = embedding_time
        
        return results
    
    def _evaluate_model_detailed(self, model: SentenceTransformer, queries: List[Dict[str, str]], top_k: int, model_name: str) -> Dict[str, Any]:
        """Evalúa un modelo y retorna resultados detallados incluyendo queries correctas."""
        
        correct_top1_queries = []
        correct_top5_queries = []
        incorrect_queries = []
        
        for i, query_data in enumerate(queries):
            query = query_data['query']
            expected_doc = query_data['expected_doc']
            expected_doc_normalized = expected_doc.replace('_limpio.txt', '_limpio')
            
            # Calcular embedding de la query
            query_embedding = model.encode([query])[0]
            
            # Calcular similitudes
            similarities = []
            for chunk in self.chunks:
                if chunk.embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                else:
                    similarity = 0.0
                similarities.append((similarity, chunk))
            
            # Ordenar por similitud
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Verificar Top1
            top1_chunk = similarities[0][1]
            top1_correct = top1_chunk.doc_id == expected_doc_normalized
            
            # Verificar Top5
            top5_chunks = [chunk for _, chunk in similarities[:top_k]]
            top5_correct = any(chunk.doc_id == expected_doc_normalized for chunk in top5_chunks)
            
            # Guardar resultados detallados
            query_result = {
                'query': query,
                'expected_doc': expected_doc,
                'top1_chunk': top1_chunk.doc_id,
                'top1_correct': top1_correct,
                'top5_correct': top5_correct,
                'top1_similarity': similarities[0][0],
                'rank_of_expected': self._find_rank_of_expected(similarities, expected_doc_normalized)
            }
            
            if top1_correct:
                correct_top1_queries.append(query_result)
            elif top5_correct:
                correct_top5_queries.append(query_result)
            else:
                incorrect_queries.append(query_result)
        
        total_queries = len(queries)
        top1_accuracy = len(correct_top1_queries) / total_queries
        top5_accuracy = (len(correct_top1_queries) + len(correct_top5_queries)) / total_queries
        
        return {
            'model_name': model_name,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'top1_correct': len(correct_top1_queries),
            'top5_correct': len(correct_top5_queries),
            'total_queries': total_queries,
            'correct_top1_queries': correct_top1_queries,
            'correct_top5_queries': correct_top5_queries,
            'incorrect_queries': incorrect_queries
        }
    
    def _find_rank_of_expected(self, similarities: List[Tuple[float, Chunk]], expected_doc: str) -> int:
        """Encuentra el ranking del documento esperado."""
        for rank, (_, chunk) in enumerate(similarities):
            if chunk.doc_id == expected_doc:
                return rank + 1
        return -1  # No encontrado
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def analyze_overlap(self) -> Dict[str, Any]:
        """Analiza el solapamiento entre los dos modelos."""
        print("\n=== ANALIZANDO SOLAPAMIENTO ENTRE MODELOS ===")
        
        # Obtener queries correctas de cada modelo
        sapbert_top1_correct = set(q['query'] for q in self.sapbert_results['correct_top1_queries'])
        sapbert_top5_correct = set(q['query'] for q in self.sapbert_results['correct_top5_queries'])
        
        allmini_top1_correct = set(q['query'] for q in self.allmini_results['correct_top1_queries'])
        allmini_top5_correct = set(q['query'] for q in self.allmini_results['correct_top5_queries'])
        
        # Calcular solapamientos
        overlap_top1 = sapbert_top1_correct.intersection(allmini_top1_correct)
        overlap_top5 = sapbert_top5_correct.intersection(allmini_top5_correct)
        
        # Calcular diferencias
        sapbert_only_top1 = sapbert_top1_correct - allmini_top1_correct
        allmini_only_top1 = allmini_top1_correct - sapbert_top1_correct
        
        sapbert_only_top5 = sapbert_top5_correct - allmini_top5_correct
        allmini_only_top5 = allmini_top5_correct - sapbert_top5_correct
        
        # Calcular métricas
        total_queries = len(self.benchmark_queries)
        
        overlap_analysis = {
            'total_queries': total_queries,
            
            # Top1 Analysis
            'top1': {
                'sapbert_correct': len(sapbert_top1_correct),
                'allmini_correct': len(allmini_top1_correct),
                'overlap': len(overlap_top1),
                'sapbert_only': len(sapbert_only_top1),
                'allmini_only': len(allmini_only_top1),
                'overlap_percentage': len(overlap_top1) / total_queries * 100,
                'sapbert_unique_percentage': len(sapbert_only_top1) / total_queries * 100,
                'allmini_unique_percentage': len(allmini_only_top1) / total_queries * 100
            },
            
            # Top5 Analysis
            'top5': {
                'sapbert_correct': len(sapbert_top5_correct),
                'allmini_correct': len(allmini_top5_correct),
                'overlap': len(overlap_top5),
                'sapbert_only': len(sapbert_only_top5),
                'allmini_only': len(allmini_only_top5),
                'overlap_percentage': len(overlap_top5) / total_queries * 100,
                'sapbert_unique_percentage': len(sapbert_only_top5) / total_queries * 100,
                'allmini_unique_percentage': len(allmini_only_top5) / total_queries * 100
            },
            
            # Detailed queries
            'overlap_top1_queries': list(overlap_top1),
            'overlap_top5_queries': list(overlap_top5),
            'sapbert_only_top1_queries': list(sapbert_only_top1),
            'allmini_only_top1_queries': list(allmini_only_top1),
            'sapbert_only_top5_queries': list(sapbert_only_top5),
            'allmini_only_top5_queries': list(allmini_only_top5)
        }
        
        return overlap_analysis
    
    def print_overlap_summary(self, overlap_analysis: Dict[str, Any]) -> None:
        """Imprime un resumen detallado del solapamiento."""
        print("\n" + "="*100)
        print("ANÁLISIS DE SOLAPAMIENTO ENTRE SAPBERT Y ALL-MINI")
        print("="*100)
        
        total = overlap_analysis['total_queries']
        
        print(f"\n📊 RESUMEN GENERAL:")
        print(f"   Total de queries: {total}")
        
        print(f"\n🎯 TOP1 ACCURACY:")
        top1 = overlap_analysis['top1']
        print(f"   SAPBERT correctas: {top1['sapbert_correct']}/{total} ({top1['sapbert_correct']/total*100:.1f}%)")
        print(f"   All-Mini correctas: {top1['allmini_correct']}/{total} ({top1['allmini_correct']/total*100:.1f}%)")
        print(f"   Solapamiento: {top1['overlap']}/{total} ({top1['overlap_percentage']:.1f}%)")
        print(f"   Solo SAPBERT: {top1['sapbert_only']}/{total} ({top1['sapbert_unique_percentage']:.1f}%)")
        print(f"   Solo All-Mini: {top1['allmini_only']}/{total} ({top1['allmini_unique_percentage']:.1f}%)")
        
        print(f"\n📈 TOP5 ACCURACY:")
        top5 = overlap_analysis['top5']
        print(f"   SAPBERT correctas: {top5['sapbert_correct']}/{total} ({top5['sapbert_correct']/total*100:.1f}%)")
        print(f"   All-Mini correctas: {top5['allmini_correct']}/{total} ({top5['allmini_correct']/total*100:.1f}%)")
        print(f"   Solapamiento: {top5['overlap']}/{total} ({top5['overlap_percentage']:.1f}%)")
        print(f"   Solo SAPBERT: {top5['sapbert_only']}/{total} ({top5['sapbert_unique_percentage']:.1f}%)")
        print(f"   Solo All-Mini: {top5['allmini_only']}/{total} ({top5['allmini_unique_percentage']:.1f}%)")
        
        print(f"\n🔍 QUERIES ÚNICAS DE SAPBERT (Top1):")
        for query in overlap_analysis['sapbert_only_top1_queries'][:5]:  # Mostrar solo las primeras 5
            print(f"   - {query}")
        if len(overlap_analysis['sapbert_only_top1_queries']) > 5:
            print(f"   ... y {len(overlap_analysis['sapbert_only_top1_queries']) - 5} más")
        
        print(f"\n🔍 QUERIES ÚNICAS DE ALL-MINI (Top1):")
        for query in overlap_analysis['allmini_only_top1_queries'][:5]:  # Mostrar solo las primeras 5
            print(f"   - {query}")
        if len(overlap_analysis['allmini_only_top1_queries']) > 5:
            print(f"   ... y {len(overlap_analysis['allmini_only_top1_queries']) - 5} más")
        
        print(f"\n💡 CONCLUSIONES:")
        sapbert_improvement = (top1['sapbert_correct'] - top1['allmini_correct']) / total * 100
        print(f"   SAPBERT mejora Top1 en: {sapbert_improvement:+.1f}%")
        
        if top1['sapbert_only'] > top1['allmini_only']:
            print(f"   SAPBERT resuelve {top1['sapbert_only'] - top1['allmini_only']} queries más únicamente")
        else:
            print(f"   All-Mini resuelve {top1['allmini_only'] - top1['sapbert_only']} queries más únicamente")
    
    def run_analysis(self, top_k: int = 5) -> Dict[str, Any]:
        """Ejecuta el análisis completo de solapamiento."""
        print("=== INICIANDO ANÁLISIS DE SOLAPAMIENTO ===")
        
        # Cargar documentos y queries
        self.load_pnts_documents()
        self.benchmark_queries = self.load_benchmark_queries()
        
        if not self.benchmark_queries:
            print("No se pudieron cargar las queries de benchmark")
            return {}
        
        # Probar ambos modelos
        self.sapbert_results = self.test_sapbert_model(self.benchmark_queries, top_k)
        self.allmini_results = self.test_allmini_model(self.benchmark_queries, top_k)
        
        # Analizar solapamiento
        overlap_analysis = self.analyze_overlap()
        
        # Mostrar resultados
        self.print_overlap_summary(overlap_analysis)
        
        # Guardar resultados
        timestamp = int(time.time())
        results_file = f"overlap_analysis_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(overlap_analysis, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResultados guardados en: {results_file}")
        
        return overlap_analysis


def main():
    parser = argparse.ArgumentParser(description="Análisis de solapamiento entre SAPBERT y All-Mini")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados top-k a evaluar")
    
    args = parser.parse_args()
    
    analyzer = OverlapAnalyzer()
    results = analyzer.run_analysis(args.top_k)
    
    print("\n¡Análisis de solapamiento completado exitosamente!")


if __name__ == "__main__":
    main()
