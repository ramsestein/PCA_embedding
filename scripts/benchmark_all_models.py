#!/usr/bin/env python3
"""
Script para probar todos los modelos de embeddings disponibles en sapbert-umls
y evaluar cuál tiene el mejor rendimiento en el benchmark de PNTs.
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
from tqdm import tqdm

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rerank_markov.types import Chunk, ScoredChunk
from rerank_markov.utils import tokenize
from rerank_markov.index_stats import compute_corpus_statistics


class ModelBenchmarker:
    """Clase para hacer benchmark de múltiples modelos de embeddings."""
    
    def __init__(self):
        self.chunks = []
        self.corpus_stats = None
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
    
    def get_available_models(self) -> List[str]:
        """Obtiene la lista de modelos disponibles en sapbert-umls."""
        models_dir = Path("sapbert-umls")
        if not models_dir.exists():
            print("Directorio sapbert-umls no encontrado")
            return []
            
        models = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("model-"):
                models.append(str(model_dir))
                
        # Ordenar por número de epoch (ascendente) con manejo de errores
        def safe_sort_key(model_path):
            try:
                # Extraer el número después de "model-"
                model_name = os.path.basename(model_path)
                number_part = model_name.split("model-")[1]
                
                # Manejar diferentes formatos
                if "_" in number_part:
                    # Formato: model-0_0000_1 -> 0.0000.1
                    parts = number_part.split("_")
                    if len(parts) >= 2:
                        return float(f"{parts[0]}.{parts[1]}")
                    else:
                        return float(parts[0])
                else:
                    # Formato: model-0_4735 -> 0.4735
                    return float(number_part.replace("_", "."))
            except (ValueError, IndexError):
                # Si no se puede convertir, usar 0.0 como valor por defecto
                return 0.0
        
        models.sort(key=safe_sort_key)
        return models
    
    def test_model(self, model_path: str, queries: List[Dict[str, str]], top_k: int = 5) -> Dict[str, Any]:
        """Prueba un modelo específico y retorna métricas de rendimiento."""
        print(f"\n=== Probando modelo: {os.path.basename(model_path)} ===")
        
        try:
            # Cargar el modelo
            start_time = time.time()
            model = SentenceTransformer(model_path)
            load_time = time.time() - start_time
            print(f"Modelo cargado en {load_time:.2f} segundos")
            
            # Calcular embeddings para todos los chunks
            print("Calculando embeddings...")
            start_time = time.time()
            
            texts = [chunk.text for chunk in self.chunks]
            embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
            
            # Asignar embeddings a los chunks
            for i, chunk in enumerate(self.chunks):
                chunk.embedding = embeddings[i]
                
            embedding_time = time.time() - start_time
            print(f"Embeddings calculados en {embedding_time:.2f} segundos")
            
            # Evaluar el modelo
            results = self._evaluate_embedding_model(model, queries, top_k)
            
            # Agregar métricas de tiempo
            results['load_time'] = load_time
            results['embedding_time'] = embedding_time
            results['total_time'] = load_time + embedding_time
            
            return results
            
        except Exception as e:
            print(f"Error al cargar el modelo {model_path}: {e}")
            return {
                'model_name': os.path.basename(model_path),
                'error': str(e),
                'top1_accuracy': 0.0,
                'top5_accuracy': 0.0,
                'top1_correct': 0,
                'top5_correct': 0,
                'total_queries': len(queries)
            }
    
    def _evaluate_embedding_model(self, model: SentenceTransformer, queries: List[Dict[str, str]], top_k: int) -> Dict[str, Any]:
        """Evalúa un modelo de embeddings usando las queries de benchmark."""
        
        correct_top1 = 0
        correct_top5 = 0
        
        for i, query_data in enumerate(queries):
            query = query_data['query']
            expected_doc = query_data['expected_doc']
            
            # Normalizar el documento esperado para que coincida con doc_id
            expected_doc_normalized = expected_doc.replace('_limpio.txt', '_limpio')
            
            # Calcular embedding de la query
            query_embedding = model.encode([query])[0]
            
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
            
            # Verificar Top1
            top1_chunk = similarities[0][1]
            if top1_chunk.doc_id == expected_doc_normalized:
                correct_top1 += 1
            
            # Verificar Top5
            top5_chunks = [chunk for _, chunk in similarities[:top_k]]
            if any(chunk.doc_id == expected_doc_normalized for chunk in top5_chunks):
                correct_top5 += 1
        
        total_queries = len(queries)
        top1_accuracy = correct_top1 / total_queries if total_queries > 0 else 0.0
        top5_accuracy = correct_top5 / total_queries if total_queries > 0 else 0.0
        
        return {
            'model_name': model.model_name if hasattr(model, 'model_name') else 'Unknown',
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'top1_correct': correct_top1,
            'top5_correct': correct_top5,
            'total_queries': total_queries
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def run_full_benchmark(self, top_k: int = 5) -> Dict[str, Any]:
        """Ejecuta el benchmark completo de todos los modelos."""
        print("=== INICIANDO BENCHMARK DE TODOS LOS MODELOS ===")
        
        # Cargar documentos y queries
        self.load_pnts_documents()
        queries = self.load_benchmark_queries()
        
        if not queries:
            print("No se pudieron cargar las queries de benchmark")
            return {}
        
        # Obtener modelos disponibles
        models = self.get_available_models()
        print(f"\nModelos encontrados: {len(models)}")
        for model in models:
            print(f"  - {os.path.basename(model)}")
        
        # Probar cada modelo
        results = {}
        for model_path in models:
            try:
                model_results = self.test_model(model_path, queries, top_k)
                results[os.path.basename(model_path)] = model_results
            except Exception as e:
                print(f"Error al probar modelo {model_path}: {e}")
                continue
        
        # Mostrar resumen comparativo
        self._print_comparative_summary(results)
        
        # Guardar resultados
        timestamp = int(time.time())
        results_file = f"benchmark_all_models_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResultados guardados en: {results_file}")
        
        return results
    
    def _print_comparative_summary(self, results: Dict[str, Any]) -> None:
        """Imprime un resumen comparativo de todos los modelos."""
        print("\n" + "="*100)
        print("RESUMEN COMPARATIVO DE TODOS LOS MODELOS")
        print("="*100)
        
        # Filtrar modelos con errores
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("No hay resultados válidos para comparar")
            return
        
        # Ordenar por Top1 accuracy
        sorted_models = sorted(valid_results.items(), 
                              key=lambda x: x[1]['top1_accuracy'], 
                              reverse=True)
        
        print(f"{'Modelo':<25} {'Top1 Acc':<10} {'Top5 Acc':<10} {'Top1':<8} {'Top5':<8} {'Tiempo':<10}")
        print("-" * 100)
        
        for model_name, result in sorted_models:
            top1_acc = f"{result['top1_accuracy']:.3f}"
            top5_acc = f"{result['top5_accuracy']:.3f}"
            top1_correct = result['top1_correct']
            top5_correct = result['top5_correct']
            total_time = f"{result.get('total_time', 0):.1f}s"
            
            print(f"{model_name:<25} {top1_acc:<10} {top5_acc:<10} {top1_correct:<8} {top5_correct:<8} {total_time:<10}")
        
        # Mostrar el mejor modelo
        best_model = sorted_models[0]
        print(f"\n🏆 MEJOR MODELO: {best_model[0]}")
        print(f"   Top1 Accuracy: {best_model[1]['top1_accuracy']:.3f}")
        print(f"   Top5 Accuracy: {best_model[1]['top5_accuracy']:.3f}")
        print(f"   Tiempo total: {best_model[1].get('total_time', 0):.1f} segundos")


def main():
    parser = argparse.ArgumentParser(description="Benchmark de todos los modelos de embeddings")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados top-k a evaluar")
    
    args = parser.parse_args()
    
    benchmarker = ModelBenchmarker()
    results = benchmarker.run_full_benchmark(args.top_k)
    
    print("\n¡Benchmark completado exitosamente!")


if __name__ == "__main__":
    main()
