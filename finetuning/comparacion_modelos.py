#!/usr/bin/env python3
"""
comparar_embeddings.py - Script simplificado de comparación de embeddings
Adaptado para tu estructura de carpetas actual
"""

import json
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Importar tu cargador de modelos
from cargador_models import UnifiedEmbeddingAdapter

class SimpleEmbeddingComparator:
    """Comparador simplificado de embeddings"""
    
    def __init__(self, models_config, documents_path, test_queries_path):
        self.documents_path = Path(documents_path)
        self.test_queries_path = test_queries_path
        self.models = {}
        self.results = {}
        
        # Cargar modelos
        print("📦 Cargando modelos...")
        for name, config in models_config.items():
            self.models[name] = UnifiedEmbeddingAdapter(
                model_path=config['path'],
                model_name=config['name'],
                pooling_strategy=config.get('pooling', 'mean')
            )
    
    def chunk_text(self, text, chunk_size=384, chunk_overlap=128):
        """Divide texto en chunks (igual que el proyecto original)"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_document_embeddings(self, model_name):
        """Crea embeddings para todos los documentos"""
        print(f"\n📚 Procesando documentos con {model_name}...")
        
        model = self.models[model_name]
        doc_embeddings = {}
        all_chunks = []
        chunk_to_doc = {}
        
        # Procesar cada documento
        doc_files = list(self.documents_path.glob("*.txt"))
        
        for doc_file in tqdm(doc_files, desc="Indexando documentos"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                continue
            
            # Dividir en chunks
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                # Generar embedding
                embedding = model.embed(chunk)
                
                # Guardar información
                chunk_id = f"{doc_file.name}_{i}"
                all_chunks.append({
                    'id': chunk_id,
                    'text': chunk,
                    'embedding': embedding,
                    'doc_name': doc_file.name,
                    'chunk_pos': i
                })
                chunk_to_doc[chunk_id] = doc_file.name
        
        return all_chunks, chunk_to_doc
    
    def search_similar_chunks(self, query, model_name, doc_chunks, top_k=5):
        """Busca chunks similares a la query"""
        model = self.models[model_name]
        
        # Generar embedding de la query
        start_time = time.time()
        query_embedding = model.embed(query)
        embed_time = time.time() - start_time
        
        # Calcular similitudes
        start_time = time.time()
        similarities = []
        
        for chunk_info in doc_chunks:
            # Similitud coseno
            cos_sim = np.dot(query_embedding, chunk_info['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_info['embedding'])
            )
            # Convertir a porcentaje (como en el proyecto original)
            similarity = cos_sim * 100
            
            similarities.append({
                'doc_name': chunk_info['doc_name'],
                'chunk_text': chunk_info['text'],
                'similarity': similarity,
                'chunk_pos': chunk_info['chunk_pos']
            })
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        search_time = time.time() - start_time
        
        return similarities[:top_k], {
            'embed_time': embed_time,
            'search_time': search_time,
            'total_time': embed_time + search_time
        }
    
    def evaluate_model(self, model_name):
        """Evalúa un modelo con todas las queries"""
        print(f"\n🔍 Evaluando {model_name}...")
        
        # Crear embeddings de documentos
        doc_chunks, chunk_to_doc = self.create_document_embeddings(model_name)
        
        # Cargar queries de prueba
        with open(self.test_queries_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        results = []
        total_time = 0
        
        for case in tqdm(test_cases, desc="Evaluando queries"):
            query = case['query']
            expected_doc = case['document_expected']
            
            # Buscar
            similar_chunks, timing = self.search_similar_chunks(
                query, model_name, doc_chunks
            )
            total_time += timing['total_time']
            
            # Verificar si encontró el documento esperado
            found = False
            position = -1
            similarity = 0
            top_doc = similar_chunks[0]['doc_name'] if similar_chunks else None
            
            # Buscar el documento esperado en los resultados
            for i, chunk in enumerate(similar_chunks):
                if chunk['doc_name'] == expected_doc:
                    found = True
                    position = i + 1
                    similarity = chunk['similarity']
                    break
            
            results.append({
                'query': query,
                'expected_doc': expected_doc,
                'found': found,
                'position': position,
                'similarity': similarity,
                'top_doc': top_doc,
                'timing': timing
            })
        
        # Calcular métricas
        metrics = self.calculate_metrics(results)
        metrics['avg_query_time'] = total_time / len(test_cases)
        
        return results, metrics
    
    def calculate_metrics(self, results):
        """Calcula métricas de evaluación"""
        total = len(results)
        
        # Accuracy@1 y @5
        acc_at_1 = sum(1 for r in results if r['position'] == 1) / total
        acc_at_5 = sum(1 for r in results if r['found']) / total
        
        # MRR
        mrr = sum(1/r['position'] for r in results if r['position'] > 0) / total
        
        # Similitud promedio cuando encuentra
        found_results = [r for r in results if r['found']]
        avg_similarity = sum(r['similarity'] for r in found_results) / len(found_results) if found_results else 0
        
        return {
            'accuracy_at_1': acc_at_1,
            'accuracy_at_5': acc_at_5,
            'mrr': mrr,
            'avg_similarity': avg_similarity,
            'not_found': sum(1 for r in results if not r['found']),
            'total_queries': total
        }
    
    def run_comparison(self):
        """Ejecuta la comparación completa"""
        print("="*60)
        print("COMPARACIÓN DE EMBEDDINGS")
        print("="*60)
        
        for model_name in self.models:
            results, metrics = self.evaluate_model(model_name)
            
            self.results[model_name] = {
                'metrics': metrics,
                'results': results,
                'model_info': {
                    'name': self.models[model_name].get_name(),
                    'dimension': self.models[model_name].get_dimension()
                }
            }
        
        return self.results
    
    def print_comparison_report(self):
        """Imprime reporte comparativo"""
        print("\n" + "="*60)
        print("RESULTADOS DE LA COMPARACIÓN")
        print("="*60)
        
        # Tabla de métricas
        print(f"\n{'Modelo':<20} {'Dim':<6} {'Acc@1':<10} {'Acc@5':<10} {'MRR':<10} {'Sim Avg':<12} {'No Found':<10}")
        print("-"*78)
        
        for model_name, data in self.results.items():
            metrics = data['metrics']
            print(f"{data['model_info']['name']:<20} "
                  f"{data['model_info']['dimension']:<6} "
                  f"{metrics['accuracy_at_1']:<10.1%} "
                  f"{metrics['accuracy_at_5']:<10.1%} "
                  f"{metrics['mrr']:<10.3f} "
                  f"{metrics['avg_similarity']:<12.1f}% "
                  f"{metrics['not_found']:<10}")
        
        # Análisis comparativo
        if len(self.results) == 2:
            models = list(self.results.keys())
            m1_metrics = self.results[models[0]]['metrics']
            m2_metrics = self.results[models[1]]['metrics']
            
            print("\n" + "="*60)
            print("ANÁLISIS COMPARATIVO")
            print("="*60)
            
            # Calcular diferencias
            acc1_diff = (m2_metrics['accuracy_at_1'] - m1_metrics['accuracy_at_1']) * 100
            acc5_diff = (m2_metrics['accuracy_at_5'] - m1_metrics['accuracy_at_5']) * 100
            mrr_diff = m2_metrics['mrr'] - m1_metrics['mrr']
            
            print(f"\nComparando {models[1]} vs {models[0]}:")
            print(f"  • Accuracy@1: {'+' if acc1_diff > 0 else ''}{acc1_diff:.1f} puntos porcentuales")
            print(f"  • Accuracy@5: {'+' if acc5_diff > 0 else ''}{acc5_diff:.1f} puntos porcentuales")
            print(f"  • MRR: {'+' if mrr_diff > 0 else ''}{mrr_diff:.3f}")
            
            # Velocidad
            avg_time_1 = self.results[models[0]]['metrics']['avg_query_time']
            avg_time_2 = self.results[models[1]]['metrics']['avg_query_time']
            speed_ratio = avg_time_1 / avg_time_2
            
            print(f"  • Velocidad: {models[1]} es {speed_ratio:.2f}x {'más rápido' if speed_ratio > 1 else 'más lento'}")
            
            # Recomendación
            print("\n🎯 RECOMENDACIÓN:")
            if acc1_diff > 5:
                print("✅ El segundo modelo muestra mejoras significativas")
            elif acc1_diff > 0:
                print("⚡ El segundo modelo muestra mejoras moderadas")
            else:
                print("⚠️  No hay mejoras claras con el segundo modelo")
    
    def save_results(self, output_dir="resultados_comparacion"):
        """Guarda los resultados en archivos"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar métricas
        with open(f"{output_dir}/metricas_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write("COMPARACIÓN DE EMBEDDINGS - MÉTRICAS\n")
            f.write("="*60 + "\n\n")
            
            for model_name, data in self.results.items():
                f.write(f"Modelo: {data['model_info']['name']}\n")
                f.write(f"Dimensión: {data['model_info']['dimension']}\n")
                f.write("-"*40 + "\n")
                
                metrics = data['metrics']
                for key, value in metrics.items():
                    if isinstance(value, float):
                        if 'accuracy' in key:
                            f.write(f"{key}: {value:.1%}\n")
                        else:
                            f.write(f"{key}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
        
        # Guardar resultados detallados en CSV
        for model_name, data in self.results.items():
            df = pd.DataFrame(data['results'])
            df.to_csv(f"{output_dir}/resultados_{model_name}_{timestamp}.csv", 
                     index=False, encoding='utf-8')
        
        print(f"\n📁 Resultados guardados en: {output_dir}/")
    
    def plot_comparison(self, output_dir="resultados_comparacion"):
        """Genera gráficos comparativos"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Preparar datos
        models = list(self.results.keys())
        metrics_names = ['accuracy_at_1', 'accuracy_at_5', 'mrr']
        
        # Crear figura
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics_names):
            values = [self.results[m]['metrics'][metric] for m in models]
            model_names = [self.results[m]['model_info']['name'] for m in models]
            
            axes[i].bar(model_names, values, color=['#3498db', '#e74c3c'])
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylim(0, 1)
            
            # Añadir valores
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparacion_metricas.png", dpi=300)
        plt.close()
        
        print(f"📊 Gráficos guardados en: {output_dir}/")


# Script principal
def main():
    print("🚀 Iniciando comparación de embeddings\n")
    
    # Configuración de modelos
    models_config = {
        'minilm': {
            'path': './all-mini-base',  # Tu carpeta actual
            'name': 'all-MiniLM-base',
            'pooling': 'mean'
        },
        'roberta': {
            'path': './models/biomedical-agressive/model-0_0088',  # Tu carpeta actual
            'name': 'biomedical-0.0088',
            'pooling': 'mean'
        }
    }
    
    # Paths
    documents_path = './PNTs'  # Tu carpeta de documentos
    test_queries_path = './preguntas_con_docs_cat.json'
    
    # Verificar que existen los paths
    for model_config in models_config.values():
        if not Path(model_config['path']).exists():
            print(f"❌ No se encuentra: {model_config['path']}")
            return
    
    if not Path(documents_path).exists():
        print(f"❌ No se encuentra carpeta de documentos: {documents_path}")
        return
        
    if not Path(test_queries_path).exists():
        print(f"❌ No se encuentra archivo de pruebas: {test_queries_path}")
        return
    
    # Crear comparador
    comparator = SimpleEmbeddingComparator(
        models_config=models_config,
        documents_path=documents_path,
        test_queries_path=test_queries_path
    )
    
    # Ejecutar comparación
    results = comparator.run_comparison()
    
    # Mostrar resultados
    comparator.print_comparison_report()
    
    # Guardar resultados
    comparator.save_results()
    
    # Generar gráficos
    comparator.plot_comparison()
    
    print("\n✅ Comparación completada!")


if __name__ == "__main__":
    main()