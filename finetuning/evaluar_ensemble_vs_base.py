#!/usr/bin/env python3
"""
evaluar_ensemble_vs_base.py - Script para evaluar un ensemble específico contra all-mini-base
Usa el método de promedio ponderado de similitudes
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

from cargador_models import UnifiedEmbeddingAdapter

class EnsembleVsBaseEvaluator:
    """Evaluador para comparar ensemble contra modelo base"""
    
    def __init__(self, ensemble_config, base_config, documents_path, test_queries_path):
        self.documents_path = Path(documents_path)
        self.test_queries_path = test_queries_path
        self.ensemble_config = ensemble_config
        self.results = {}
        
        # Cargar modelos del ensemble
        print("📦 Configurando ensemble con promedio ponderado...")
        self.ensemble_models = {}
        print("🔄 Cargando modelos del ensemble:")
        for config in ensemble_config['models']:
            print(f"   • Cargando {config['name']}...")
            self.ensemble_models[config['name']] = UnifiedEmbeddingAdapter(
                model_path=config['path'],
                model_name=config['name'],
                pooling_strategy=config.get('pooling', 'mean')
            )
        
        self.ensemble_weights = ensemble_config['weights']
        
        print(f"\n✅ Ensemble configurado (Promedio Ponderado):")
        for name, weight in zip(self.ensemble_models.keys(), self.ensemble_weights):
            print(f"   • {name}: {weight:.3f}")
        
        # Cargar modelo base
        print(f"\n📦 Cargando modelo base: {base_config['name']}...")
        self.base_model = UnifiedEmbeddingAdapter(
            model_path=base_config['path'],
            model_name=base_config['name'],
            pooling_strategy=base_config.get('pooling', 'mean')
        )
        
        # Crear índices de documentos para cada modelo
        self.doc_chunks_per_model = {}
    
    def chunk_text(self, text, chunk_size=384, chunk_overlap=128):
        """Divide texto en chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_document_embeddings(self):
        """Crea embeddings para todos los documentos con cada modelo"""
        print("\n📚 Creando índices de documentos...")
        
        doc_files = list(self.documents_path.glob("*.txt"))
        
        # Para el modelo base
        print("\n  Procesando documentos con all-MiniLM-base...")
        base_chunks = []
        
        for doc_file in tqdm(doc_files, desc="  Indexando", leave=False):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                continue
            
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                embedding = self.base_model.embed(chunk)
                embedding = embedding / np.linalg.norm(embedding)
                
                base_chunks.append({
                    'id': f"{doc_file.name}_{i}",
                    'text': chunk,
                    'embedding': embedding,
                    'doc_name': doc_file.name,
                    'chunk_pos': i
                })
        
        self.doc_chunks_per_model['base'] = base_chunks
        
        # Para cada modelo del ensemble
        for model_name, model in self.ensemble_models.items():
            print(f"\n  Procesando documentos con {model_name}...")
            model_chunks = []
            
            for doc_file in tqdm(doc_files, desc="  Indexando", leave=False):
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                chunks = self.chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    embedding = model.embed(chunk)
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    model_chunks.append({
                        'id': f"{doc_file.name}_{i}",
                        'text': chunk,
                        'embedding': embedding,
                        'doc_name': doc_file.name,
                        'chunk_pos': i
                    })
            
            self.doc_chunks_per_model[model_name] = model_chunks
    
    def search_base_model(self, query, top_k=5):
        """Búsqueda con el modelo base"""
        # Generar embedding de la query
        start_time = time.time()
        query_embedding = self.base_model.embed(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        embed_time = time.time() - start_time
        
        # Calcular similitudes
        start_time = time.time()
        similarities = []
        
        for chunk_info in self.doc_chunks_per_model['base']:
            cos_sim = np.dot(query_embedding, chunk_info['embedding'])
            similarity = cos_sim * 100
            
            similarities.append({
                'doc_name': chunk_info['doc_name'],
                'chunk_text': chunk_info['text'],
                'similarity': similarity,
                'chunk_pos': chunk_info['chunk_pos']
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        search_time = time.time() - start_time
        
        return similarities[:top_k], embed_time + search_time
    
    def search_ensemble_weighted_average(self, query, top_k=5):
        """Búsqueda usando promedio ponderado de similitudes"""
        start_time = time.time()
        
        # Calcular similitudes para cada modelo por separado
        all_similarities = {}
        
        for model_name, model in self.ensemble_models.items():
            query_embedding = model.embed(query)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            doc_chunks = self.doc_chunks_per_model[model_name]
            
            similarities = {}
            for chunk_info in doc_chunks:
                cos_sim = np.dot(query_embedding, chunk_info['embedding'])
                doc_name = chunk_info['doc_name']
                
                if doc_name not in similarities:
                    similarities[doc_name] = {
                        'chunk_text': chunk_info['text'],
                        'chunk_pos': chunk_info['chunk_pos'],
                        'similarity': cos_sim
                    }
                else:
                    # Mantener el chunk con mayor similitud
                    if cos_sim > similarities[doc_name]['similarity']:
                        similarities[doc_name] = {
                            'chunk_text': chunk_info['text'],
                            'chunk_pos': chunk_info['chunk_pos'],
                            'similarity': cos_sim
                        }
            
            all_similarities[model_name] = similarities
        
        # Combinar similitudes ponderadas
        combined_scores = {}
        
        # Obtener todos los documentos únicos
        all_docs = set()
        for model_sims in all_similarities.values():
            all_docs.update(model_sims.keys())
        
        # Calcular score combinado para cada documento
        for doc_name in all_docs:
            combined_score = 0
            doc_info = None
            
            for i, model_name in enumerate(self.ensemble_models.keys()):
                if doc_name in all_similarities[model_name]:
                    sim_data = all_similarities[model_name][doc_name]
                    combined_score += self.ensemble_weights[i] * sim_data['similarity']
                    
                    # Guardar info del documento del primer modelo que lo tenga
                    if doc_info is None:
                        doc_info = {
                            'chunk_text': sim_data['chunk_text'],
                            'chunk_pos': sim_data['chunk_pos']
                        }
            
            combined_scores[doc_name] = {
                'doc_name': doc_name,
                'similarity': combined_score * 100,  # Convertir a porcentaje
                'chunk_text': doc_info['chunk_text'],
                'chunk_pos': doc_info['chunk_pos']
            }
        
        # Ordenar por score combinado
        sorted_docs = sorted(combined_scores.values(), 
                            key=lambda x: x['similarity'], 
                            reverse=True)
        
        total_time = time.time() - start_time
        
        return sorted_docs[:top_k], total_time
    
    def evaluate_models(self):
        """Evalúa ambos modelos con todas las queries"""
        print("\n🔍 Evaluando modelos...")
        
        # Cargar queries de prueba
        with open(self.test_queries_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        # Resultados para modelo base
        base_results = []
        base_total_time = 0
        
        # Resultados para ensemble
        ensemble_results = []
        ensemble_total_time = 0
        
        for case in tqdm(test_cases, desc="Evaluando queries"):
            query = case['query']
            expected_doc = case['document_expected']
            
            # Evaluar modelo base
            base_similar, base_time = self.search_base_model(query)
            base_total_time += base_time
            
            # Verificar resultados del modelo base
            base_found = False
            base_position = -1
            base_similarity = 0
            
            for i, chunk in enumerate(base_similar):
                if chunk['doc_name'] == expected_doc:
                    base_found = True
                    base_position = i + 1
                    base_similarity = chunk['similarity']
                    break
            
            base_results.append({
                'query': query,
                'expected_doc': expected_doc,
                'found': base_found,
                'position': base_position,
                'similarity': base_similarity
            })
            
            # Evaluar ensemble
            ensemble_similar, ensemble_time = self.search_ensemble_weighted_average(query)
            ensemble_total_time += ensemble_time
            
            # Verificar resultados del ensemble
            ensemble_found = False
            ensemble_position = -1
            ensemble_similarity = 0
            
            for i, chunk in enumerate(ensemble_similar):
                if chunk['doc_name'] == expected_doc:
                    ensemble_found = True
                    ensemble_position = i + 1
                    ensemble_similarity = chunk['similarity']
                    break
            
            ensemble_results.append({
                'query': query,
                'expected_doc': expected_doc,
                'found': ensemble_found,
                'position': ensemble_position,
                'similarity': ensemble_similarity
            })
        
        # Calcular métricas
        base_metrics = self.calculate_metrics(base_results)
        base_metrics['avg_query_time'] = base_total_time / len(test_cases)
        
        ensemble_metrics = self.calculate_metrics(ensemble_results)
        ensemble_metrics['avg_query_time'] = ensemble_total_time / len(test_cases)
        
        self.results = {
            'base': {
                'metrics': base_metrics,
                'results': base_results
            },
            'ensemble': {
                'metrics': ensemble_metrics,
                'results': ensemble_results
            }
        }
        
        return self.results
    
    def calculate_metrics(self, results):
        """Calcula métricas completas de evaluación"""
        total = len(results)
        
        # Métricas básicas
        acc_at_1 = sum(1 for r in results if r['position'] == 1) / total
        acc_at_5 = sum(1 for r in results if r['found']) / total
        
        # MRR
        mrr = sum(1/r['position'] for r in results if r['position'] > 0) / total
        
        # MAP
        ap_scores = []
        for r in results:
            if r['found']:
                ap_scores.append(1.0 / r['position'])
            else:
                ap_scores.append(0.0)
        map_score = np.mean(ap_scores)
        
        # NDCG@5
        ndcg_scores = []
        for r in results:
            if r['found'] and r['position'] <= 5:
                ndcg = 1.0 / np.log2(r['position'] + 1)
            else:
                ndcg = 0.0
            ndcg_scores.append(ndcg)
        ndcg_at_5 = np.mean(ndcg_scores)
        
        # Similitud promedio
        found_results = [r for r in results if r['found']]
        avg_similarity = sum(r['similarity'] for r in found_results) / len(found_results) if found_results else 0
        
        return {
            'accuracy_at_1': acc_at_1,
            'accuracy_at_5': acc_at_5,
            'mrr': mrr,
            'map': map_score,
            'ndcg_at_5': ndcg_at_5,
            'avg_similarity': avg_similarity,
            'not_found': sum(1 for r in results if not r['found']),
            'total_queries': total
        }
    
    def print_evaluation_report(self):
        """Imprime reporte detallado de evaluación"""
        print("\n" + "="*80)
        print("RESULTADOS DE LA EVALUACIÓN - PROMEDIO PONDERADO")
        print("="*80)
        
        # Información del ensemble
        print(f"\n📊 ENSEMBLE (Promedio Ponderado):")
        print(f"   Componentes:")
        for model, weight in zip(self.ensemble_models.keys(), self.ensemble_weights):
            print(f"   • {model}: {weight:.3f}")
        
        # Tabla comparativa
        print("\n" + "-"*80)
        print(f"{'Modelo':<40} {'Acc@1':<10} {'Acc@5':<10} {'MRR':<10} {'MAP':<10} {'NDCG@5':<10}")
        print("-"*80)
        
        # Modelo base
        base_metrics = self.results['base']['metrics']
        print(f"{'all-MiniLM-base':<40} "
              f"{base_metrics['accuracy_at_1']:<10.1%} "
              f"{base_metrics['accuracy_at_5']:<10.1%} "
              f"{base_metrics['mrr']:<10.3f} "
              f"{base_metrics['map']:<10.3f} "
              f"{base_metrics['ndcg_at_5']:<10.3f}")
        
        # Ensemble
        ensemble_metrics = self.results['ensemble']['metrics']
        print(f"{'Ensemble (Promedio Ponderado)':<40} "
              f"{ensemble_metrics['accuracy_at_1']:<10.1%} "
              f"{ensemble_metrics['accuracy_at_5']:<10.1%} "
              f"{ensemble_metrics['mrr']:<10.3f} "
              f"{ensemble_metrics['map']:<10.3f} "
              f"{ensemble_metrics['ndcg_at_5']:<10.3f}")
        
        # Mejoras
        print("\n" + "="*80)
        print("ANÁLISIS DE MEJORAS")
        print("="*80)
        
        acc1_diff = (ensemble_metrics['accuracy_at_1'] - base_metrics['accuracy_at_1']) * 100
        acc5_diff = (ensemble_metrics['accuracy_at_5'] - base_metrics['accuracy_at_5']) * 100
        mrr_diff = ensemble_metrics['mrr'] - base_metrics['mrr']
        map_diff = ensemble_metrics['map'] - base_metrics['map']
        ndcg_diff = ensemble_metrics['ndcg_at_5'] - base_metrics['ndcg_at_5']
        
        print(f"\n📈 Mejoras del Ensemble sobre all-MiniLM-base:")
        print(f"   • Accuracy@1: {'+' if acc1_diff > 0 else ''}{acc1_diff:.1f} pp "
              f"({base_metrics['accuracy_at_1']:.1%} → {ensemble_metrics['accuracy_at_1']:.1%})")
        print(f"   • Accuracy@5: {'+' if acc5_diff > 0 else ''}{acc5_diff:.1f} pp "
              f"({base_metrics['accuracy_at_5']:.1%} → {ensemble_metrics['accuracy_at_5']:.1%})")
        print(f"   • MRR: {'+' if mrr_diff > 0 else ''}{mrr_diff:.3f} "
              f"({base_metrics['mrr']:.3f} → {ensemble_metrics['mrr']:.3f})")
        print(f"   • MAP: {'+' if map_diff > 0 else ''}{map_diff:.3f} "
              f"({base_metrics['map']:.3f} → {ensemble_metrics['map']:.3f})")
        print(f"   • NDCG@5: {'+' if ndcg_diff > 0 else ''}{ndcg_diff:.3f} "
              f"({base_metrics['ndcg_at_5']:.3f} → {ensemble_metrics['ndcg_at_5']:.3f})")
        
        # Velocidad
        base_time = self.results['base']['metrics']['avg_query_time']
        ensemble_time = self.results['ensemble']['metrics']['avg_query_time']
        speed_ratio = ensemble_time / base_time
        
        print(f"\n⏱️  Velocidad:")
        print(f"   • all-MiniLM-base: {base_time:.4f}s/query")
        print(f"   • Ensemble: {ensemble_time:.4f}s/query")
        print(f"   • Factor: {speed_ratio:.2f}x {'más lento' if speed_ratio > 1 else 'más rápido'}")
        
        # Resumen
        print("\n" + "="*80)
        print("💡 RESUMEN")
        print("="*80)
        
        if acc1_diff > 0 and acc5_diff > 0 and mrr_diff > 0:
            print("✅ El ensemble muestra mejoras consistentes en todas las métricas principales")
            if speed_ratio < 3:
                print("✅ El overhead de velocidad es aceptable para las mejoras obtenidas")
            else:
                print("⚠️  El ensemble es significativamente más lento")
        else:
            print("⚠️  El ensemble no muestra mejoras claras sobre el modelo base")
    
    def save_results(self, output_dir="resultados_ensemble_cat"):
        """Guarda los resultados detallados"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar reporte
        with open(output_dir / f"evaluacion_ensemble_{timestamp}.txt", 'w', encoding='utf-8') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            self.print_evaluation_report()
            sys.stdout = old_stdout
        
        # Guardar resultados JSON
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        results_data = {
            'method': 'weighted_average',
            'ensemble_config': {
                'models': [{'name': name} for name in self.ensemble_models.keys()],
                'weights': convert_to_native(self.ensemble_weights)
            },
            'base_model': 'all-MiniLM-base',
            'metrics': {
                'base': convert_to_native(self.results['base']['metrics']),
                'ensemble': convert_to_native(self.results['ensemble']['metrics'])
            },
            'improvements': {
                'accuracy_at_1_pp': float((self.results['ensemble']['metrics']['accuracy_at_1'] - 
                                         self.results['base']['metrics']['accuracy_at_1']) * 100),
                'accuracy_at_5_pp': float((self.results['ensemble']['metrics']['accuracy_at_5'] - 
                                         self.results['base']['metrics']['accuracy_at_5']) * 100),
                'mrr_diff': float(self.results['ensemble']['metrics']['mrr'] - 
                                self.results['base']['metrics']['mrr']),
                'map_diff': float(self.results['ensemble']['metrics']['map'] - 
                                self.results['base']['metrics']['map']),
                'ndcg_at_5_diff': float(self.results['ensemble']['metrics']['ndcg_at_5'] - 
                                      self.results['base']['metrics']['ndcg_at_5'])
            }
        }
        
        with open(output_dir / f"resultados_ensemble_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Resultados guardados en: {output_dir}/")
    
    def plot_comparison(self, output_dir="resultados_ensemble_cat"):
        """Genera visualizaciones comparativas"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Preparar datos
        metrics_names = ['accuracy_at_1', 'accuracy_at_5', 'mrr', 'map', 'ndcg_at_5']
        metrics_labels = ['Acc@1', 'Acc@5', 'MRR', 'MAP', 'NDCG@5']
        base_values = [self.results['base']['metrics'][m] for m in metrics_names]
        ensemble_values = [self.results['ensemble']['metrics'][m] for m in metrics_names]
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico 1: Comparación de métricas
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, base_values, width, label='all-MiniLM-base', color='#3498db')
        bars2 = ax1.bar(x + width/2, ensemble_values, width, label='Ensemble (Prom. Ponderado)', color='#2ecc71')
        
        ax1.set_xlabel('Métricas')
        ax1.set_ylabel('Valor')
        ax1.set_title('Comparación: Ensemble (Promedio Ponderado) vs Base')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_labels)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Añadir valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico 2: Mejoras
        improvements = []
        for i in range(len(metrics_names)):
            if metrics_names[i].startswith('accuracy'):
                imp = (ensemble_values[i] - base_values[i]) * 100
            else:
                imp = ((ensemble_values[i] - base_values[i]) / base_values[i]) * 100
            improvements.append(imp)
        
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax2.bar(metrics_labels, improvements, color=colors)
        
        ax2.set_xlabel('Métricas')
        ax2.set_ylabel('Mejora (%)')
        ax2.set_title('Mejoras del Ensemble sobre el Modelo Base')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', alpha=0.3)
        
        # Añadir valores
        for bar, val in zip(bars, improvements):
            if val >= 0:
                va = 'bottom'
                y = val + 0.5
            else:
                va = 'top'
                y = val - 0.5
            ax2.text(bar.get_x() + bar.get_width()/2., y,
                    f'{val:.1f}%', ha='center', va=va)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(output_dir / f"comparacion_ensemble_{timestamp}.png", dpi=300)
        plt.close()
        
        print(f"📊 Gráficos guardados en: {output_dir}/")


def main():
    print("🚀 Evaluación de Ensemble vs all-MiniLM-base")
    print("📊 Método: PROMEDIO PONDERADO")
    print("="*60)
    
    # Configuración del ensemble específico
    ensemble_config = {
        'models': [
            {
                'path': 'models/pubmedbert-marco/model-0_3372',
                'name': 'pubmedbert-marco',
                'pooling': 'mean'
            },
            {
                'path': 'models/sapbert-umls/model-0_0001',
                'name': 'sapbert-umls',
                'pooling': 'mean'
            }
        ],
        'weights': [0.571, 0.429]
    }
    
    # Configuración del modelo base
    base_config = {
        'path': './all-mini-base',
        'name': 'all-MiniLM-base',
        'pooling': 'mean'
    }
    
    # Paths
    documents_path = './PNTs'
    test_queries_path = './preguntas_con_docs_cat.json'
    
    # Verificar paths
    for model in ensemble_config['models']:
        if not Path(model['path']).exists():
            print(f"❌ No se encuentra: {model['path']}")
            return
    
    if not Path(base_config['path']).exists():
        print(f"❌ No se encuentra modelo base: {base_config['path']}")
        return
    
    if not Path(documents_path).exists():
        print(f"❌ No se encuentra carpeta de documentos: {documents_path}")
        return
        
    if not Path(test_queries_path).exists():
        print(f"❌ No se encuentra archivo de pruebas: {test_queries_path}")
        return
    
    # Crear evaluador
    evaluator = EnsembleVsBaseEvaluator(
        ensemble_config=ensemble_config,
        base_config=base_config,
        documents_path=documents_path,
        test_queries_path=test_queries_path
    )
    
    # Crear índices de documentos
    evaluator.create_document_embeddings()
    
    # Ejecutar evaluación
    results = evaluator.evaluate_models()
    
    # Mostrar resultados
    evaluator.print_evaluation_report()
    
    # Guardar resultados
    evaluator.save_results()
    
    # Generar gráficos
    evaluator.plot_comparison()
    
    print("\n✅ Evaluación completada!")


if __name__ == "__main__":
    main()