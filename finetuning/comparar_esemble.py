#!/usr/bin/env python3
"""
ensemble_embeddings.py - Sistema de Ensemble de Embeddings con doble estrategia
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
from itertools import product
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from cargador_models import UnifiedEmbeddingAdapter

class EnsembleEmbeddingSystem:
    """Sistema de Ensemble con promedio ponderado y fusión de rankings"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = []
        self.initial_weights = []
        self.documents_path = None
        self.test_queries_path = None
        self.doc_chunks_per_model = {}
        self.optimization_metric = 'accuracy_at_1'  # Por defecto
        self.results = {
            'weighted_average': {},
            'rank_fusion': {},
            'individual_models': {},
            'diversity_analysis': {}
        }
    
    def configure_ensemble(self):
        """Configura el ensemble con selección manual de modelos y pesos"""
        model_configs = []
        weights = []
        
        print("🔧 CONFIGURACIÓN DEL ENSEMBLE")
        print("="*60)
        
        # Paths de documentos y queries
        self.documents_path = Path('./PNTs')
        self.test_queries_path = './preguntas_con_docs_cat.json'
        
        # Selección de modelos
        print("\nSelecciona entre 2 y 5 modelos para el ensemble:")
        print("(Ingresa la ruta del modelo y un nombre descriptivo)")
        
        num_models = 0
        while num_models < 5:
            print(f"\nModelo {num_models + 1}:")
            path = input("  Ruta (Enter para terminar): ").strip()
            
            if not path and num_models >= 2:
                break
            elif not path and num_models < 2:
                print("  ⚠️  Necesitas al menos 2 modelos")
                continue
            
            if not Path(path).exists():
                print("  ❌ Ruta no encontrada")
                continue
            
            name = input("  Nombre del modelo: ").strip()
            if not name:
                name = f"modelo_{num_models + 1}"
            
            model_configs.append({
                'path': path,
                'name': name,
                'pooling': 'mean'
            })
            num_models += 1
        
        # Asignación de pesos iniciales
        print(f"\n📊 Asigna pesos iniciales para {len(model_configs)} modelos:")
        print("(Los pesos deben sumar 1.0)")
        
        while True:
            weights = []
            for i, config in enumerate(model_configs):
                weight = float(input(f"  Peso para {config['name']}: "))
                weights.append(weight)
            
            if abs(sum(weights) - 1.0) < 0.001:
                break
            else:
                print(f"  ⚠️  Los pesos suman {sum(weights):.3f}, deben sumar 1.0")
        
        self.model_configs = model_configs
        self.initial_weights = weights
        
        # Cargar modelos
        print("\n📦 Cargando modelos...")
        for config in model_configs:
            self.models[config['name']] = UnifiedEmbeddingAdapter(
                model_path=config['path'],
                model_name=config['name'],
                pooling_strategy=config.get('pooling', 'mean')
            )
        
        print("\n✅ Ensemble configurado:")
        for config, weight in zip(model_configs, weights):
            print(f"   • {config['name']}: peso inicial = {weight:.2f}")

    def select_optimization_metric(self):
        """Permite al usuario seleccionar qué métrica optimizar"""
        print("\n🎯 SELECCIÓN DE MÉTRICA PARA OPTIMIZACIÓN")
        print("="*60)
        print("\nMétricas disponibles:")
        print("1. accuracy_at_1 - Precisión en la primera posición")
        print("2. accuracy_at_5 - Precisión en las primeras 5 posiciones")
        print("3. mrr - Mean Reciprocal Rank")
        print("4. map - Mean Average Precision")
        print("5. ndcg_at_5 - Normalized Discounted Cumulative Gain @5")
        
        metrics_map = {
            '1': 'accuracy_at_1',
            '2': 'accuracy_at_5',
            '3': 'mrr',
            '4': 'map',
            '5': 'ndcg_at_5'
        }
        
        while True:
            choice = input("\nSelecciona métrica a optimizar (1-5): ").strip()
            if choice in metrics_map:
                self.optimization_metric = metrics_map[choice]
                print(f"\n✅ Optimizando por: {self.optimization_metric}")
                return
            else:
                print("❌ Opción inválida. Por favor selecciona 1-5.")
    
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
        print("\n📚 Creando embeddings de documentos para cada modelo...")
        
        doc_files = list(self.documents_path.glob("*.txt"))
        
        for model_name in self.models:
            print(f"\n  Procesando con {model_name}...")
            model = self.models[model_name]
            all_chunks = []
            chunk_to_doc = {}
            
            for doc_file in tqdm(doc_files, desc=f"  Indexando", leave=False):
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # Dividir en chunks
                chunks = self.chunk_text(content, 384, 128)
                
                for i, chunk in enumerate(chunks):
                    # Generar embedding
                    embedding = model.embed(chunk)
                    
                    # L2 normalization
                    embedding = embedding / np.linalg.norm(embedding)
                    
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
            
            self.doc_chunks_per_model[model_name] = {
                'chunks': all_chunks,
                'chunk_to_doc': chunk_to_doc
            }
    
    def weighted_average_search(self, query, weights):
        """Búsqueda usando promedio ponderado de similitudes (no embeddings)"""
        # Calcular similitudes para cada modelo por separado
        all_similarities = {}
        
        for model_name in self.models:
            model = self.models[model_name]
            query_embedding = model.embed(query)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            doc_chunks = self.doc_chunks_per_model[model_name]['chunks']
            
            similarities = {}
            for chunk_info in doc_chunks:
                # Similitud coseno
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
            
            for i, model_name in enumerate(self.models):
                if doc_name in all_similarities[model_name]:
                    sim_data = all_similarities[model_name][doc_name]
                    combined_score += weights[i] * sim_data['similarity']
                    
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
        
        return sorted_docs[:5]
    
    def rank_fusion_search(self, query, weights, k=60):
        """Búsqueda usando fusión de rankings con RRF ponderado"""

        # Obtener rankings de cada modelo
        all_rankings = {}
        
        for model_name in self.models:
            model = self.models[model_name]
            embedding = model.embed(query)
            embedding = embedding / np.linalg.norm(embedding)
            
            doc_chunks = self.doc_chunks_per_model[model_name]['chunks']
            
            similarities = []
            for chunk_info in doc_chunks:
                cos_sim = np.dot(embedding, chunk_info['embedding'])
                similarity = cos_sim * 100
                
                similarities.append({
                    'doc_name': chunk_info['doc_name'],
                    'chunk_text': chunk_info['text'],
                    'similarity': similarity,
                    'chunk_pos': chunk_info['chunk_pos']
                })
            
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            all_rankings[model_name] = similarities[:20]  # Top-20 para fusión
        
        # Calcular RRF scores ponderados
        rrf_scores = defaultdict(float)
        doc_info = {}
        
        for i, model_name in enumerate(self.models):
            weight = weights[i]
            rankings = all_rankings[model_name]
            
            for rank, item in enumerate(rankings):
                doc_name = item['doc_name']
                rrf_score = weight / (k + rank + 1)
                rrf_scores[doc_name] += rrf_score
                
                # Guardar info del documento
                if doc_name not in doc_info:
                    doc_info[doc_name] = item
        
        # Ordenar por RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Construir resultado final
        final_ranking = []
        for doc_name, score in sorted_docs[:5]:
            item = doc_info[doc_name].copy()
            item['rrf_score'] = score * 100  # Escalar para visualización
            final_ranking.append(item)
        
        return final_ranking
    
    def generate_weight_combinations(self):
        """Genera combinaciones de pesos para exploración"""
        n_models = len(self.initial_weights)
        weight_variations = []
        
        # Para cada peso inicial, generar variaciones ±50%
        for w in self.initial_weights:
            variations = [
                w * 0.5,   
                w * 0.66,
                w * 0.75,
                w * 0.85,  
                w * 1.0,
                w * 1.15,   
                w * 1.25,  
                w * 1.33,
                w * 1.5    
            ]
            weight_variations.append(variations)
        
        # Generar todas las combinaciones
        all_combinations = list(product(*weight_variations))
        
        # Filtrar y normalizar para que sumen 1.0
        valid_combinations = []
        for combo in all_combinations:
            total = sum(combo)
            if total > 0:  # Evitar división por cero
                normalized = [w/total for w in combo]
                valid_combinations.append(normalized)
        
        # Eliminar duplicados (con tolerancia)
        unique_combinations = []
        for combo in valid_combinations:
            is_duplicate = False
            for existing in unique_combinations:
                if all(abs(a - b) < 0.01 for a, b in zip(combo, existing)):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_combinations.append(combo)
        
        return unique_combinations
    
    def evaluate_weights(self, weights, method):
        """Evalúa una combinación de pesos específica"""
        
        # Cargar queries de prueba
        with open(self.test_queries_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        results = []
        
        for case in test_cases:
            query = case['query']
            expected_doc = case['document_expected']
            
            # Buscar según el método
            if method == 'weighted_average':
                similar_chunks = self.weighted_average_search(query, weights)
            else:  # rank_fusion
                similar_chunks = self.rank_fusion_search(query, weights)
            
            # Verificar si encontró el documento esperado
            found = False
            position = -1
            similarity = 0
            top_doc = similar_chunks[0]['doc_name'] if similar_chunks else None
            
            for i, chunk in enumerate(similar_chunks):
                if chunk['doc_name'] == expected_doc:
                    found = True
                    position = i + 1
                    similarity = chunk.get('similarity', chunk.get('rrf_score', 0))
                    break
            
            results.append({
                'query': query,
                'expected_doc': expected_doc,
                'found': found,
                'position': position,
                'similarity': similarity,
                'top_doc': top_doc
            })
        
        # Calcular métricas
        metrics = self.calculate_metrics(results)
        
        return results, metrics
    
    def calculate_metrics(self, results):
        """Calcula métricas completas de evaluación"""
        
        total = len(results)
        
        # Métricas básicas
        acc_at_1 = sum(1 for r in results if r['position'] == 1) / total
        acc_at_5 = sum(1 for r in results if r['found']) / total
        
        # MRR
        mrr = sum(1/r['position'] for r in results if r['position'] > 0) / total
        
        # MAP (Mean Average Precision)
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
    
    def find_optimal_weights(self, method, optimization_metric='accuracy_at_1'):
        """Encuentra los pesos óptimos para cada método según la métrica especificada"""
        
        print(f"\n🔍 Buscando pesos óptimos para {method} optimizando {optimization_metric}...")
        
        weight_combinations = self.generate_weight_combinations()
        print(f"   Evaluando {len(weight_combinations)} combinaciones de pesos...")
        
        best_weights = None
        best_metric_value = 0
        all_results = []
        
        for weights in tqdm(weight_combinations, desc="Evaluando combinaciones"):
            _, metrics = self.evaluate_weights(weights, method)
            
            all_results.append({
                'weights': weights,
                'metrics': metrics
            })
            
            # Usar la métrica especificada para optimización
            current_metric_value = metrics[optimization_metric]
            
            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_weights = weights
        
        print(f"   Mejor {optimization_metric}: {best_metric_value:.3f}")
        
        return best_weights, all_results
    
    def analyze_diversity(self):
        """Analiza la diversidad entre las predicciones de los modelos"""
        print("\n🔬 Analizando diversidad del ensemble...")
        
        # Cargar queries
        with open(self.test_queries_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        diversity_data = {
            'pairwise_overlap': {},
            'query_agreement': [],
            'model_contributions': defaultdict(int),
            'unique_correct_predictions': defaultdict(int)
        }
        
        # Para cada query, obtener predicciones de cada modelo
        all_predictions = defaultdict(lambda: defaultdict(list))
        
        for case in tqdm(test_cases, desc="Analizando predicciones"):
            query = case['query']
            expected_doc = case['document_expected']
            
            predictions_this_query = {}
            
            for model_name in self.models:
                model = self.models[model_name]
                embedding = model.embed(query)
                embedding = embedding / np.linalg.norm(embedding)
                
                doc_chunks = self.doc_chunks_per_model[model_name]['chunks']
                
                similarities = []
                for chunk_info in doc_chunks:
                    cos_sim = np.dot(embedding, chunk_info['embedding'])
                    similarity = cos_sim * 100
                    
                    similarities.append({
                        'doc_name': chunk_info['doc_name'],
                        'similarity': similarity
                    })
                
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                top_5_docs = [s['doc_name'] for s in similarities[:5]]
                
                predictions_this_query[model_name] = top_5_docs
                all_predictions[query][model_name] = top_5_docs
                
                # Verificar si este modelo fue el único en acertar
                if expected_doc in top_5_docs[:1]:  # Top-1
                    diversity_data['model_contributions'][model_name] += 1
            
            # Calcular agreement para esta query
            all_top1 = [preds[0] for preds in predictions_this_query.values()]
            unique_predictions = len(set(all_top1))
            agreement_score = 1.0 / unique_predictions  # 1.0 si todos predicen lo mismo
            
            diversity_data['query_agreement'].append({
                'query': query,
                'agreement_score': agreement_score,
                'unique_predictions': unique_predictions,
                'predictions': predictions_this_query
            })
        
        # Calcular overlap entre pares de modelos
        model_names = list(self.models.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                overlap_scores = []
                for query in all_predictions:
                    preds1 = set(all_predictions[query][model1][:5])
                    preds2 = set(all_predictions[query][model2][:5])
                    
                    if preds1 and preds2:
                        overlap = len(preds1.intersection(preds2)) / 5.0
                        overlap_scores.append(overlap)
                
                avg_overlap = np.mean(overlap_scores)
                diversity_data['pairwise_overlap'][f"{model1} vs {model2}"] = avg_overlap
        
        # Calcular en cuántas queries el ensemble supera a todos los individuales
        ensemble_wins = 0
        for case in test_cases:
            query = case['query']
            expected_doc = case['document_expected']
            
            # Verificar predicciones individuales
            any_individual_correct = False
            for model_name in self.models:
                if expected_doc == all_predictions[query][model_name][0]:
                    any_individual_correct = True
                    break
            
            # Verificar predicción del ensemble (usando pesos iniciales)
            ensemble_result = self.weighted_average_search(query, self.initial_weights)
            ensemble_correct = ensemble_result[0]['doc_name'] == expected_doc if ensemble_result else False
            
            if ensemble_correct and not any_individual_correct:
                ensemble_wins += 1
        
        diversity_data['ensemble_unique_wins'] = ensemble_wins
        diversity_data['average_agreement'] = np.mean([d['agreement_score'] for d in diversity_data['query_agreement']])
        
        return diversity_data
    
    def run_complete_evaluation(self):
        """Ejecuta la evaluación completa del ensemble"""
        print("\n" + "="*80)
        print("EVALUACIÓN COMPLETA DEL ENSEMBLE DE EMBEDDINGS")
        print("="*80)
        
        # Crear embeddings de documentos
        self.create_document_embeddings()
        
        # Evaluar modelos individuales primero
        print("\n📊 Evaluando modelos individuales...")
        for i, model_name in enumerate(self.models):
            weights = [0] * len(self.models)
            weights[i] = 1.0  # Solo este modelo
            
            results, metrics = self.evaluate_weights(weights, 'weighted_average')
            self.results['individual_models'][model_name] = {
                'metrics': metrics,
                'results': results
            }
        
        # Evaluar con pesos iniciales
        print("\n📊 Evaluando ensemble con pesos iniciales...")
        
        # Método 1: Promedio ponderado
        results_wa_initial, metrics_wa_initial = self.evaluate_weights(
            self.initial_weights, 'weighted_average'
        )
        
        # Método 2: Fusión de rankings
        results_rf_initial, metrics_rf_initial = self.evaluate_weights(
            self.initial_weights, 'rank_fusion'
        )
        
        # Buscar pesos óptimos
        print("\n🔎 Buscando pesos óptimos...")
        
        # Para promedio ponderado - usar la métrica seleccionada
        optimal_weights_wa, all_results_wa = self.find_optimal_weights(
            'weighted_average', 
            self.optimization_metric
        )
        results_wa_optimal, metrics_wa_optimal = self.evaluate_weights(
            optimal_weights_wa, 'weighted_average'
        )
        
        # Para fusión de rankings - usar la métrica seleccionada
        optimal_weights_rf, all_results_rf = self.find_optimal_weights(
            'rank_fusion',
            self.optimization_metric
        )
        results_rf_optimal, metrics_rf_optimal = self.evaluate_weights(
            optimal_weights_rf, 'rank_fusion'
        )

        # Guardar resultados
        self.results['weighted_average'] = {
            'initial': {
                'weights': self.initial_weights,
                'metrics': metrics_wa_initial,
                'results': results_wa_initial
            },
            'optimal': {
                'weights': optimal_weights_wa,
                'metrics': metrics_wa_optimal,
                'results': results_wa_optimal
            },
            'all_explorations': all_results_wa
        }
        
        self.results['rank_fusion'] = {
            'initial': {
                'weights': self.initial_weights,
                'metrics': metrics_rf_initial,
                'results': results_rf_initial
            },
            'optimal': {
                'weights': optimal_weights_rf,
                'metrics': metrics_rf_optimal,
                'results': results_rf_optimal
            },
            'all_explorations': all_results_rf
        }
        
        # Análisis de diversidad
        self.results['diversity_analysis'] = self.analyze_diversity()
        
        return self.results
    
    def print_results_report(self):
        """Imprime el reporte de resultados"""
        print("\n" + "="*80)
        print("REPORTE DE RESULTADOS - ENSEMBLE DE EMBEDDINGS")
        print("="*80)
        print(f"\n🎯 Métrica optimizada: {self.optimization_metric}")

        # Modelos individuales
        print("\n📊 RENDIMIENTO DE MODELOS INDIVIDUALES")
        print("-"*80)
        print(f"{'Modelo':<25} {'Acc@1':<10} {'Acc@5':<10} {'MRR':<10} {'MAP':<10} {'NDCG@5':<10}")
        print("-"*80)
        
        for model_name, data in self.results['individual_models'].items():
            m = data['metrics']
            print(f"{model_name:<25} {m['accuracy_at_1']:<10.1%} {m['accuracy_at_5']:<10.1%} "
                  f"{m['mrr']:<10.3f} {m['map']:<10.3f} {m['ndcg_at_5']:<10.3f}")
        
        # Método 1: Promedio Ponderado
        print("\n\n📊 MÉTODO 1: PROMEDIO PONDERADO DE EMBEDDINGS")
        print("-"*80)
        
        # Pesos iniciales
        print("\n🔸 Con pesos iniciales:")
        for i, (model_name, weight) in enumerate(zip(self.models.keys(), self.initial_weights)):
            print(f"   {model_name}: {weight:.3f}")
        
        m_initial = self.results['weighted_average']['initial']['metrics']
        print(f"\n   Métricas:")
        print(f"   • Accuracy@1: {m_initial['accuracy_at_1']:.1%}")
        print(f"   • Accuracy@5: {m_initial['accuracy_at_5']:.1%}")
        print(f"   • MRR: {m_initial['mrr']:.3f}")
        print(f"   • MAP: {m_initial['map']:.3f}")
        print(f"   • NDCG@5: {m_initial['ndcg_at_5']:.3f}")
        
        # Pesos óptimos
        print("\n🔸 Con pesos óptimos:")
        optimal_weights = self.results['weighted_average']['optimal']['weights']
        for i, (model_name, weight) in enumerate(zip(self.models.keys(), optimal_weights)):
            print(f"   {model_name}: {weight:.3f}")
        
        m_optimal = self.results['weighted_average']['optimal']['metrics']
        print(f"\n   Métricas:")
        print(f"   • Accuracy@1: {m_optimal['accuracy_at_1']:.1%} "
              f"({'+'}{(m_optimal['accuracy_at_1'] - m_initial['accuracy_at_1'])*100:.1f} pp)")
        print(f"   • Accuracy@5: {m_optimal['accuracy_at_5']:.1%} "
              f"({'+'}{(m_optimal['accuracy_at_5'] - m_initial['accuracy_at_5'])*100:.1f} pp)")
        print(f"   • MRR: {m_optimal['mrr']:.3f} "
              f"({'+'}{m_optimal['mrr'] - m_initial['mrr']:.3f})")
        print(f"   • MAP: {m_optimal['map']:.3f} "
              f"({'+'}{m_optimal['map'] - m_initial['map']:.3f})")
        print(f"   • NDCG@5: {m_optimal['ndcg_at_5']:.3f} "
              f"({'+'}{m_optimal['ndcg_at_5'] - m_initial['ndcg_at_5']:.3f})")
        
        # Método 2: Fusión de Rankings
        print("\n\n📊 MÉTODO 2: FUSIÓN DE RANKINGS (RRF PONDERADO)")
        print("-"*80)
        
        # Pesos iniciales
        print("\n🔸 Con pesos iniciales:")
        for i, (model_name, weight) in enumerate(zip(self.models.keys(), self.initial_weights)):
            print(f"   {model_name}: {weight:.3f}")
        
        m_initial = self.results['rank_fusion']['initial']['metrics']
        print(f"\n   Métricas:")
        print(f"   • Accuracy@1: {m_initial['accuracy_at_1']:.1%}")
        print(f"   • Accuracy@5: {m_initial['accuracy_at_5']:.1%}")
        print(f"   • MRR: {m_initial['mrr']:.3f}")
        print(f"   • MAP: {m_initial['map']:.3f}")
        print(f"   • NDCG@5: {m_initial['ndcg_at_5']:.3f}")
        
        # Pesos óptimos
        print("\n🔸 Con pesos óptimos:")
        optimal_weights = self.results['rank_fusion']['optimal']['weights']
        for i, (model_name, weight) in enumerate(zip(self.models.keys(), optimal_weights)):
            print(f"   {model_name}: {weight:.3f}")
        
        m_optimal = self.results['rank_fusion']['optimal']['metrics']
        print(f"\n   Métricas:")
        print(f"   • Accuracy@1: {m_optimal['accuracy_at_1']:.1%} "
              f"({'+'}{(m_optimal['accuracy_at_1'] - m_initial['accuracy_at_1'])*100:.1f} pp)")
        print(f"   • Accuracy@5: {m_optimal['accuracy_at_5']:.1%} "
              f"({'+'}{(m_optimal['accuracy_at_5'] - m_initial['accuracy_at_5'])*100:.1f} pp)")
        print(f"   • MRR: {m_optimal['mrr']:.3f} "
              f"({'+'}{m_optimal['mrr'] - m_initial['mrr']:.3f})")
        print(f"   • MAP: {m_optimal['map']:.3f} "
              f"({'+'}{m_optimal['map'] - m_initial['map']:.3f})")
        print(f"   • NDCG@5: {m_optimal['ndcg_at_5']:.3f} "
              f"({'+'}{m_optimal['ndcg_at_5'] - m_initial['ndcg_at_5']:.3f})")
        
        # Comparación de métodos
        print("\n\n📊 COMPARACIÓN DE MÉTODOS (CON PESOS ÓPTIMOS)")
        print("-"*80)
        
        wa_optimal = self.results['weighted_average']['optimal']['metrics']
        rf_optimal = self.results['rank_fusion']['optimal']['metrics']
        
        print(f"{'Método':<25} {'Acc@1':<10} {'Acc@5':<10} {'MRR':<10} {'MAP':<10} {'NDCG@5':<10}")
        print("-"*80)
        print(f"{'Promedio Ponderado':<25} {wa_optimal['accuracy_at_1']:<10.1%} "
              f"{wa_optimal['accuracy_at_5']:<10.1%} {wa_optimal['mrr']:<10.3f} "
              f"{wa_optimal['map']:<10.3f} {wa_optimal['ndcg_at_5']:<10.3f}")
        print(f"{'Fusión de Rankings':<25} {rf_optimal['accuracy_at_1']:<10.1%} "
              f"{rf_optimal['accuracy_at_5']:<10.1%} {rf_optimal['mrr']:<10.3f} "
              f"{rf_optimal['map']:<10.3f} {rf_optimal['ndcg_at_5']:<10.3f}")
        
        # Mejor modelo individual vs mejor ensemble
        best_individual = max(self.results['individual_models'].values(), 
                            key=lambda x: x['metrics']['accuracy_at_1'])
        best_ensemble = wa_optimal if wa_optimal['accuracy_at_1'] > rf_optimal['accuracy_at_1'] else rf_optimal
        
        improvement = (best_ensemble['accuracy_at_1'] - best_individual['metrics']['accuracy_at_1']) * 100
        
        print(f"\n💡 Mejora del mejor ensemble sobre mejor modelo individual: "
              f"{'+' if improvement > 0 else ''}{improvement:.1f} pp")
    
    def print_diversity_analysis(self):
        """Imprime el análisis de diversidad"""
        diversity = self.results['diversity_analysis']
        
        print("\n\n🔬 ANÁLISIS DE DIVERSIDAD DEL ENSEMBLE")
        print("="*80)
        
        # Overlap entre pares
        print("\n📊 Overlap promedio entre pares de modelos (en top-5):")
        for pair, overlap in diversity['pairwise_overlap'].items():
            print(f"   • {pair}: {overlap:.1%}")
        
        # Agreement promedio
        print(f"\n📊 Agreement promedio en predicciones: {diversity['average_agreement']:.1%}")
        print(f"   (1.0 = todos predicen lo mismo, 0.0 = todos predicen diferente)")
        
        # Contribuciones por modelo
        print("\n📊 Contribuciones únicas por modelo (veces que solo ese modelo acertó):")
        total_queries = len(diversity['query_agreement'])
        for model, count in diversity['model_contributions'].items():
            print(f"   • {model}: {count} queries ({count/total_queries*100:.1f}%)")
        
        # Victorias únicas del ensemble
        print(f"\n📊 Queries donde el ensemble acertó pero ningún modelo individual: "
              f"{diversity['ensemble_unique_wins']} ({diversity['ensemble_unique_wins']/total_queries*100:.1f}%)")
        
        # Queries con mayor desacuerdo
        print("\n📊 Top 5 queries con mayor desacuerdo entre modelos:")
        sorted_queries = sorted(diversity['query_agreement'], 
                              key=lambda x: x['agreement_score'])
        
        for i, query_data in enumerate(sorted_queries[:5], 1):
            print(f"\n   {i}. Query: {query_data['query'][:50]}...")
            print(f"      Agreement score: {query_data['agreement_score']:.2f}")
            print(f"      Predicciones únicas: {query_data['unique_predictions']}")
    
    def create_visualizations(self):
        """Crea visualizaciones de los resultados"""
        output_dir = Path("resultados_ensemble")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Comparación de métodos y modelos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Preparar datos
        model_names = list(self.results['individual_models'].keys())
        individual_acc = [self.results['individual_models'][m]['metrics']['accuracy_at_1'] 
                         for m in model_names]
        
        wa_initial = self.results['weighted_average']['initial']['metrics']['accuracy_at_1']
        wa_optimal = self.results['weighted_average']['optimal']['metrics']['accuracy_at_1']
        rf_initial = self.results['rank_fusion']['initial']['metrics']['accuracy_at_1']
        rf_optimal = self.results['rank_fusion']['optimal']['metrics']['accuracy_at_1']
        
        # Gráfico 1: Comparación general
        all_names = model_names + ['WA Initial', 'WA Optimal', 'RF Initial', 'RF Optimal']
        all_acc = individual_acc + [wa_initial, wa_optimal, rf_initial, rf_optimal]
        colors = ['#3498db'] * len(model_names) + ['#e74c3c', '#27ae60', '#f39c12', '#16a085']
        
        bars = ax1.bar(range(len(all_names)), all_acc, color=colors)
        ax1.set_xticks(range(len(all_names)))
        ax1.set_xticklabels(all_names, rotation=45, ha='right')
        ax1.set_ylabel('Accuracy@1')
        ax1.set_title('Comparación de Modelos Individuales vs Ensemble')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Añadir valores
        for bar, val in zip(bars, all_acc):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.1%}', ha='center', va='bottom')
        
        # Gráfico 2: Exploración de pesos para WA
        wa_explorations = self.results['weighted_average']['all_explorations']
        accuracies = [exp['metrics']['accuracy_at_1'] for exp in wa_explorations]
        
        ax2.scatter(range(len(accuracies)), accuracies, alpha=0.5, s=20)
        ax2.axhline(y=wa_initial, color='red', linestyle='--', 
                   label=f'Inicial: {wa_initial:.1%}')
        ax2.axhline(y=wa_optimal, color='green', linestyle='--', 
                   label=f'Óptimo: {wa_optimal:.1%}')
        ax2.set_xlabel('Combinación de pesos')
        ax2.set_ylabel('Accuracy@1')
        ax2.set_title('Exploración de Pesos - Promedio Ponderado')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{output_dir}/comparacion_ensemble_{timestamp}.png", dpi=300)
        plt.close()
        
        # 2. Heatmap de pesos óptimos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Preparar datos de pesos
        models = list(self.models.keys())
        weight_data = {
            'Initial': self.initial_weights,
            'WA Optimal': self.results['weighted_average']['optimal']['weights'],
            'RF Optimal': self.results['rank_fusion']['optimal']['weights']
        }
        
        weight_matrix = np.array([weight_data[key] for key in weight_data])
        
        # Heatmap
        sns.heatmap(weight_matrix.T, 
                   xticklabels=list(weight_data.keys()),
                   yticklabels=models,
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   ax=ax1)
        ax1.set_title('Distribución de Pesos')
        
        # Gráfico de diversidad
        diversity = self.results['diversity_analysis']
        overlap_values = list(diversity['pairwise_overlap'].values())
        overlap_labels = [pair.replace(' vs ', '\nvs\n') 
                         for pair in diversity['pairwise_overlap'].keys()]
        
        ax2.bar(range(len(overlap_values)), overlap_values, color='#3498db')
        ax2.set_xticks(range(len(overlap_values)))
        ax2.set_xticklabels(overlap_labels, rotation=0, ha='center')
        ax2.set_ylabel('Overlap en Top-5')
        ax2.set_title('Diversidad entre Modelos')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, val in enumerate(overlap_values):
            ax2.text(i, val + 0.01, f'{val:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pesos_diversidad_{timestamp}.png", dpi=300)
        plt.close()
        
        print(f"\n📊 Visualizaciones guardadas en: {output_dir}/")
    
    def save_detailed_results(self):
        """Guarda resultados detallados en archivos"""
        output_dir = Path("resultados_ensemble_cat")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Función auxiliar para convertir numpy a Python nativo
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        # Guardar configuración y resultados completos
        config_results = {
            'configuration': {
                'models': self.model_configs,
                'initial_weights': convert_to_native(self.initial_weights)
            },
            'results': {
                'individual_models': {
                    name: {
                        'metrics': convert_to_native(data['metrics'])
                    } for name, data in self.results['individual_models'].items()
                },
                'weighted_average': {
                    'initial': {
                        'weights': convert_to_native(self.results['weighted_average']['initial']['weights']),
                        'metrics': convert_to_native(self.results['weighted_average']['initial']['metrics'])
                    },
                    'optimal': {
                        'weights': convert_to_native(self.results['weighted_average']['optimal']['weights']),
                        'metrics': convert_to_native(self.results['weighted_average']['optimal']['metrics'])
                    }
                },
                'rank_fusion': {
                    'initial': {
                        'weights': convert_to_native(self.results['rank_fusion']['initial']['weights']),
                        'metrics': convert_to_native(self.results['rank_fusion']['initial']['metrics'])
                    },
                    'optimal': {
                        'weights': convert_to_native(self.results['rank_fusion']['optimal']['weights']),
                        'metrics': convert_to_native(self.results['rank_fusion']['optimal']['metrics'])
                    }
                },
                'diversity_analysis': {
                    'pairwise_overlap': convert_to_native(self.results['diversity_analysis']['pairwise_overlap']),
                    'average_agreement': convert_to_native(self.results['diversity_analysis']['average_agreement']),
                    'ensemble_unique_wins': convert_to_native(self.results['diversity_analysis']['ensemble_unique_wins'])
                }
            }
        }
        
        # Guardar JSON
        with open(f"{output_dir}/ensemble_results_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(config_results, f, indent=2, ensure_ascii=False)
        
        # Guardar reporte en texto
        with open(f"{output_dir}/ensemble_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            
            self.print_results_report()
            self.print_diversity_analysis()
            
            sys.stdout = old_stdout
        
        print(f"\n📁 Resultados detallados guardados en: {output_dir}/")


def main():
    """Función principal"""
    print("🚀 Sistema de Ensemble de Embeddings")
    print("="*60)
    
    # Crear sistema
    ensemble_system = EnsembleEmbeddingSystem()
    
    # Configurar ensemble
    ensemble_system.configure_ensemble()

    # Seleccionar métrica de optimización
    ensemble_system.select_optimization_metric()
    
    # Ejecutar evaluación completa
    results = ensemble_system.run_complete_evaluation()
    
    # Mostrar resultados
    ensemble_system.print_results_report()
    ensemble_system.print_diversity_analysis()
    
    # Crear visualizaciones
    ensemble_system.create_visualizations()
    
    # Guardar resultados
    ensemble_system.save_detailed_results()
    
    print("\n✅ Evaluación del ensemble completada!")


if __name__ == "__main__":
    main()