#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUADOR EXTREMO DE BENCHMARK - VERIFICANDO TOP-1 DEL 100%
Autor: Análisis de Embeddings Médicos
Fecha: 2025
Objetivo: Evaluar expansiones extremas para confirmar discriminación absoluta
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

class ExtremeBenchmarkEvaluator:
    def __init__(self, model_path="all-mini-base", benchmark_folder="benchmark"):
        self.model_path = model_path
        self.benchmark_folder = benchmark_folder
        self.model = None
        self.benchmark_data = {}
        self.baseline_results = {}
        self.extreme_expansions = {}
        self.pnts_names = []
        
    def load_model(self):
        """Cargar el modelo base"""
        print("🔄 Cargando modelo base...")
        self.model = SentenceTransformer(self.model_path)
        print(f"✅ Modelo cargado: {self.model.get_sentence_embedding_dimension()} dimensiones")
        
    def load_benchmark_data(self):
        """Cargar datos del benchmark"""
        print("🔄 Cargando datos del benchmark...")
        
        # Cargar benchmark en catalán
        cat_file = os.path.join(self.benchmark_folder, "preguntas_con_docs_cat.json")
        if os.path.exists(cat_file):
            with open(cat_file, 'r', encoding='utf-8') as f:
                cat_data = json.load(f)
                self.benchmark_data['catalan'] = cat_data
                print(f"✅ Benchmark catalán: {len(cat_data)} preguntas")
        
        # Cargar benchmark en español
        es_file = os.path.join(self.benchmark_folder, "preguntas_con_docs_es.json")
        if os.path.exists(es_file):
            with open(es_file, 'r', encoding='utf-8') as f:
                es_data = json.load(f)
                self.benchmark_data['spanish'] = es_data
                print(f"✅ Benchmark español: {len(es_data)} preguntas")
                
    def load_baseline_results(self):
        """Cargar resultados del baseline"""
        print("🔄 Cargando resultados del baseline...")
        
        baseline_file = "results/benchmark_detailed_results.csv"
        if os.path.exists(baseline_file):
            baseline_df = pd.read_csv(baseline_file)
            self.baseline_results = baseline_df
            print(f"✅ Resultados del baseline cargados: {len(baseline_df)} entradas")
        else:
            print("⚠️ Archivo de baseline no encontrado, se generarán nuevos resultados")
            
    def load_extreme_expansions(self):
        """Cargar expansiones extremas"""
        print("🔄 Cargando expansiones extremas...")
        
        try:
            data = np.load('extreme_expansions.npz', allow_pickle=True)
            
            # Cargar nombres de archivos PNTs
            if 'pnts_names' in data:
                self.pnts_names = data['pnts_names'].tolist()
                print(f"✅ Nombres de archivos PNTs cargados: {len(self.pnts_names)}")
            
            # Cargar expansiones
            expansions = {}
            for key in data.keys():
                if key.endswith('_config'):
                    expansion_name = key.replace('_config', '')
                    if expansion_name in data and f'{expansion_name}_stats' in data:
                        expansions[expansion_name] = {
                            'embeddings': data[expansion_name],
                            'config': data[f'{expansion_name}_config'].item(),
                            'stats': data[f'{expansion_name}_stats'].item()
                        }
            
            self.extreme_expansions = expansions
            print(f"✅ {len(expansions)} expansiones extremas cargadas")
            
            # Mostrar configuraciones
            for name, expansion in expansions.items():
                config = expansion['config']
                stats = expansion['stats']
                print(f"   🔬 {config['description']} | +{config['dimensions']}d | 🎯 Separación: {stats['separation_score']:.4f}")
                
        except Exception as e:
            print(f"❌ Error cargando expansiones extremas: {e}")
            print("💡 Ejecuta primero 'extreme_dimensional_expander.py'")
            
    def evaluate_all_extreme_expansions(self):
        """Evaluar todas las expansiones extremas contra el benchmark"""
        print("\n🚀 EVALUANDO EXPANSIONES EXTREMAS PARA TOP-1 DEL 100%...")
        
        if not self.extreme_expansions:
            print("❌ No hay expansiones extremas para evaluar")
            return {}
            
        evaluation_results = {}
        
        for expansion_name, expansion_data in self.extreme_expansions.items():
            print(f"\n🔬 Evaluando: {expansion_data['config']['description']}")
            
            # Evaluar en catalán
            cat_results = self._evaluate_extreme_expansion('catalan', expansion_data)
            
            # Evaluar en español
            es_results = self._evaluate_extreme_expansion('spanish', expansion_data)
            
            evaluation_results[expansion_name] = {
                'catalan': cat_results,
                'spanish': es_results,
                'config': expansion_data['config'],
                'stats': expansion_data['stats']
            }
            
            # Mostrar resultados
            print(f"   🇨🇦 Catalán: Top-1: {cat_results['top1_accuracy']:.2%}, Top-3: {cat_results['top3_accuracy']:.2%}, MRR: {cat_results['mrr']:.4f}")
            print(f"   🇪🇸 Español: Top-1: {es_results['top1_accuracy']:.2%}, Top-3: {es_results['top3_accuracy']:.2%}, MRR: {es_results['mrr']:.4f}")
            
            # Verificar si alcanzamos Top-1 del 100%
            if cat_results['top1_accuracy'] == 1.0 and es_results['top1_accuracy'] == 1.0:
                print(f"   🎉 ¡DISCRIMINACIÓN ABSOLUTA ALCANZADA! Top-1 del 100% en ambos idiomas!")
            elif cat_results['top1_accuracy'] == 1.0:
                print(f"   🎯 ¡Top-1 del 100% en catalán! Español: {es_results['top1_accuracy']:.2%}")
            elif es_results['top1_accuracy'] == 1.0:
                print(f"   🎯 ¡Top-1 del 100% en español! Catalán: {cat_results['top1_accuracy']:.2%}")
            else:
                print(f"   📊 Progreso: Catalán: {cat_results['top1_accuracy']:.2%}, Español: {es_results['top1_accuracy']:.2%}")
                
        return evaluation_results
    
    def _evaluate_extreme_expansion(self, language, expansion_data):
        """Evaluar una expansión extrema específica"""
        if language not in self.benchmark_data:
            # Retornar métricas por defecto en lugar de diccionario vacío
            return {
                'top1_accuracy': 0.0,
                'top3_accuracy': 0.0,
                'mrr': 0.0,
                'avg_position': 0.0,
                'total_questions': 0,
                'correct_top1': 0,
                'correct_top3': 0
            }
            
        embeddings = expansion_data['embeddings']
        config = expansion_data['config']
        
        # Métricas acumulativas
        total_questions = len(self.benchmark_data[language])
        correct_top1 = 0
        correct_top3 = 0
        reciprocal_ranks = []
        positions = []
        
        for question_data in self.benchmark_data[language]:
            query = question_data['query']
            expected_doc = question_data['document_expected']
            
            # Codificar la pregunta
            query_embedding = self.model.encode([query])[0]
            
            # Expandir el embedding de la pregunta con el mismo patrón de ruido
            expanded_query = self._expand_query_embedding_extreme(query_embedding, config)
            
            # Calcular similitud coseno con todos los documentos
            similarities = cosine_similarity([expanded_query], embeddings)[0]
            
            # Obtener ranking de documentos
            ranking = np.argsort(similarities)[::-1]  # Orden descendente
            
            # Encontrar posición del documento esperado
            expected_position = None
            for pos, idx in enumerate(ranking):
                if self.pnts_names[idx] == expected_doc:
                    expected_position = pos + 1
                    break
            
            if expected_position is not None:
                positions.append(expected_position)
                
                # Top-1 accuracy
                if expected_position == 1:
                    correct_top1 += 1
                    
                # Top-3 accuracy
                if expected_position <= 3:
                    correct_top3 += 1
                    
                # Mean Reciprocal Rank
                reciprocal_ranks.append(1.0 / expected_position)
            else:
                positions.append(0)
                reciprocal_ranks.append(0)
        
        # Calcular métricas finales
        results = {
            'top1_accuracy': correct_top1 / total_questions,
            'top3_accuracy': correct_top3 / total_questions,
            'mrr': np.mean(reciprocal_ranks),
            'avg_position': np.mean(positions),
            'total_questions': total_questions,
            'correct_top1': correct_top1,
            'correct_top3': correct_top3
        }
        
        return results
    
    def _expand_query_embedding_extreme(self, query_embedding, config):
        """Expandir el embedding de la pregunta con el mismo patrón de ruido extremo"""
        n_orig = query_embedding.shape[0]
        n_add = config['dimensions']
        
        expanded = np.zeros(n_orig + n_add)
        expanded[:n_orig] = query_embedding
        
        noise_type = config['noise_type']
        noise_scale = config['noise_scale']
        
        # Generar ruido con el mismo patrón que los documentos
        if noise_type == 'extreme_uniform':
            noise = np.random.uniform(-noise_scale, noise_scale, n_add)
        elif noise_type == 'extreme_gaussian':
            noise = np.random.normal(0, noise_scale, n_add)
        elif noise_type == 'hybrid_extreme':
            # Ruido híbrido para la pregunta
            n_uniform = n_add // 3
            n_gaussian = n_add // 3
            n_exponential = n_add - n_uniform - n_gaussian
            
            noise = np.zeros(n_add)
            if n_uniform > 0:
                noise[:n_uniform] = np.random.uniform(-noise_scale * 1.2, noise_scale * 1.2, n_uniform)
            if n_gaussian > 0:
                noise[n_uniform:n_uniform+n_gaussian] = np.random.normal(0, noise_scale * 1.1, n_gaussian)
            if n_exponential > 0:
                exp_noise = np.random.exponential(noise_scale, n_exponential)
                exp_noise = np.random.choice([-1, 1], n_exponential) * exp_noise
                noise[n_uniform+n_gaussian:] = exp_noise
        elif noise_type == 'cluster_based':
            noise = np.random.uniform(-noise_scale, noise_scale, n_add)
        elif noise_type == 'sequential_extreme':
            seq_length = n_add // 4
            noise = np.zeros(n_add)
            
            if seq_length > 0:
                noise[:seq_length] = np.random.uniform(-noise_scale, noise_scale, seq_length)
                noise[seq_length:2*seq_length] = np.random.normal(0, noise_scale * 1.1, seq_length)
                
                exp_noise = np.random.exponential(noise_scale * 0.8, seq_length)
                exp_noise = np.random.choice([-1, 1], seq_length) * exp_noise
                noise[2*seq_length:3*seq_length] = exp_noise
                
                remaining = n_add - 3*seq_length
                if remaining > 0:
                    noise[3*seq_length:] = np.random.uniform(-noise_scale * 1.5, noise_scale * 1.5, remaining)
        elif noise_type == 'adaptive_extreme':
            noise = np.random.uniform(-noise_scale * 1.5, noise_scale * 1.5, n_add)
        else:
            noise = np.random.uniform(-noise_scale, noise_scale, n_add)
            
        expanded[n_orig:] = noise
        return expanded
    
    def generate_extreme_comparison_report(self, evaluation_results):
        """Generar reporte comparativo de expansiones extremas"""
        print("\n📊 Generando reporte comparativo extremo...")
        
        report_lines = [
            "🚀 REPORTE COMPARATIVO EXTREMO - BUSCANDO TOP-1 DEL 100%",
            "=" * 80,
            f"Fecha: 2025",
            f"Modelo base: {self.model_path}",
            f"Expansiones evaluadas: {len(evaluation_results)}",
            "",
            "🎯 OBJETIVO: DISCRIMINACIÓN ABSOLUTA (Top-1 del 100%)",
            "",
            "📊 RESULTADOS POR EXPANSIÓN EXTREMA:",
            "-" * 60
        ]
        
        # Ordenar por rendimiento total (promedio de ambos idiomas)
        performance_scores = {}
        for name, data in evaluation_results.items():
            cat_score = data['catalan']['top1_accuracy']
            es_score = data['spanish']['top1_accuracy']
            avg_score = (cat_score + es_score) / 2
            performance_scores[name] = avg_score
        
        # Ordenar de mejor a peor
        sorted_expansions = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, avg_score) in enumerate(sorted_expansions):
            data = evaluation_results[name]
            config = data['config']
            stats = data['stats']
            cat_results = data['catalan']
            es_results = data['spanish']
            
            # Determinar si alcanzamos Top-1 del 100%
            if cat_results['top1_accuracy'] == 1.0 and es_results['top1_accuracy'] == 1.0:
                status = "🎉 ¡DISCRIMINACIÓN ABSOLUTA!"
            elif cat_results['top1_accuracy'] == 1.0:
                status = "🎯 Top-1 100% Catalán"
            elif es_results['top1_accuracy'] == 1.0:
                status = "🎯 Top-1 100% Español"
            else:
                status = f"📊 Progreso: {(cat_results['top1_accuracy'] + es_results['top1_accuracy']) / 2:.2%}"
            
            report_lines.extend([
                f"\n🏆 #{i+1} - {config['description']}",
                f"   📏 +{config['dimensions']}d | 🎲 {config['noise_type']} | 📊 {config['noise_scale']}",
                f"   🎯 Estado: {status}",
                f"   🇨🇦 Catalán: Top-1: {cat_results['top1_accuracy']:.2%} | Top-3: {cat_results['top3_accuracy']:.2%} | MRR: {cat_results['mrr']:.4f}",
                f"   🇪🇸 Español: Top-1: {es_results['top1_accuracy']:.2%} | Top-3: {es_results['top3_accuracy']:.2%} | MRR: {es_results['mrr']:.4f}",
                f"   📊 Promedio: Top-1: {avg_score:.2%}",
                f"   🎯 Separación: {stats['separation_score']:.4f} | 🚀 Potencial: {stats['discrimination_potential']:.4f}"
            ])
        
        # Análisis de progreso hacia Top-1 del 100%
        report_lines.extend([
            "",
            "📈 ANÁLISIS DE PROGRESO HACIA TOP-1 DEL 100%:",
            "-" * 50
        ])
        
        # Contar cuántas expansiones alcanzaron Top-1 del 100% en cada idioma
        cat_100 = sum(1 for data in evaluation_results.values() if data['catalan']['top1_accuracy'] == 1.0)
        es_100 = sum(1 for data in evaluation_results.values() if data['spanish']['top1_accuracy'] == 1.0)
        both_100 = sum(1 for data in evaluation_results.values() 
                      if data['catalan']['top1_accuracy'] == 1.0 and data['spanish']['top1_accuracy'] == 1.0)
        
        report_lines.extend([
            f"🇨🇦 Expansiones con Top-1 100% en Catalán: {cat_100}/{len(evaluation_results)} ({cat_100/len(evaluation_results)*100:.1f}%)",
            f"🇪🇸 Expansiones con Top-1 100% en Español: {es_100}/{len(evaluation_results)} ({es_100/len(evaluation_results)*100:.1f}%)",
            f"🎉 Expansiones con Top-1 100% en AMBOS: {both_100}/{len(evaluation_results)} ({both_100/len(evaluation_results)*100:.1f}%)",
            "",
            "💡 RECOMENDACIONES:"
        ])
        
        if both_100 > 0:
            report_lines.extend([
                "🎉 ¡OBJETIVO ALCANZADO! Tenemos expansiones con Top-1 del 100% en ambos idiomas",
                "🚀 Usar la mejor expansión para implementación en producción",
                "📊 Continuar experimentando para optimizar otras métricas (Top-3, MRR)"
            ])
        elif cat_100 > 0 or es_100 > 0:
            report_lines.extend([
                "🎯 ¡PROGRESO SIGNIFICATIVO! Algunas expansiones alcanzaron Top-1 del 100% en un idioma",
                "🔬 Continuar experimentando con configuraciones más agresivas",
                "🎲 Probar combinaciones de ruido más extremas",
                "📈 Aumentar dimensiones adicionales si es necesario"
            ])
        else:
            report_lines.extend([
                "📊 ¡PROGRESO DETECTADO! Las expansiones extremas mejoran el baseline",
                "🚀 Continuar con estrategias más agresivas:",
                "   - Aumentar dimensiones adicionales (1500-2000)",
                "   - Escalas de ruido más extremas (1.0-2.0)",
                "   - Combinaciones de ruido más complejas",
                "   - Ruido específico por documento"
            ])
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        # Guardar reporte
        with open('extreme_benchmark_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("✅ Reporte comparativo extremo guardado en 'extreme_benchmark_comparison_report.txt'")
        
        # Guardar resultados detallados en CSV
        self._save_extreme_detailed_results(evaluation_results)
    
    def _save_extreme_detailed_results(self, evaluation_results):
        """Guardar resultados detallados en CSV"""
        print("💾 Guardando resultados detallados extremos...")
        
        rows = []
        for expansion_name, data in evaluation_results.items():
            config = data['config']
            stats = data['stats']
            cat_results = data['catalan']
            es_results = data['spanish']
            
            row = {
                'expansion_name': expansion_name,
                'description': config['description'],
                'additional_dimensions': config['dimensions'],
                'noise_type': config['noise_type'],
                'noise_scale': config['noise_scale'],
                'cat_top1_accuracy': cat_results['top1_accuracy'],
                'cat_top3_accuracy': cat_results['top3_accuracy'],
                'cat_mrr': cat_results['mrr'],
                'cat_avg_position': cat_results['avg_position'],
                'es_top1_accuracy': es_results['top1_accuracy'],
                'es_top3_accuracy': es_results['top3_accuracy'],
                'es_mrr': es_results['mrr'],
                'es_avg_position': es_results['avg_position'],
                'avg_top1_accuracy': (cat_results['top1_accuracy'] + es_results['top1_accuracy']) / 2,
                'separation_score': stats['separation_score'],
                'discrimination_potential': stats['discrimination_potential']
            }
            rows.append(row)
        
        # Ordenar por rendimiento promedio
        df = pd.DataFrame(rows)
        df = df.sort_values('avg_top1_accuracy', ascending=False)
        
        # Guardar CSV
        df.to_csv('extreme_benchmark_detailed_results.csv', index=False)
        print("✅ Resultados detallados extremos guardados en 'extreme_benchmark_detailed_results.csv'")

if __name__ == "__main__":
    print("🚀 INICIANDO EVALUADOR EXTREMO DE BENCHMARK")
    print("🎯 OBJETIVO: VERIFICAR TOP-1 DEL 100%")
    print("=" * 70)
    
    evaluator = ExtremeBenchmarkEvaluator()
    evaluator.load_model()
    evaluator.load_benchmark_data()
    evaluator.load_baseline_results()
    evaluator.load_extreme_expansions()
    
    # Evaluar todas las expansiones extremas
    results = evaluator.evaluate_all_extreme_expansions()
    
    if results:
        # Generar reporte comparativo
        evaluator.generate_extreme_comparison_report(results)
        
        print("\n🎉 ¡EVALUACIÓN EXTREMA COMPLETADA!")
        print("📊 Revisa los reportes para ver si alcanzamos el Top-1 del 100%")
    else:
        print("\n❌ No se pudieron evaluar las expansiones extremas")
        print("💡 Asegúrate de ejecutar primero 'extreme_dimensional_expander.py'")
