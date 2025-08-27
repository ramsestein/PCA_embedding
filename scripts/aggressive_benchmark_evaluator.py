#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador de Benchmark para Expansiones Agresivas
Autor: Análisis de Embeddings Médicos
Fecha: 2025
"""

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
warnings.filterwarnings('ignore')

class AggressiveBenchmarkEvaluator:
    """Evaluador que prueba expansiones agresivas con el benchmark"""
    
    def __init__(self, model_path="all-mini-base", benchmark_folder="benchmark"):
        self.model_path = model_path
        self.benchmark_folder = benchmark_folder
        self.model = None
        self.benchmark_data = {}
        self.baseline_metrics = None
        self.expansion_results = {}
        
    def load_model(self):
        """Carga el modelo de embeddings"""
        print("🔄 Cargando modelo de embeddings...")
        try:
            self.model = SentenceTransformer(self.model_path)
            print(f"✅ Modelo cargado: {self.model.get_sentence_embedding_dimension()} dimensiones")
            return True
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            return False
    
    def load_benchmark_data(self):
        """Carga los datos del benchmark en ambos idiomas"""
        print(f"📊 Cargando datos del benchmark desde: {self.benchmark_folder}")
        
        benchmark_files = {
            'catalan': 'preguntas_con_docs_cat.json',
            'spanish': 'preguntas_con_docs_es.json'
        }
        
        for language, filename in benchmark_files.items():
            file_path = os.path.join(self.benchmark_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.benchmark_data[language] = json.load(f)
                print(f"   ✅ {language.capitalize()}: {len(self.benchmark_data[language])} preguntas")
            except Exception as e:
                print(f"   ❌ Error al cargar {filename}: {e}")
                return False
        
        return True
    
    def load_baseline_results(self):
        """Carga los resultados del baseline para comparación"""
        print("📁 Cargando resultados del baseline...")
        
        if not os.path.exists('benchmark_detailed_results.csv'):
            print("❌ Error: No se encontró el archivo de resultados del baseline")
            print("💡 Ejecuta primero: python benchmark_evaluator.py")
            return False
        
        try:
            df = pd.read_csv('benchmark_detailed_results.csv', encoding='utf-8')
            
            # Separar resultados por idioma
            cat_baseline = df[df['Idioma'] == 'Catalán']
            es_baseline = df[df['Idioma'] == 'Español']
            
            # Calcular métricas del baseline
            self.baseline_metrics = {
                'catalan': self._calculate_metrics_from_csv(cat_baseline),
                'spanish': self._calculate_metrics_from_csv(es_baseline)
            }
            
            print(f"✅ Baseline cargado: Catalán ({len(cat_baseline)} preguntas), Español ({len(es_baseline)} preguntas)")
            return True
            
        except Exception as e:
            print(f"❌ Error al cargar resultados del baseline: {e}")
            return False
    
    def _calculate_metrics_from_csv(self, df):
        """Calcula métricas desde DataFrame CSV"""
        if df.empty:
            return None
        
        total_questions = len(df)
        
        # Métricas básicas
        top1_accuracy = sum(1 for _, row in df.iterrows() if row['Top1_Correcto']) / total_questions
        top3_accuracy = sum(1 for _, row in df.iterrows() if row['Top3_Correcto']) / total_questions
        mrr_score = np.mean([row['MRR'] for _, row in df.iterrows()])
        
        # Análisis de posiciones
        positions = [row['Posicion_Encontrada'] for _, row in df.iterrows() if pd.notna(row['Posicion_Encontrada'])]
        avg_position = np.mean(positions) if positions else 0
        
        return {
            'total_questions': total_questions,
            'top1_accuracy': top1_accuracy,
            'top3_accuracy': top3_accuracy,
            'mrr_score': mrr_score,
            'average_position': avg_position
        }
    
    def load_aggressive_expansions(self):
        """Carga las expansiones agresivas generadas"""
        print("📁 Cargando expansiones agresivas...")
        
        if not os.path.exists('aggressive_expansions.npz'):
            print("❌ Error: No se encontraron las expansiones agresivas")
            print("💡 Ejecuta primero: python aggressive_dimensional_expander.py")
            return False
        
        try:
            data = np.load('aggressive_expansions.npz', allow_pickle=True)
            
            # Cargar nombres de archivos PNTs
            if 'pnts_names' in data:
                self.pnts_names = data['pnts_names'].tolist()
                print(f"✅ Nombres de archivos PNTs cargados: {len(self.pnts_names)}")
            
            # Cargar expansiones
            expansions = {}
            for key in data.keys():
                if key.startswith('expansion_') and key.endswith('_embeddings'):
                    exp_id = key.replace('_embeddings', '')
                    config_key = f'{exp_id}_config'
                    stats_key = f'{exp_id}_stats'
                    
                    if config_key in data and stats_key in data:
                        expansions[exp_id] = {
                            'embeddings': data[key],
                            'config': data[config_key].item(),
                            'stats': data[stats_key].item()
                        }
            
            self.expansion_results = expansions
            print(f"✅ Expansiones cargadas: {len(expansions)}")
            
            # Mostrar información de cada expansión
            for exp_id, exp_data in expansions.items():
                config = exp_data['config']
                stats = exp_data['stats']
                print(f"   📊 {config['description']}: {exp_data['embeddings'].shape}, Ratio: {stats['variance_ratio']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error al cargar expansiones agresivas: {e}")
            return False
    
    def evaluate_all_expansions(self):
        """Evalúa todas las expansiones agresivas con el benchmark"""
        print("\n🚀 EVALUANDO TODAS LAS EXPANSIONES AGRESIVAS")
        print("=" * 60)
        
        results = {}
        
        for exp_id, exp_data in self.expansion_results.items():
            print(f"\n🔬 EVALUANDO: {exp_data['config']['description']}")
            print("-" * 50)
            
            # Evaluar en ambos idiomas
            exp_results = {}
            for language in ['catalan', 'spanish']:
                print(f"   🔍 Evaluando en {language.capitalize()}...")
                exp_results[language] = self._evaluate_expansion(
                    language, 
                    exp_data['embeddings'], 
                    exp_data['config']
                )
            
            # Calcular métricas
            metrics = {}
            for language in ['catalan', 'spanish']:
                metrics[language] = self._calculate_metrics_from_results(exp_results[language])
            
            results[exp_id] = {
                'config': exp_data['config'],
                'stats': exp_data['stats'],
                'results': exp_results,
                'metrics': metrics
            }
            
            print(f"   ✅ Evaluación completada")
        
        return results
    
    def _evaluate_expansion(self, language, embeddings, expansion_config):
        """Evalúa una expansión específica en un idioma"""
        if language not in self.benchmark_data:
            return None
        
        questions = self.benchmark_data[language]
        results = []
        
        for i, qa_pair in enumerate(questions):
            query = qa_pair['query']
            expected_doc = qa_pair['document_expected']
            
            # Generar embedding para la pregunta (solo las dimensiones originales)
            query_embedding = self.model.encode([query])
            
            # Expandir la pregunta con las mismas dimensiones adicionales
            expanded_query = self._expand_query_embedding(
                query_embedding[0], 
                expansion_config
            )
            
            # Calcular similitud con todos los documentos PNTs expandidos
            similarities = cosine_similarity([expanded_query], embeddings)[0]
            
            # Obtener ranking de documentos
            ranking = np.argsort(similarities)[::-1]  # Orden descendente
            
            # Encontrar posición del documento esperado
            expected_position = None
            for pos, idx in enumerate(ranking):
                if self.pnts_names[idx] == expected_doc:  # Usar nombres reales de archivos
                    expected_position = pos + 1
                    break
            
            # Calcular métricas
            top1_correct = expected_position == 1
            top3_correct = expected_position <= 3 if expected_position else False
            mrr = 1.0 / expected_position if expected_position else 0.0
            
            result = {
                'question_id': i + 1,
                'query': query,
                'expected_document': expected_doc,
                'expected_position': expected_position,
                'top1_correct': top1_correct,
                'top3_correct': top3_correct,
                'mrr': mrr,
                'max_similarity': similarities[ranking[0]]
            }
            
            results.append(result)
            
            # Mostrar progreso cada 10 preguntas
            if (i + 1) % 10 == 0:
                print(f"      Procesadas {i + 1}/{len(questions)} preguntas...")
        
        return results
    
    def _expand_query_embedding(self, query_embedding, expansion_config):
        """Expande el embedding de la pregunta con las mismas dimensiones adicionales"""
        n_orig = 384  # Dimensiones originales del modelo
        n_add = expansion_config['dimensions']
        
        # Crear embedding expandido
        expanded_query = np.zeros(n_orig + n_add)
        
        # Copiar dimensiones originales
        expanded_query[:n_orig] = query_embedding
        
        # Generar dimensiones adicionales con el mismo patrón de ruido
        original_std = np.std(query_embedding)
        original_mean = np.mean(query_embedding)
        scaled_noise_std = original_std * expansion_config['noise_scale']
        
        if expansion_config['noise_type'] == 'mixed':
            # Combinar diferentes tipos de ruido
            half = n_add // 2
            expanded_query[n_orig:n_orig+half] = np.random.uniform(
                low=original_mean - scaled_noise_std,
                high=original_mean + scaled_noise_std,
                size=half
            )
            expanded_query[n_orig+half:] = np.random.normal(
                loc=original_mean, 
                scale=scaled_noise_std, 
                size=n_add - half
            )
        elif expansion_config['noise_type'] == 'uniform':
            expanded_query[n_orig:] = np.random.uniform(
                low=original_mean - scaled_noise_std,
                high=original_mean + scaled_noise_std,
                size=n_add
            )
        else:  # gaussian
            expanded_query[n_orig:] = np.random.normal(
                loc=original_mean, 
                scale=scaled_noise_std, 
                size=n_add
            )
        
        return expanded_query
    
    def _calculate_metrics_from_results(self, results):
        """Calcula métricas desde resultados de evaluación"""
        if not results:
            return None
        
        total_questions = len(results)
        
        # Métricas básicas
        top1_accuracy = sum(1 for r in results if r['top1_correct']) / total_questions
        top3_accuracy = sum(1 for r in results if r['top3_correct']) / total_questions
        mrr_score = np.mean([r['mrr'] for r in results])
        
        # Análisis de posiciones
        positions = [r['expected_position'] for r in results if r['expected_position']]
        avg_position = np.mean(positions) if positions else 0
        
        return {
            'total_questions': total_questions,
            'top1_accuracy': top1_accuracy,
            'top3_accuracy': top3_accuracy,
            'mrr_score': mrr_score,
            'average_position': avg_position
        }
    
    def generate_aggressive_comparison_report(self, evaluation_results):
        """Genera reporte comparativo de todas las expansiones agresivas"""
        print("\n📊 Generando reporte comparativo agresivo...")
        
        with open('aggressive_benchmark_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE COMPARATIVO: EXPANSIONES AGRESIVAS vs BASELINE\n")
            f.write("OBJETIVO: Hiper-especialización para benchmark PNTs\n")
            f.write("=" * 80 + "\n\n")
            
            # Información del baseline
            f.write("BASELINE (384 dimensiones):\n")
            f.write("-" * 30 + "\n")
            f.write("Catalán - Top-1: {:.2f}%, Top-3: {:.2f}%, MRR: {:.4f}, Pos: {:.2f}\n".format(
                self.baseline_metrics['catalan']['top1_accuracy']*100,
                self.baseline_metrics['catalan']['top3_accuracy']*100,
                self.baseline_metrics['catalan']['mrr_score'],
                self.baseline_metrics['catalan']['average_position']
            ))
            f.write("Español  - Top-1: {:.2f}%, Top-3: {:.2f}%, MRR: {:.4f}, Pos: {:.2f}\n\n".format(
                self.baseline_metrics['spanish']['top1_accuracy']*100,
                self.baseline_metrics['spanish']['top3_accuracy']*100,
                self.baseline_metrics['spanish']['mrr_score'],
                self.baseline_metrics['spanish']['average_position']
            ))
            
            # Comparación de expansiones agresivas
            f.write("COMPARACIÓN DE EXPANSIONES AGRESIVAS:\n")
            f.write("-" * 50 + "\n")
            f.write("Expansión                        | Dims | Ratio | Top-1 Cat | Top-3 Cat | MRR Cat | Pos Cat | Top-1 Esp | Top-3 Esp | MRR Esp | Pos Esp | Mejora\n")
            f.write("-" * 140 + "\n")
            
            best_expansion = None
            best_improvement = -1
            
            for exp_id, result in evaluation_results.items():
                config = result['config']
                stats = result['stats']
                metrics_cat = result['metrics']['catalan']
                metrics_es = result['metrics']['spanish']
                
                # Calcular mejoras vs baseline
                total_improvements = 0
                if metrics_cat['top1_accuracy'] > self.baseline_metrics['catalan']['top1_accuracy']: total_improvements += 1
                if metrics_cat['top3_accuracy'] > self.baseline_metrics['catalan']['top3_accuracy']: total_improvements += 1
                if metrics_cat['mrr_score'] > self.baseline_metrics['catalan']['mrr_score']: total_improvements += 1
                if metrics_cat['average_position'] < self.baseline_metrics['catalan']['average_position']: total_improvements += 1
                if metrics_es['top1_accuracy'] > self.baseline_metrics['spanish']['top1_accuracy']: total_improvements += 1
                if metrics_es['top3_accuracy'] > self.baseline_metrics['spanish']['top3_accuracy']: total_improvements += 1
                if metrics_es['mrr_score'] > self.baseline_metrics['spanish']['mrr_score']: total_improvements += 1
                if metrics_es['average_position'] < self.baseline_metrics['spanish']['average_position']: total_improvements += 1
                
                improvement_percentage = (total_improvements / 8) * 100
                
                f.write(f"{config['description']:<35} | {config['dimensions']:4d} | {stats['variance_ratio']:5.3f} | {metrics_cat['top1_accuracy']*100:7.2f}% | {metrics_cat['top3_accuracy']*100:7.2f}% | {metrics_cat['mrr_score']:6.4f} | {metrics_cat['average_position']:6.2f} | {metrics_es['top1_accuracy']*100:8.2f}% | {metrics_es['top3_accuracy']*100:8.2f}% | {metrics_es['mrr_score']:7.4f} | {metrics_es['average_position']:7.2f} | {improvement_percentage:5.1f}%\n")
                
                if improvement_percentage > best_improvement:
                    best_improvement = improvement_percentage
                    best_expansion = result
            
            f.write("\n")
            
            # Análisis de la mejor expansión
            if best_expansion:
                f.write("🏆 MEJOR EXPANSIÓN AGRESIVA:\n")
                f.write("-" * 35 + "\n")
                f.write(f"Configuración: {best_expansion['config']['description']}\n")
                f.write(f"Dimensiones adicionales: {best_expansion['config']['dimensions']}\n")
                f.write(f"Tipo de ruido: {best_expansion['config']['noise_type']}\n")
                f.write(f"Escala de ruido: {best_expansion['config']['noise_scale']}\n")
                f.write(f"Ratio de varianza: {best_expansion['stats']['variance_ratio']:.4f}\n")
                f.write(f"Mejora total: {best_improvement:.1f}% de las métricas\n\n")
                
                f.write("💡 Esta expansión produce la máxima diferenciación de documentos\n")
                f.write("   para tu benchmark específico de PNTs.\n")
            
            # Guardar CSV con resultados detallados
            self._save_aggressive_detailed_results(evaluation_results)
        
        print("✅ Reporte comparativo agresivo generado:")
        print("   - aggressive_benchmark_comparison_report.txt")
        print("   - aggressive_benchmark_detailed_results.csv")
        
        return True
    
    def _save_aggressive_detailed_results(self, evaluation_results):
        """Guarda resultados detallados de expansiones agresivas en CSV"""
        rows = []
        
        # Agregar baseline
        for language in ['catalan', 'spanish']:
            lang_name = 'Catalán' if language == 'catalan' else 'Español'
            metrics = self.baseline_metrics[language]
            
            rows.append({
                'Configuracion': 'Baseline (384d)',
                'Tipo_Ruido': 'N/A',
                'Escala_Ruido': 'N/A',
                'Dimensiones_Adicionales': 0,
                'Ratio_Varianza': 1.0,
                'Idioma': lang_name,
                'Top1_Accuracy': metrics['top1_accuracy'],
                'Top3_Accuracy': metrics['top3_accuracy'],
                'MRR_Score': metrics['mrr_score'],
                'Posicion_Promedio': metrics['average_position'],
                'Total_Preguntas': metrics['total_questions']
            })
        
        # Agregar expansiones agresivas
        for exp_id, result in evaluation_results.items():
            config = result['config']
            stats = result['stats']
            
            for language in ['catalan', 'spanish']:
                lang_name = 'Catalán' if language == 'catalan' else 'Español'
                metrics = result['metrics'][language]
                
                rows.append({
                    'Configuracion': config['description'],
                    'Tipo_Ruido': config['noise_type'],
                    'Escala_Ruido': config['noise_scale'],
                    'Dimensiones_Adicionales': config['dimensions'],
                    'Ratio_Varianza': stats['variance_ratio'],
                    'Idioma': lang_name,
                    'Top1_Accuracy': metrics['top1_accuracy'],
                    'Top3_Accuracy': metrics['top3_accuracy'],
                    'MRR_Score': metrics['mrr_score'],
                    'Posicion_Promedio': metrics['average_position'],
                    'Total_Preguntas': metrics['total_questions']
                })
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(rows)
        df.to_csv('aggressive_benchmark_detailed_results.csv', index=False, encoding='utf-8')
    
    def run_aggressive_evaluation(self):
        """Ejecuta evaluación completa de expansiones agresivas"""
        print("🚀 INICIANDO EVALUACIÓN AGRESIVA COMPLETA")
        print("=" * 60)
        
        # Cargar datos necesarios
        if not self.load_model():
            return False
        
        if not self.load_benchmark_data():
            return False
        
        if not self.load_baseline_results():
            return False
        
        if not self.load_aggressive_expansions():
            return False
        
        # Evaluar todas las expansiones
        evaluation_results = self.evaluate_all_expansions()
        
        # Generar reporte comparativo
        if evaluation_results:
            self.generate_aggressive_comparison_report(evaluation_results)
            return True
        else:
            print("❌ No se completaron evaluaciones exitosamente")
            return False

def main():
    """Función principal para evaluación agresiva"""
    print("🚀 INICIANDO EVALUACIÓN AGRESIVA DE EXPANSIONES")
    print("=" * 60)
    
    # Crear evaluador
    evaluator = AggressiveBenchmarkEvaluator()
    
    # Ejecutar evaluación completa
    if evaluator.run_aggressive_evaluation():
        print("\n🎉 EVALUACIÓN AGRESIVA COMPLETADA!")
        print("\n📊 RESUMEN:")
        print(f"   • Expansiones evaluadas: {len(evaluator.expansion_results)}")
        print(f"   • Baseline establecido para comparación")
        print(f"   • Reporte comparativo agresivo generado")
        
        print(f"\n📁 Archivos generados:")
        print(f"   - aggressive_benchmark_comparison_report.txt")
        print(f"   - aggressive_benchmark_detailed_results.csv")
        
        print(f"\n💡 Próximo paso: Revisar el reporte para identificar la mejor expansión agresiva")
    else:
        print("❌ Error en la evaluación agresiva")

if __name__ == "__main__":
    main()
