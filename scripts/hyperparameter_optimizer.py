#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizador de Hiperparámetros - Estrategia 8 (CORREGIDO)
Optimiza parámetros REALES del modelo sentence-transformers para maximizar rendimiento en benchmark
"""

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid
from typing import List, Dict, Tuple, Any
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class HyperparameterOptimizer:
    """
    Optimizador de hiperparámetros para modelo sentence-transformers
    Evalúa diferentes configuraciones contra benchmark real
    """
    
    def __init__(self, model_path: str = "all-mini-base"):
        self.model_path = model_path
        self.model = None
        self.pnts_documents = []
        self.pnts_filenames = []
        self.benchmark_data = []
        
        # ESPACIO DE HIPERPARÁMETROS REALES Y APLICABLES
        self.param_grid = {
            'normalize_embeddings': [True, False],
            'convert_to_numpy': [True, False],
            'batch_size': [8, 16, 32],
            'show_progress_bar': [False]  # Mantener False para velocidad
        }
        
        # Resultados de optimización
        self.optimization_results = []
        self.best_config = None
        self.best_score = 0.0
        
    def load_model(self):
        """Carga el modelo base"""
        print("🔄 Cargando modelo base...")
        try:
            self.model = SentenceTransformer(self.model_path)
            print(f"✅ Modelo cargado: {self.model_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def load_pnts_documents(self):
        """Carga todos los documentos PNTs"""
        print("🔄 Cargando documentos PNTs...")
        pnts_dir = Path("PNTs")
        
        if not pnts_dir.exists():
            raise FileNotFoundError("Carpeta PNTs no encontrada")
        
        for file_path in pnts_dir.glob("*_limpio.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Solo documentos no vacíos
                        self.pnts_documents.append(content)
                        self.pnts_filenames.append(file_path.name)
            except Exception as e:
                print(f"⚠️ Error leyendo {file_path}: {e}")
        
        print(f"✅ {len(self.pnts_documents)} documentos PNTs cargados")
    
    def load_benchmark(self):
        """Carga los datos del benchmark"""
        print("🔄 Cargando benchmark...")
        
        # Cargar benchmark en español
        es_benchmark_path = Path("benchmark/preguntas_con_docs_es.json")
        if es_benchmark_path.exists():
            with open(es_benchmark_path, 'r', encoding='utf-8') as f:
                es_data = json.load(f)
                self.benchmark_data.extend(es_data)
                print(f"✅ Benchmark ES cargado: {len(es_data)} consultas")
        
        # Cargar benchmark en catalán
        cat_benchmark_path = Path("benchmark/preguntas_con_docs_cat.json")
        if cat_benchmark_path.exists():
            with open(cat_benchmark_path, 'r', encoding='utf-8') as f:
                cat_data = json.load(f)
                self.benchmark_data.extend(cat_data)
                print(f"✅ Benchmark CAT cargado: {len(cat_data)} consultas")
        
        print(f"✅ Total benchmark: {len(self.benchmark_data)} consultas")
    
    def apply_hyperparameters(self, params: Dict[str, Any]) -> SentenceTransformer:
        """Aplica hiperparámetros REALES al modelo"""
        try:
            # Crear una copia del modelo con nuevos parámetros
            model_copy = SentenceTransformer(self.model_path)
            
            # Aplicar parámetros de configuración que SÍ se pueden cambiar
            if 'normalize_embeddings' in params:
                model_copy.normalize_embeddings = params['normalize_embeddings']
            
            if 'convert_to_numpy' in params:
                model_copy.convert_to_numpy = params['convert_to_numpy']
            
            # Verificar que los cambios se aplicaron
            print(f"  ✅ normalize_embeddings: {model_copy.normalize_embeddings}")
            print(f"  ✅ convert_to_numpy: {model_copy.convert_to_numpy}")
            
            return model_copy
            
        except Exception as e:
            print(f"❌ Error aplicando hiperparámetros: {e}")
            raise  # NO retornar modelo original, fallar explícitamente
    
    def generate_embeddings_with_params(self, model: SentenceTransformer, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Genera embeddings con parámetros específicos"""
        try:
            # Parámetros para encode
            encode_params = {
                'show_progress_bar': params.get('show_progress_bar', False),
                'convert_to_numpy': params.get('convert_to_numpy', True),
                'batch_size': params.get('batch_size', 16)
            }
            
            print(f"  🔄 Generando embeddings con batch_size={encode_params['batch_size']}")
            
            # Embeddings de documentos PNTs
            doc_embeddings = model.encode(
                self.pnts_documents, 
                **encode_params
            )
            
            # Embeddings de consultas del benchmark
            query_embeddings = model.encode(
                [item['query'] for item in self.benchmark_data],
                **encode_params
            )
            
            # Normalizar embeddings si se especifica
            if params.get('normalize_embeddings', False):
                print(f"  🔄 Normalizando embeddings...")
                doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
                query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            
            print(f"  ✅ Embeddings generados: docs={doc_embeddings.shape}, queries={query_embeddings.shape}")
            return doc_embeddings, query_embeddings
            
        except Exception as e:
            print(f"❌ Error generando embeddings: {e}")
            raise
    
    def evaluate_retrieval_performance(self, doc_emb: np.ndarray, query_emb: np.ndarray) -> Dict[str, float]:
        """Evalúa el rendimiento de recuperación"""
        try:
            # Calcular similitudes
            similarities = cosine_similarity(query_emb, doc_emb)
            
            # Métricas de evaluación
            top1_accuracy = 0
            top3_accuracy = 0
            top5_accuracy = 0
            mrr_scores = []
            
            for i, query_similarities in enumerate(similarities):
                expected_doc = self.benchmark_data[i]['document_expected']
                
                # Crear ranking de documentos
                doc_ranking = [(j, sim) for j, sim in enumerate(query_similarities)]
                doc_ranking.sort(key=lambda x: x[1], reverse=True)
                
                # Encontrar posición del documento esperado
                expected_found = False
                for rank, (doc_idx, sim_score) in enumerate(doc_ranking):
                    if self.pnts_filenames[doc_idx] == expected_doc:
                        expected_found = True
                        
                        # Top-1 accuracy
                        if rank == 0:
                            top1_accuracy += 1
                        
                        # Top-3 accuracy
                        if rank < 3:
                            top3_accuracy += 1
                        
                        # Top-5 accuracy
                        if rank < 5:
                            top5_accuracy += 1
                        
                        # Mean Reciprocal Rank
                        mrr_scores.append(1.0 / (rank + 1))
                        break
                
                if not expected_found:
                    mrr_scores.append(0.0)
            
            # Calcular promedios
            total_queries = len(self.benchmark_data)
            results = {
                'top1_accuracy': top1_accuracy / total_queries,
                'top3_accuracy': top3_accuracy / total_queries,
                'top5_accuracy': top5_accuracy / total_queries,
                'mrr': np.mean(mrr_scores),
                'total_queries': total_queries
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error evaluando rendimiento: {e}")
            raise
    
    def run_hyperparameter_optimization(self, max_combinations: int = 20):
        """Ejecuta la optimización de hiperparámetros"""
        print("🚀 INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS (CORREGIDA)")
        print("=" * 80)
        
        try:
            # PASO 1: Cargar modelo y datos
            self.load_model()
            self.load_pnts_documents()
            self.load_benchmark()
            
            # PASO 2: Generar combinaciones de parámetros
            print("\n📊 GENERANDO COMBINACIONES DE PARÁMETROS")
            print("-" * 50)
            
            param_combinations = list(ParameterGrid(self.param_grid))
            print(f"Total combinaciones posibles: {len(param_combinations)}")
            
            # Limitar el número de combinaciones si es muy alto
            if len(param_combinations) > max_combinations:
                # Seleccionar combinaciones estratégicamente
                step = len(param_combinations) // max_combinations
                param_combinations = param_combinations[::step][:max_combinations]
                print(f"Combinaciones seleccionadas para evaluación: {len(param_combinations)}")
            
            # PASO 3: Evaluar cada combinación
            print("\n🔍 EVALUANDO COMBINACIONES DE PARÁMETROS")
            print("-" * 50)
            
            for i, params in enumerate(param_combinations, 1):
                print(f"\n📊 Evaluando combinación {i}/{len(param_combinations)}")
                print(f"Parámetros: {params}")
                
                try:
                    # Aplicar hiperparámetros
                    model_with_params = self.apply_hyperparameters(params)
                    
                    # Generar embeddings
                    doc_emb, query_emb = self.generate_embeddings_with_params(model_with_params, params)
                    
                    # Evaluar rendimiento
                    performance = self.evaluate_retrieval_performance(doc_emb, query_emb)
                    
                    # Guardar resultados
                    result = {
                        'combination_id': i,
                        'parameters': params.copy(),
                        'performance': performance,
                        'doc_embeddings_shape': doc_emb.shape,
                        'query_embeddings_shape': query_emb.shape
                    }
                    
                    self.optimization_results.append(result)
                    
                    # Actualizar mejor configuración
                    if performance['mrr'] > self.best_score:
                        self.best_score = performance['mrr']
                        self.best_config = params.copy()
                        print(f"🏆 ¡NUEVA MEJOR CONFIGURACIÓN! MRR: {self.best_score:.4f}")
                    
                    print(f"✅ MRR: {performance['mrr']:.4f}, Top-1: {performance['top1_accuracy']:.4f}, Top-3: {performance['top3_accuracy']:.4f}, Top-5: {performance['top5_accuracy']:.4f}")
                    
                except Exception as e:
                    print(f"❌ Error en combinación {i}: {e}")
                    continue
            
            # PASO 4: Generar reporte
            print("\n📊 GENERANDO REPORTE DE OPTIMIZACIÓN")
            print("-" * 50)
            self.generate_optimization_report()
            
            print("\n🎉 OPTIMIZACIÓN DE HIPERPARÁMETROS COMPLETADA!")
            
        except Exception as e:
            print(f"❌ ERROR EN LA OPTIMIZACIÓN: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_optimization_report(self):
        """Genera reporte de la optimización de hiperparámetros"""
        print("🔄 Generando reporte de optimización...")
        
        if not self.optimization_results:
            print("❌ No hay resultados para generar reporte")
            return
        
        # Ordenar resultados por MRR
        sorted_results = sorted(self.optimization_results, key=lambda x: x['performance']['mrr'], reverse=True)
        
        # Preparar datos para el reporte
        report_data = []
        for result in sorted_results:
            params = result['parameters']
            perf = result['performance']
            
            report_data.append({
                'Combinación': result['combination_id'],
                'Normalizar': params.get('normalize_embeddings', 'N/A'),
                'Batch Size': params.get('batch_size', 'N/A'),
                'Top-1': f"{perf['top1_accuracy']:.4f}",
                'Top-3': f"{perf['top3_accuracy']:.4f}",
                'Top-5': f"{perf['top5_accuracy']:.4f}",
                'MRR': f"{perf['mrr']:.4f}",
                'Ranking': len(report_data) + 1
            })
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(report_data)
        
        # Guardar reporte CSV
        csv_path = "hyperparameter_optimization_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Reporte CSV guardado: {csv_path}")
        
        # Guardar reporte detallado en texto
        txt_path = "hyperparameter_optimization_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE OPTIMIZACIÓN DE HIPERPARÁMETROS - ESTRATEGIA 8 (CORREGIDA)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total combinaciones evaluadas: {len(self.optimization_results)}\n")
            f.write(f"Mejor MRR obtenido: {self.best_score:.4f}\n\n")
            
            f.write("🏆 MEJOR CONFIGURACIÓN ENCONTRADA:\n")
            f.write("-" * 40 + "\n")
            for key, value in self.best_config.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nRendimiento: MRR = {self.best_score:.4f}\n\n")
            
            f.write("RANKING COMPLETO DE CONFIGURACIONES:\n")
            f.write("-" * 40 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Análisis de parámetros más influyentes
            f.write("ANÁLISIS DE PARÁMETROS MÁS INFLUYENTES:\n")
            f.write("-" * 40 + "\n")
            
            # Analizar impacto de cada parámetro
            param_impact = {}
            for param_name in self.param_grid.keys():
                if param_name in self.best_config:
                    # Encontrar configuraciones con y sin este parámetro
                    with_param = [r for r in self.optimization_results if r['parameters'].get(param_name) == self.best_config[param_name]]
                    without_param = [r for r in self.optimization_results if r['parameters'].get(param_name) != self.best_config[param_name]]
                    
                    if with_param and without_param:
                        avg_with = np.mean([r['performance']['mrr'] for r in with_param])
                        avg_without = np.mean([r['performance']['mrr'] for r in without_param])
                        impact = avg_with - avg_without
                        param_impact[param_name] = impact
            
            # Ordenar por impacto
            sorted_impact = sorted(param_impact.items(), key=lambda x: abs(x[1]), reverse=True)
            
            f.write("Impacto de parámetros en MRR:\n")
            for param, impact in sorted_impact:
                f.write(f"  {param}: {impact:+.4f}\n")
            
            # Recomendaciones
            f.write("\n💡 RECOMENDACIONES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Implementar la configuración ganadora en producción\n")
            if sorted_impact:
                f.write(f"• Los parámetros más críticos son: {', '.join([p for p, _ in sorted_impact[:3]])}\n")
            f.write(f"• Considerar fine-tuning adicional con estos hiperparámetros\n")
        
        print(f"✅ Reporte detallado guardado: {txt_path}")
        
        # Generar visualización
        self.generate_optimization_visualization(df)
    
    def generate_optimization_visualization(self, df):
        """Genera visualizaciones de los resultados de optimización"""
        print("🔄 Generando visualizaciones...")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Optimización de Hiperparámetros - Estrategia 8 (CORREGIDA)', fontsize=16, fontweight='bold')
        
        # 1. Ranking de MRR por combinación
        ax1 = axes[0, 0]
        combinations = df['Combinación']
        mrr_values = [float(x) for x in df['MRR']]
        
        bars1 = ax1.bar(range(len(combinations)), mrr_values, color='#2E86AB', alpha=0.8)
        ax1.set_title('MRR por Combinación de Parámetros', fontweight='bold')
        ax1.set_ylabel('MRR')
        ax1.set_xlabel('Combinación')
        
        # Resaltar la mejor combinación
        best_idx = mrr_values.index(max(mrr_values))
        bars1[best_idx].set_color('#E74C3C')
        
        # Añadir valores en las barras
        for bar, value in zip(bars1, mrr_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Top-1 Accuracy por combinación
        ax2 = axes[0, 1]
        top1_values = [float(x) for x in df['Top-1']]
        
        bars2 = ax2.bar(range(len(combinations)), top1_values, color='#A23B72', alpha=0.8)
        ax2.set_title('Top-1 Accuracy por Combinación', fontweight='bold')
        ax2.set_ylabel('Top-1 Accuracy')
        ax2.set_xlabel('Combinación')
        
        # Resaltar la mejor combinación
        bars2[best_idx].set_color('#E74C3C')
        
        for bar, value in zip(bars2, top1_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Comparación de Top-1 vs Top-3
        ax3 = axes[1, 0]
        top3_values = [float(x) for x in df['Top-3']]
        
        x = np.arange(len(combinations))
        width = 0.35
        
        bars3 = ax3.bar(x - width/2, top1_values, width, label='Top-1', color='#2E86AB', alpha=0.8)
        bars4 = ax3.bar(x + width/2, top3_values, width, label='Top-3', color='#A23B72', alpha=0.8)
        
        ax3.set_title('Top-1 vs Top-3 Accuracy', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Combinación')
        ax3.legend()
        
        # 4. Distribución de MRR
        ax4 = axes[1, 1]
        ax4.hist(mrr_values, bins=10, color='#F18F01', alpha=0.8, edgecolor='black')
        ax4.axvline(self.best_score, color='#E74C3C', linestyle='--', linewidth=2, label=f'Mejor: {self.best_score:.4f}')
        ax4.set_title('Distribución de MRR', fontweight='bold')
        ax4.set_xlabel('MRR')
        ax4.set_ylabel('Frecuencia')
        ax4.legend()
        
        plt.tight_layout()
        
        # Guardar visualización
        viz_path = "hyperparameter_optimization_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualización guardada: {viz_path}")
        
        plt.show()

def main():
    """Función principal"""
    print("🚀 OPTIMIZADOR DE HIPERPARÁMETROS (CORREGIDO)")
    print("Estrategia 8: Optimización de Hiperparámetros del Modelo")
    print("=" * 80)
    
    try:
        # Crear optimizador
        optimizer = HyperparameterOptimizer()
        
        # Ejecutar optimización
        optimizer.run_hyperparameter_optimization(max_combinations=20)
        
        print("\n🎯 OPTIMIZACIÓN COMPLETADA")
        print("Revisa los archivos generados:")
        print("  📊 hyperparameter_optimization_results.csv - Resultados en formato tabla")
        print("  📋 hyperparameter_optimization_report.txt - Reporte detallado")
        print("  🖼️  hyperparameter_optimization_visualization.png - Visualizaciones")
        
    except Exception as e:
        print(f"❌ ERROR EN LA EJECUCIÓN: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
