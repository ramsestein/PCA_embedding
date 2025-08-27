#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador Completo de Benchmark para Estrategias de Embedding
Integra todas las estrategias implementadas y evalúa contra benchmark real
"""

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import time
from pathlib import Path

class ComprehensiveBenchmarkEvaluator:
    """
    Evaluador que integra todas las estrategias implementadas:
    - Estrategia 4: Reducción de Dimensionalidad Inteligente (PCA multi-dim, t-SNE)
    - Estrategia 5: Métricas de Similitud Avanzadas
    - Evalúa contra benchmark real para medir rendimiento de recuperación
    """
    
    def __init__(self, model_path: str = "all-mini-base"):
        self.model_path = model_path
        self.model = None
        self.pnts_documents = []
        self.pnts_filenames = []
        self.benchmark_data = []
        
        # Resultados de evaluación
        self.baseline_results = {}
        self.pca_results = {}
        self.tsne_results = {}
        self.comprehensive_results = {}
        
    def load_model(self):
        """Carga el modelo de embedding"""
        print("🔄 Cargando modelo de embedding...")
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
    
    def generate_embeddings(self):
        """Genera embeddings para documentos y consultas"""
        print("🔄 Generando embeddings...")
        
        # Embeddings de documentos PNTs
        self.doc_embeddings = self.model.encode(
            self.pnts_documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✅ Embeddings de documentos: {self.doc_embeddings.shape}")
        
        # Embeddings de consultas del benchmark
        self.query_embeddings = self.model.encode(
            [item['query'] for item in self.benchmark_data],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✅ Embeddings de consultas: {self.query_embeddings.shape}")
    
    def apply_pca_dimensionality_reduction(self):
        """Aplica PCA con múltiples dimensiones"""
        print("🔄 Aplicando PCA multi-dimensional...")
        
        pca_dimensions = [2, 5, 10, 15]
        
        for dim in pca_dimensions:
            print(f"  📊 PCA {dim}D...")
            pca = PCA(n_components=dim, random_state=42)
            
            # Aplicar PCA a documentos
            doc_pca = pca.fit_transform(self.doc_embeddings)
            
            # Aplicar PCA a consultas (usando el mismo fit)
            query_pca = pca.transform(self.query_embeddings)
            
            self.pca_results[f"PCA_{dim}D"] = {
                'doc_embeddings': doc_pca,
                'query_embeddings': query_pca,
                'explained_variance_ratio': pca.explained_variance_ratio_.sum(),
                'n_components': dim
            }
            
            print(f"    ✅ PCA {dim}D completado - Var explicada: {pca.explained_variance_ratio_.sum():.4f}")
    
    def apply_tsne_dimensionality_reduction(self):
        """Aplica t-SNE optimizado"""
        print("🔄 Aplicando t-SNE optimizado...")
        
        try:
            # t-SNE para documentos
            tsne_doc = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(self.pnts_documents) - 1),
                max_iter=1000,  # Cambiado de n_iter a max_iter
                metric='cosine'
            )
            doc_tsne = tsne_doc.fit_transform(self.doc_embeddings)
            
            # t-SNE para consultas (usando el mismo fit)
            tsne_query = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(self.benchmark_data) - 1),
                max_iter=1000,  # Cambiado de n_iter a max_iter
                metric='cosine'
            )
            query_tsne = tsne_query.fit_transform(self.query_embeddings)
            
            self.tsne_results = {
                'doc_embeddings': doc_tsne,
                'query_embeddings': query_tsne,
                'n_components': 2
            }
            
            print("✅ t-SNE completado")
        except Exception as e:
            print(f"⚠️ Error en t-SNE: {e}")
            self.tsne_results = None
    
    def calculate_similarity_metrics(self, doc_emb, query_emb, method_name: str):
        """Calcula métricas de similitud entre consultas y documentos"""
        similarities = []
        
        for i, query_emb_i in enumerate(query_emb):
            query_similarities = []
            
            for j, doc_emb_j in enumerate(doc_emb):
                # Coseno
                cos_sim = cosine_similarity([query_emb_i], [doc_emb_j])[0][0]
                
                # Euclidiana
                eucl_dist = euclidean(query_emb_i, doc_emb_j)
                
                # Distancia híbrida (combinación de coseno y euclidiana)
                hybrid_dist = (1 - cos_sim) + (eucl_dist / np.max(eucl_dist)) if np.max(eucl_dist) > 0 else (1 - cos_sim)
                
                query_similarities.append({
                    'document_index': j,
                    'filename': self.pnts_filenames[j],
                    'cosine_similarity': cos_sim,
                    'euclidean_distance': eucl_dist,
                    'hybrid_distance': hybrid_dist
                })
            
            # Ordenar por similitud de coseno (mayor = mejor)
            query_similarities.sort(key=lambda x: x['cosine_similarity'], reverse=True)
            similarities.append(query_similarities)
        
        return similarities
    
    def evaluate_retrieval_performance(self, doc_emb: np.ndarray, query_emb: np.ndarray, method_name: str = "Unknown") -> Dict[str, float]:
        """Evalúa el rendimiento de recuperación de manera consistente"""
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
                        
                        # Top-5 accuracy - CORREGIDO para ser consistente
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
            
            print(f"  ✅ {method_name}: Top-1={results['top1_accuracy']:.4f}, Top-3={results['top3_accuracy']:.4f}, Top-5={results['top5_accuracy']:.4f}, MRR={results['mrr']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error evaluando rendimiento para {method_name}: {e}")
            raise
    
    def run_comprehensive_evaluation(self):
        """Ejecuta la evaluación completa de todas las estrategias"""
        print("🚀 INICIANDO EVALUACIÓN COMPREHENSIVA")
        print("=" * 80)
        
        try:
            # PASO 1: Cargar modelo
            self.load_model()
            
            # PASO 2: Cargar documentos PNTs
            self.load_pnts_documents()
            
            # PASO 3: Cargar benchmark
            self.load_benchmark()
            
            # PASO 4: Generar embeddings
            self.generate_embeddings()
            
            # PASO 5: Evaluar baseline (embeddings originales)
            print("\n📊 EVALUANDO BASELINE (Embeddings Originales)")
            print("-" * 50)
            self.baseline_results = self.evaluate_retrieval_performance(
                self.doc_embeddings, 
                self.query_embeddings, 
                "Baseline_Original"
            )
            
            # PASO 6: Aplicar PCA multi-dimensional
            print("\n📊 APLICANDO ESTRATEGIA 4: PCA Multi-Dimensional")
            print("-" * 50)
            self.apply_pca_dimensionality_reduction()
            
            # PASO 7: Evaluar PCA
            for pca_name, pca_data in self.pca_results.items():
                print(f"\n📊 Evaluando {pca_name}...")
                self.pca_results[pca_name]['evaluation'] = self.evaluate_retrieval_performance(
                    pca_data['doc_embeddings'],
                    pca_data['query_embeddings'],
                    pca_name
                )
            
            # PASO 8: Aplicar t-SNE
            print("\n📊 APLICANDO ESTRATEGIA 4: t-SNE")
            print("-" * 50)
            self.apply_tsne_dimensionality_reduction()
            
            # PASO 9: Evaluar t-SNE
            if self.tsne_results:
                print("\n📊 Evaluando t-SNE...")
                self.tsne_results['evaluation'] = self.evaluate_retrieval_performance(
                    self.tsne_results['doc_embeddings'],
                    self.tsne_results['query_embeddings'],
                    "t-SNE_2D"
                )
            
            # PASO 10: Generar reporte comprehensivo
            print("\n📊 GENERANDO REPORTE COMPREHENSIVO")
            print("-" * 50)
            self.generate_comprehensive_report()
            
            print("\n🎉 EVALUACIÓN COMPREHENSIVA COMPLETADA!")
            
        except Exception as e:
            print(f"❌ ERROR EN LA EVALUACIÓN: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_comprehensive_report(self):
        """Genera un reporte comprehensivo de todos los resultados"""
        print("🔄 Generando reporte comprehensivo...")
        
        # Preparar datos para el reporte
        report_data = []
        
        # Baseline
        baseline = self.baseline_results
        report_data.append({
            'Método': 'Baseline (Original)',
            'Dimensiones': self.doc_embeddings.shape[1],
            'Top-1 Accuracy': f"{baseline['top1_accuracy']:.4f}",
            'Top-3 Accuracy': f"{baseline['top3_accuracy']:.4f}",
            'Top-5 Accuracy': f"{baseline['top5_accuracy']:.4f}",
            'MRR': f"{baseline['mrr']:.4f}",
            'Varianza Explicada': 'N/A'
        })
        
        # PCA results
        for pca_name, pca_data in self.pca_results.items():
            eval_data = pca_data['evaluation']
            report_data.append({
                'Método': pca_name,
                'Dimensiones': pca_data['n_components'],
                'Top-1 Accuracy': f"{eval_data['top1_accuracy']:.4f}",
                'Top-3 Accuracy': f"{eval_data['top3_accuracy']:.4f}",
                'Top-5 Accuracy': f"{eval_data['top5_accuracy']:.4f}",
                'MRR': f"{eval_data['mrr']:.4f}",
                'Varianza Explicada': f"{pca_data['explained_variance_ratio']:.4f}"
            })
        
        # t-SNE results
        if self.tsne_results:
            tsne_eval = self.tsne_results['evaluation']
            report_data.append({
                'Método': 't-SNE 2D',
                'Dimensiones': 2,
                'Top-1 Accuracy': f"{tsne_eval['top1_accuracy']:.4f}",
                'Top-3 Accuracy': f"{tsne_eval['top3_accuracy']:.4f}",
                'Top-5 Accuracy': f"{tsne_eval['top5_accuracy']:.4f}",
                'MRR': f"{tsne_eval['mrr']:.4f}",
                'Varianza Explicada': 'N/A'
            })
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(report_data)
        
        # Guardar reporte CSV
        csv_path = "comprehensive_benchmark_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Reporte CSV guardado: {csv_path}")
        
        # Guardar reporte detallado en texto
        txt_path = "comprehensive_benchmark_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE COMPREHENSIVO DE BENCHMARK\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total consultas evaluadas: {len(self.benchmark_data)}\n")
            f.write(f"Total documentos PNTs: {len(self.pnts_documents)}\n\n")
            
            f.write("RESULTADOS DE EVALUACIÓN:\n")
            f.write("-" * 40 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Análisis detallado
            f.write("ANÁLISIS DETALLADO:\n")
            f.write("-" * 40 + "\n")
            
            # Encontrar mejor método
            best_method = df.loc[df['MRR'].astype(float).idxmax()]
            f.write(f"🏆 MEJOR MÉTODO: {best_method['Método']}\n")
            f.write(f"   MRR: {best_method['MRR']}\n")
            f.write(f"   Top-1: {best_method['Top-1 Accuracy']}\n")
            f.write(f"   Top-3: {best_method['Top-3 Accuracy']}\n\n")
            
            # Comparación con baseline
            baseline_row = df[df['Método'] == 'Baseline (Original)'].iloc[0]
            baseline_mrr = float(baseline_row['MRR'])
            
            for _, row in df.iterrows():
                if row['Método'] != 'Baseline (Original)':
                    method_mrr = float(row['MRR'])
                    improvement = ((method_mrr - baseline_mrr) / baseline_mrr) * 100
                    f.write(f"📊 {row['Método']} vs Baseline:\n")
                    f.write(f"   Mejora MRR: {improvement:+.2f}%\n")
                    f.write(f"   Top-1: {row['Top-1 Accuracy']} vs {baseline_row['Top-1 Accuracy']}\n\n")
        
        print(f"✅ Reporte detallado guardado: {txt_path}")
        
        # Generar visualización
        self.generate_comprehensive_visualization(df)
    
    def generate_comprehensive_visualization(self, df):
        """Genera visualizaciones comprehensivas de los resultados"""
        print("🔄 Generando visualizaciones...")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Evaluación Comprehensiva de Estrategias de Embedding', fontsize=16, fontweight='bold')
        
        # 1. Comparación de MRR
        ax1 = axes[0, 0]
        methods = df['Método']
        mrr_values = [float(x) for x in df['MRR']]
        colors = ['#2E86AB' if 'Baseline' in m else '#A23B72' if 'PCA' in m else '#F18F01' for m in methods]
        
        bars1 = ax1.bar(methods, mrr_values, color=colors, alpha=0.8)
        ax1.set_title('Mean Reciprocal Rank (MRR) por Método', fontweight='bold')
        ax1.set_ylabel('MRR')
        ax1.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, value in zip(bars1, mrr_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Comparación de Top-1 Accuracy
        ax2 = axes[0, 1]
        top1_values = [float(x) for x in df['Top-1 Accuracy']]
        bars2 = ax2.bar(methods, top1_values, color=colors, alpha=0.8)
        ax2.set_title('Top-1 Accuracy por Método', fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, top1_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Comparación de Top-3 vs Top-5
        ax3 = axes[1, 0]
        x = np.arange(len(methods))
        width = 0.35
        
        top3_values = [float(x) for x in df['Top-3 Accuracy']]
        top5_values = [float(x) for x in df['Top-5 Accuracy']]
        
        bars3 = ax3.bar(x - width/2, top3_values, width, label='Top-3', color='#2E86AB', alpha=0.8)
        bars4 = ax3.bar(x + width/2, top5_values, width, label='Top-5', color='#A23B72', alpha=0.8)
        
        ax3.set_title('Top-3 vs Top-5 Accuracy', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods, rotation=45)
        ax3.legend()
        
        # 4. Varianza explicada (solo para PCA)
        ax4 = axes[1, 1]
        pca_methods = [m for m in methods if 'PCA' in m]
        pca_variance = [float(df[df['Método'] == m]['Varianza Explicada'].iloc[0]) for m in pca_methods]
        
        if pca_variance:
            bars5 = ax4.bar(pca_methods, pca_variance, color='#F18F01', alpha=0.8)
            ax4.set_title('Varianza Explicada por PCA', fontweight='bold')
            ax4.set_ylabel('Varianza Explicada')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars5, pca_variance):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'No hay datos de PCA\npara mostrar', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Varianza Explicada por PCA', fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar visualización
        viz_path = "comprehensive_benchmark_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualización guardada: {viz_path}")
        
        plt.show()

def main():
    """Función principal"""
    print("🚀 EVALUADOR COMPREHENSIVO DE BENCHMARK")
    print("Integra todas las estrategias implementadas")
    print("=" * 80)
    
    try:
        # Crear evaluador
        evaluator = ComprehensiveBenchmarkEvaluator()
        
        # Ejecutar evaluación completa
        evaluator.run_comprehensive_evaluation()
        
        print("\n🎯 EVALUACIÓN COMPLETADA")
        print("Revisa los archivos generados:")
        print("  📊 comprehensive_benchmark_results.csv - Resultados en formato tabla")
        print("  📋 comprehensive_benchmark_report.txt - Reporte detallado")
        print("  🖼️  comprehensive_benchmark_visualization.png - Visualizaciones")
        
    except Exception as e:
        print(f"❌ ERROR EN LA EJECUCIÓN: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
