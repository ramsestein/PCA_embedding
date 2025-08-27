#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Inteligente Funcional - Estrategias 4 y 5
Autor: Análisis de Embeddings Médicos
Fecha: 2025
Objetivo: Implementar Estrategias 4 y 5 de forma robusta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
warnings.filterwarnings('ignore')

# Configuración de matplotlib para español
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class WorkingIntelligentSystem:
    """Sistema inteligente funcional implementando Estrategias 4 y 5"""
    
    def __init__(self, model_path="all-mini-base", pnts_folder="PNTs", benchmark_folder="benchmark"):
        self.model_path = model_path
        self.pnts_folder = pnts_folder
        self.benchmark_folder = benchmark_folder
        self.model = None
        self.pnts_documents = []
        self.pnts_names = []
        self.benchmark_data = {}
        self.original_embeddings = None
        
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
    
    def load_pnts_documents(self):
        """Carga todos los documentos PNTs"""
        print(f"📁 Cargando documentos PNTs desde: {self.pnts_folder}")
        
        if not os.path.exists(self.pnts_folder):
            print(f"❌ Error: Carpeta {self.pnts_folder} no encontrada")
            return False
        
        txt_files = [f for f in os.listdir(self.pnts_folder) if f.endswith('.txt')]
        
        for filename in txt_files:
            file_path = os.path.join(self.pnts_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.pnts_documents.append(content)
                        self.pnts_names.append(filename)
            except Exception as e:
                print(f"   ⚠️ Error al leer {filename}: {e}")
        
        print(f"✅ Documentos cargados: {len(self.pnts_documents)}")
        return len(self.pnts_documents) > 0
    
    def load_benchmark_data(self):
        """Carga datos del benchmark"""
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
    
    def generate_embeddings(self):
        """Genera embeddings para los documentos PNTs"""
        if self.model is None:
            print("❌ Error: Modelo no cargado")
            return False
        
        print("🔄 Generando embeddings...")
        try:
            self.original_embeddings = self.model.encode(self.pnts_documents, show_progress_bar=True)
            print(f"✅ Embeddings generados: {self.original_embeddings.shape}")
            return True
        except Exception as e:
            print(f"❌ Error al generar embeddings: {e}")
            return False
    
    def implement_strategy_4_dimensionality_reduction(self):
        """Implementa Estrategia 4: Reducción de Dimensionalidad Inteligente"""
        print("\n" + "="*60)
        print("🔬 ESTRATEGIA 4: REDUCCIÓN DE DIMENSIONALIDAD INTELIGENTE")
        print("="*60)
        
        results = {}
        
        # 1. PCA como baseline
        print("\n📊 1. ANÁLISIS PCA (Baseline)")
        embeddings_pca, pca_reducer = self._perform_pca_analysis()
        results['PCA_Baseline'] = {'embeddings': embeddings_pca, 'reducer': pca_reducer}
        
        # 2. t-SNE Optimizado
        print("\n📊 2. ANÁLISIS t-SNE OPTIMIZADO")
        embeddings_tsne, tsne_reducer = self._perform_tsne_analysis()
        results['TSNE_Optimized'] = {'embeddings': embeddings_tsne, 'reducer': tsne_reducer}
        
        # 3. PCA con diferentes números de componentes
        print("\n📊 3. ANÁLISIS PCA MULTI-DIMENSIONAL")
        pca_results = self._perform_multi_pca_analysis()
        results.update(pca_results)
        
        return results
    
    def implement_strategy_5_advanced_metrics(self):
        """Implementa Estrategia 5: Métricas de Similitud Avanzadas"""
        print("\n" + "="*60)
        print("🔬 ESTRATEGIA 5: MÉTRICAS DE SIMILITUD AVANZADAS")
        print("="*60)
        
        results = {}
        
        # Calcular métricas avanzadas para embeddings originales
        print("🔄 Calculando métricas de similitud avanzadas...")
        advanced_metrics = self._calculate_advanced_similarity_metrics(self.original_embeddings)
        results['Advanced_Metrics'] = advanced_metrics
        
        # Calcular métricas para diferentes representaciones
        print("🔄 Calculando métricas para representaciones reducidas...")
        
        # PCA 2D
        pca_2d = PCA(n_components=2, random_state=42)
        embeddings_pca_2d = pca_2d.fit_transform(StandardScaler().fit_transform(self.original_embeddings))
        metrics_pca_2d = self._calculate_advanced_similarity_metrics(embeddings_pca_2d)
        results['PCA_2D_Metrics'] = metrics_pca_2d
        
        # PCA 5D
        pca_5d = PCA(n_components=5, random_state=42)
        embeddings_pca_5d = pca_5d.fit_transform(StandardScaler().fit_transform(self.original_embeddings))
        metrics_pca_5d = self._calculate_advanced_similarity_metrics(embeddings_pca_5d)
        results['PCA_5D_Metrics'] = metrics_pca_5d
        
        return results
    
    def _perform_pca_analysis(self, n_components=10):
        """Realiza análisis PCA"""
        print(f"   🔄 Aplicando PCA con {n_components} componentes...")
        
        # Normalización
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.original_embeddings)
        
        # Aplicar PCA
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        
        print(f"   ✅ PCA completado: {embeddings_pca.shape}")
        print(f"   📊 Varianza explicada: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        return embeddings_pca, pca
    
    def _perform_tsne_analysis(self, n_components=2, perplexity=10):
        """Realiza análisis t-SNE optimizado"""
        print(f"   🔄 Aplicando t-SNE con {n_components} componentes...")
        
        # Normalización
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.original_embeddings)
        
        # Aplicar t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        embeddings_tsne = tsne.fit_transform(embeddings_scaled)
        
        print(f"   ✅ t-SNE completado: {embeddings_tsne.shape}")
        return embeddings_tsne, tsne
    
    def _perform_multi_pca_analysis(self):
        """Realiza análisis PCA con diferentes números de componentes"""
        results = {}
        
        for n_components in [2, 5, 10, 15]:
            print(f"   🔄 PCA con {n_components} componentes...")
            embeddings_pca, pca = self._perform_pca_analysis(n_components)
            results[f'PCA_{n_components}D'] = {'embeddings': embeddings_pca, 'reducer': pca}
        
        return results
    
    def _calculate_advanced_similarity_metrics(self, embeddings):
        """Calcula métricas de similitud avanzadas"""
        results = {}
        
        # 1. Cosine Similarity
        print("      📊 Calculando Cosine Similarity...")
        cosine_sim = cosine_similarity(embeddings)
        results['cosine_similarity'] = cosine_sim
        
        # 2. Euclidean Distance
        print("      📏 Calculando Euclidean Distance...")
        euclidean_dist = euclidean_distances(embeddings)
        results['euclidean_distance'] = euclidean_dist
        
        # 3. Hybrid Distance (Cosine + Euclidean)
        print("      🔀 Calculando Hybrid Distance...")
        hybrid_dist = self._calculate_hybrid_distance(cosine_sim, euclidean_dist)
        results['hybrid_distance'] = hybrid_dist
        
        # 4. Estadísticas de separación
        print("      📈 Calculando estadísticas de separación...")
        separation_stats = self._calculate_separation_statistics(embeddings)
        results['separation_statistics'] = separation_stats
        
        return results
    
    def _calculate_hybrid_distance(self, cosine_sim, euclidean_dist):
        """Calcula distancia híbrida combinando cosine y euclidean"""
        # Normalizar ambas métricas a [0,1]
        cosine_norm = (cosine_sim + 1) / 2  # Convertir [-1,1] a [0,1]
        euclidean_norm = euclidean_dist / np.max(euclidean_dist)
        
        # Combinar con pesos
        alpha = 0.6  # Peso para cosine
        hybrid_dist = alpha * cosine_norm + (1 - alpha) * euclidean_norm
        
        return hybrid_dist
    
    def _calculate_separation_statistics(self, embeddings):
        """Calcula estadísticas de separación entre embeddings"""
        distances = euclidean_distances(embeddings)
        np.fill_diagonal(distances, np.inf)
        
        # Distancia mínima promedio
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)
        
        # Distancia máxima
        max_distance = np.max(distances)
        
        # Ratio de varianza
        variance_ratio = np.var(embeddings) / np.var(self.original_embeddings)
        
        # Silhouette score
        silhouette_score_val = self._calculate_silhouette_score(embeddings)
        
        return {
            'avg_min_distance': avg_min_distance,
            'max_distance': max_distance,
            'variance_ratio': variance_ratio,
            'silhouette_score': silhouette_score_val
        }
    
    def _calculate_silhouette_score(self, embeddings):
        """Calcula silhouette score para evaluar calidad de clustering"""
        try:
            # Determinar número óptimo de clusters
            n_clusters = min(8, len(embeddings) // 3)
            if n_clusters < 2:
                return 0.0
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            return silhouette_avg
        except Exception as e:
            print(f"         ⚠️ Error calculando silhouette: {e}")
            return 0.0
    
    def evaluate_all_methods(self, dimensionality_results, metrics_results):
        """Evalúa todos los métodos implementados"""
        print("\n🔍 EVALUANDO CALIDAD DE DISCRIMINACIÓN")
        print("-" * 50)
        
        evaluation_results = []
        
        # Evaluar embeddings originales como baseline
        baseline_eval = self._evaluate_discrimination_quality(
            self.original_embeddings, 
            'Original_Embeddings'
        )
        evaluation_results.append(baseline_eval)
        
        # Evaluar métodos de reducción de dimensionalidad
        for method_name, result in dimensionality_results.items():
            if 'embeddings' in result:
                eval_result = self._evaluate_discrimination_quality(
                    result['embeddings'], 
                    method_name
                )
                evaluation_results.append(eval_result)
        
        return evaluation_results
    
    def _evaluate_discrimination_quality(self, embeddings, method_name):
        """Evalúa la calidad de discriminación de los embeddings"""
        print(f"🔍 Evaluando: {method_name}")
        
        # Calcular métricas de separación
        distances = euclidean_distances(embeddings)
        np.fill_diagonal(distances, np.inf)
        
        # Distancia mínima promedio (mayor = mejor separación)
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)
        
        # Ratio de varianza (mayor = mejor separación)
        variance_ratio = np.var(embeddings) / np.var(self.original_embeddings)
        
        # Silhouette score aproximado
        silhouette_score_val = self._calculate_silhouette_score(embeddings)
        
        results = {
            'method': method_name,
            'avg_min_distance': avg_min_distance,
            'variance_ratio': variance_ratio,
            'silhouette_score': silhouette_score_val,
            'shape': embeddings.shape
        }
        
        print(f"   📊 Resultados:")
        print(f"      • Distancia mínima promedio: {avg_min_distance:.4f}")
        print(f"      • Ratio de varianza: {variance_ratio:.4f}")
        print(f"      • Silhouette score: {silhouette_score_val:.4f}")
        
        return results
    
    def visualize_comprehensive_results(self, dimensionality_results, metrics_results):
        """Visualiza todos los resultados de forma comprehensiva"""
        print("🎨 Generando visualizaciones comprehensivas...")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(3, 3, figsize=(24, 20))
        fig.suptitle('Sistema Inteligente - Estrategias 4 y 5', fontsize=20, fontweight='bold')
        
        # 1. Embeddings originales (primeras 2 dimensiones)
        axes[0, 0].scatter(self.original_embeddings[:, 0], self.original_embeddings[:, 1], 
                           alpha=0.7, s=50, c='blue')
        axes[0, 0].set_title('Original Embeddings (Dim 1 vs Dim 2)', fontweight='bold')
        axes[0, 0].set_xlabel('Dimensión 1')
        axes[0, 0].set_ylabel('Dimensión 2')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PCA 2D
        if 'PCA_2D' in dimensionality_results:
            embeddings_pca_2d = dimensionality_results['PCA_2D']['embeddings']
            axes[0, 1].scatter(embeddings_pca_2d[:, 0], embeddings_pca_2d[:, 1], 
                              alpha=0.7, s=50, c='red')
            axes[0, 1].set_title('PCA 2D', fontweight='bold')
            axes[0, 1].set_xlabel('Componente 1')
            axes[0, 1].set_ylabel('Componente 2')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. PCA 5D (primeras 2 dimensiones)
        if 'PCA_5D' in dimensionality_results:
            embeddings_pca_5d = dimensionality_results['PCA_5D']['embeddings']
            axes[0, 2].scatter(embeddings_pca_5d[:, 0], embeddings_pca_5d[:, 1], 
                              alpha=0.7, s=50, c='orange')
            axes[0, 2].set_title('PCA 5D (Comp 1 vs Comp 2)', fontweight='bold')
            axes[0, 2].set_xlabel('Componente 1')
            axes[0, 2].set_ylabel('Componente 2')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. PCA 10D (primeras 2 dimensiones)
        if 'PCA_10D' in dimensionality_results:
            embeddings_pca_10d = dimensionality_results['PCA_10D']['embeddings']
            axes[1, 0].scatter(embeddings_pca_10d[:, 0], embeddings_pca_10d[:, 1], 
                              alpha=0.7, s=50, c='purple')
            axes[1, 0].set_title('PCA 10D (Comp 1 vs Comp 2)', fontweight='bold')
            axes[1, 0].set_xlabel('Componente 1')
            axes[1, 0].set_ylabel('Componente 2')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. PCA 15D (primeras 2 dimensiones)
        if 'PCA_15D' in dimensionality_results:
            embeddings_pca_15d = dimensionality_results['PCA_15D']['embeddings']
            axes[1, 1].scatter(embeddings_pca_15d[:, 0], embeddings_pca_15d[:, 1], 
                              alpha=0.7, s=50, c='brown')
            axes[1, 1].set_title('PCA 15D (Comp 1 vs Comp 2)', fontweight='bold')
            axes[1, 1].set_xlabel('Componente 1')
            axes[1, 1].set_ylabel('Componente 2')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. t-SNE
        if 'TSNE_Optimized' in dimensionality_results:
            embeddings_tsne = dimensionality_results['TSNE_Optimized']['embeddings']
            axes[1, 2].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                              alpha=0.7, s=50, c='green')
            axes[1, 2].set_title('t-SNE Optimizado', fontweight='bold')
            axes[1, 2].set_xlabel('t-SNE 1')
            axes[1, 2].set_ylabel('t-SNE 2')
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Heatmap de similitud cosine (original)
        if 'Advanced_Metrics' in metrics_results:
            cosine_sim = metrics_results['Advanced_Metrics']['cosine_similarity']
            im1 = axes[2, 0].imshow(cosine_sim, cmap='RdBu_r', aspect='auto')
            axes[2, 0].set_title('Cosine Similarity (Original)', fontweight='bold')
            axes[2, 0].set_xlabel('Documento')
            axes[2, 0].set_ylabel('Documento')
            plt.colorbar(im1, ax=axes[2, 0])
        
        # 8. Heatmap de similitud cosine (PCA 2D)
        if 'PCA_2D_Metrics' in metrics_results:
            cosine_sim_pca = metrics_results['PCA_2D_Metrics']['cosine_similarity']
            im2 = axes[2, 1].imshow(cosine_sim_pca, cmap='RdBu_r', aspect='auto')
            axes[2, 1].set_title('Cosine Similarity (PCA 2D)', fontweight='bold')
            axes[2, 1].set_xlabel('Documento')
            axes[2, 1].set_ylabel('Documento')
            plt.colorbar(im2, ax=axes[2, 1])
        
        # 9. Comparación de métricas
        if 'Advanced_Metrics' in metrics_results:
            sep_stats = metrics_results['Advanced_Metrics']['separation_statistics']
            axes[2, 2].bar(['Min Dist', 'Max Dist', 'Var Ratio'], 
                          [sep_stats['avg_min_distance'], sep_stats['max_distance'], 
                           sep_stats['variance_ratio']], 
                          color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[2, 2].set_title('Estadísticas de Separación (Original)', fontweight='bold')
            axes[2, 2].set_ylabel('Valor')
            axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('working_intelligent_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizaciones guardadas en 'working_intelligent_results.png'")
    
    def generate_comprehensive_report(self, dimensionality_results, metrics_results, evaluation_results):
        """Genera reporte completo del análisis"""
        print("📊 Generando reporte comprehensivo...")
        
        with open('working_intelligent_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE COMPREHENSIVO - SISTEMA INTELIGENTE\n")
            f.write("Estrategias 4 y 5 Implementadas Exitosamente\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("ESTRATEGIA 4: REDUCCIÓN DE DIMENSIONALIDAD INTELIGENTE\n")
            f.write("-" * 50 + "\n")
            
            for method_name, result in dimensionality_results.items():
                if 'embeddings' in result:
                    f.write(f"\n{method_name}:\n")
                    f.write(f"  • Forma: {result['embeddings'].shape}\n")
                    if 'reducer' in result and hasattr(result['reducer'], 'explained_variance_ratio_'):
                        var_explained = np.sum(result['reducer'].explained_variance_ratio_) * 100
                        f.write(f"  • Varianza explicada: {var_explained:.2f}%\n")
            
            f.write("\nESTRATEGIA 5: MÉTRICAS DE SIMILITUD AVANZADAS\n")
            f.write("-" * 50 + "\n")
            
            for method_name, result in metrics_results.items():
                f.write(f"\n{method_name}:\n")
                if 'separation_statistics' in result:
                    stats = result['separation_statistics']
                    f.write(f"  • Distancia mínima promedio: {stats['avg_min_distance']:.4f}\n")
                    f.write(f"  • Distancia máxima: {stats['max_distance']:.4f}\n")
                    f.write(f"  • Ratio de varianza: {stats['variance_ratio']:.4f}\n")
                    f.write(f"  • Silhouette score: {stats['silhouette_score']:.4f}\n")
            
            f.write("\nEVALUACIÓN COMPARATIVA DE MÉTODOS:\n")
            f.write("-" * 40 + "\n")
            
            for eval_result in evaluation_results:
                f.write(f"\n{eval_result['method']}:\n")
                f.write(f"  • Distancia mínima promedio: {eval_result['avg_min_distance']:.4f}\n")
                f.write(f"  • Ratio de varianza: {eval_result['variance_ratio']:.4f}\n")
                f.write(f"  • Silhouette score: {eval_result['silhouette_score']:.4f}\n")
                f.write(f"  • Forma: {eval_result['shape']}\n")
            
            f.write("\nRECOMENDACIONES:\n")
            f.write("-" * 20 + "\n")
            f.write("1. Usar método con mayor silhouette score para discriminación\n")
            f.write("2. Considerar PCA 2D para visualización y análisis rápido\n")
            f.write("3. Usar t-SNE para visualización 2D de alta calidad\n")
            f.write("4. Evaluar métricas híbridas para mejor rendimiento\n")
            f.write("5. Comparar con embeddings originales como baseline\n")
        
        print("✅ Reporte generado: working_intelligent_report.txt")

def main():
    """Función principal del sistema inteligente funcional"""
    print("🚀 INICIANDO SISTEMA INTELIGENTE FUNCIONAL")
    print("=" * 80)
    print("Implementando Estrategias 4 y 5:")
    print("• Estrategia 4: Técnicas de Reducción de Dimensionalidad Inteligente")
    print("• Estrategia 5: Métricas de Similitud Avanzadas")
    print("=" * 80)
    
    # Crear sistema inteligente
    system = WorkingIntelligentSystem()
    
    # Cargar modelo y datos
    if not system.load_model():
        return
    
    if not system.load_pnts_documents():
        return
    
    if not system.load_benchmark_data():
        return
    
    if not system.generate_embeddings():
        return
    
    # Implementar Estrategia 4
    dimensionality_results = system.implement_strategy_4_dimensionality_reduction()
    
    # Implementar Estrategia 5
    metrics_results = system.implement_strategy_5_advanced_metrics()
    
    # Evaluar todos los métodos
    evaluation_results = system.evaluate_all_methods(dimensionality_results, metrics_results)
    
    # Visualizar resultados
    system.visualize_comprehensive_results(dimensionality_results, metrics_results)
    
    # Generar reporte
    system.generate_comprehensive_report(dimensionality_results, metrics_results, evaluation_results)
    
    print("\n🎉 SISTEMA INTELIGENTE COMPLETADO EXITOSAMENTE!")
    print("📁 Archivos generados:")
    print("   - working_intelligent_results.png")
    print("   - working_intelligent_report.txt")
    
    # Mostrar resumen de resultados
    print("\n📊 RESUMEN DE RESULTADOS:")
    print("-" * 30)
    for eval_result in evaluation_results:
        print(f"   {eval_result['method']}: Silhouette={eval_result['silhouette_score']:.4f}, "
              f"Var_Ratio={eval_result['variance_ratio']:.4f}")

if __name__ == "__main__":
    main()
