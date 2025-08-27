#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión Simplificada del Sistema de Reducción de Dimensionalidad Inteligente
Autor: Análisis de Embeddings Médicos
Fecha: 2025
Objetivo: Probar Estrategias 4 y 5 de forma simplificada
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
import json
warnings.filterwarnings('ignore')

# Configuración de matplotlib para español
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class SimpleIntelligentReducer:
    """Versión simplificada del sistema inteligente"""
    
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
    
    def perform_pca_analysis(self, n_components=10):
        """Realiza análisis PCA como baseline"""
        print(f"🔄 Realizando PCA con {n_components} componentes...")
        
        # Normalización
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.original_embeddings)
        
        # Aplicar PCA
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        
        print(f"✅ PCA completado: {embeddings_pca.shape}")
        print(f"📊 Varianza explicada: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        return embeddings_pca, pca
    
    def perform_tsne_analysis(self, n_components=2, perplexity=10):
        """Realiza análisis t-SNE optimizado"""
        print(f"🔄 Realizando t-SNE con {n_components} componentes y perplexity {perplexity}...")
        
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
        
        print(f"✅ t-SNE completado: {embeddings_tsne.shape}")
        return embeddings_tsne, tsne
    
    def calculate_advanced_similarity_metrics(self, embeddings):
        """Calcula métricas de similitud avanzadas"""
        print("🔄 Calculando métricas de similitud avanzadas...")
        
        results = {}
        
        # 1. Cosine Similarity
        print("   📊 Calculando Cosine Similarity...")
        cosine_sim = cosine_similarity(embeddings)
        results['cosine_similarity'] = cosine_sim
        
        # 2. Euclidean Distance
        print("   📏 Calculando Euclidean Distance...")
        euclidean_dist = euclidean_distances(embeddings)
        results['euclidean_distance'] = euclidean_dist
        
        # 3. Hybrid Distance (Cosine + Euclidean)
        print("   🔀 Calculando Hybrid Distance...")
        hybrid_dist = self._calculate_hybrid_distance(cosine_sim, euclidean_dist)
        results['hybrid_distance'] = hybrid_dist
        
        print("✅ Métricas de similitud calculadas")
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
    
    def evaluate_discrimination_quality(self, embeddings, method_name):
        """Evalúa la calidad de discriminación de los embeddings"""
        print(f"🔍 Evaluando calidad de discriminación: {method_name}")
        
        # Calcular métricas de separación
        distances = euclidean_distances(embeddings)
        np.fill_diagonal(distances, np.inf)
        
        # Distancia mínima promedio (mayor = mejor separación)
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)
        
        # Ratio de varianza (mayor = mejor separación)
        variance_ratio = np.var(embeddings) / np.var(self.original_embeddings)
        
        # Silhouette score aproximado
        silhouette_score = self._calculate_silhouette_score(embeddings)
        
        results = {
            'method': method_name,
            'avg_min_distance': avg_min_distance,
            'variance_ratio': variance_ratio,
            'silhouette_score': silhouette_score,
            'shape': embeddings.shape
        }
        
        print(f"   📊 Resultados:")
        print(f"      • Distancia mínima promedio: {avg_min_distance:.4f}")
        print(f"      • Ratio de varianza: {variance_ratio:.4f}")
        print(f"      • Silhouette score: {silhouette_score:.4f}")
        
        return results
    
    def _calculate_silhouette_score(self, embeddings):
        """Calcula un score de silhouette aproximado"""
        try:
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import KMeans
            
            # Determinar número óptimo de clusters
            n_clusters = min(8, len(embeddings) // 3)
            if n_clusters < 2:
                return 0.0
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            return silhouette_avg
        except Exception as e:
            print(f"      ⚠️ Error calculando silhouette: {e}")
            return 0.0
    
    def visualize_results(self, results_dict):
        """Visualiza todos los resultados"""
        print("🎨 Generando visualizaciones...")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Análisis de Reducción de Dimensionalidad Inteligente (Simplificado)', 
                     fontsize=20, fontweight='bold')
        
        methods = list(results_dict.keys())
        
        for i, method in enumerate(methods[:4]):  # Máximo 4 métodos
            row = i // 2
            col = i % 2
            
            if method in results_dict and 'embeddings' in results_dict[method]:
                embeddings = results_dict[method]['embeddings']
                
                if embeddings.shape[1] >= 2:
                    # Visualización 2D
                    axes[row, col].scatter(embeddings[:, 0], embeddings[:, 1], 
                                         alpha=0.7, s=50, c='blue')
                    axes[row, col].set_title(f'{method}', fontweight='bold')
                    axes[row, col].set_xlabel('Dimensión 1')
                    axes[row, col].set_ylabel('Dimensión 2')
                    axes[row, col].grid(True, alpha=0.3)
                else:
                    # Visualización 1D
                    axes[row, col].hist(embeddings.flatten(), bins=20, alpha=0.7, color='blue')
                    axes[row, col].set_title(f'{method} (1D)', fontweight='bold')
                    axes[row, col].set_xlabel('Valor')
                    axes[row, col].set_ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.savefig('simple_intelligent_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizaciones guardadas en 'simple_intelligent_results.png'")
    
    def generate_report(self, results_dict, evaluation_results):
        """Genera reporte del análisis"""
        print("📊 Generando reporte...")
        
        with open('simple_intelligent_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE SIMPLIFICADO - ESTRATEGIAS 4 Y 5\n")
            f.write("Reducción de Dimensionalidad Inteligente y Métricas Avanzadas\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("RESUMEN DE MÉTODOS IMPLEMENTADOS:\n")
            f.write("-" * 40 + "\n")
            
            for method, result in results_dict.items():
                if 'embeddings' in result:
                    f.write(f"\n{method}:\n")
                    f.write(f"  • Forma: {result['embeddings'].shape}\n")
            
            f.write("\nEVALUACIÓN DE CALIDAD:\n")
            f.write("-" * 30 + "\n")
            
            for eval_result in evaluation_results:
                f.write(f"\n{eval_result['method']}:\n")
                f.write(f"  • Distancia mínima promedio: {eval_result['avg_min_distance']:.4f}\n")
                f.write(f"  • Ratio de varianza: {eval_result['variance_ratio']:.4f}\n")
                f.write(f"  • Silhouette score: {eval_result['silhouette_score']:.4f}\n")
            
            f.write("\nRECOMENDACIONES:\n")
            f.write("-" * 20 + "\n")
            f.write("1. Usar método con mayor silhouette score para discriminación\n")
            f.write("2. Considerar t-SNE para visualización 2D\n")
            f.write("3. Evaluar métricas híbridas para mejor rendimiento\n")
            f.write("4. Comparar con embeddings originales como baseline\n")
        
        print("✅ Reporte generado: simple_intelligent_report.txt")

def main():
    """Función principal simplificada"""
    print("🚀 INICIANDO ANÁLISIS SIMPLIFICADO - ESTRATEGIAS 4 Y 5")
    print("=" * 80)
    print("Implementando:")
    print("• Estrategia 4: Reducción de Dimensionalidad Inteligente (PCA + t-SNE)")
    print("• Estrategia 5: Métricas de Similitud Avanzadas (Cosine + Euclidean + Hybrid)")
    print("=" * 80)
    
    # Crear analizador
    analyzer = SimpleIntelligentReducer()
    
    # Cargar modelo y datos
    if not analyzer.load_model():
        return
    
    if not analyzer.load_pnts_documents():
        return
    
    if not analyzer.load_benchmark_data():
        return
    
    if not analyzer.generate_embeddings():
        return
    
    # ESTRATEGIA 4: Reducción de Dimensionalidad Inteligente
    print("\n" + "="*60)
    print("🔬 ESTRATEGIA 4: REDUCCIÓN DE DIMENSIONALIDAD INTELIGENTE")
    print("="*60)
    
    results = {}
    evaluation_results = []
    
    # 1. PCA como baseline
    print("\n📊 1. ANÁLISIS PCA (Baseline)")
    embeddings_pca, pca_reducer = analyzer.perform_pca_analysis(n_components=10)
    results['PCA_Baseline'] = {'embeddings': embeddings_pca, 'reducer': pca_reducer}
    
    # 2. t-SNE Optimizado
    print("\n📊 2. ANÁLISIS t-SNE OPTIMIZADO")
    embeddings_tsne, tsne_reducer = analyzer.perform_tsne_analysis(perplexity=10)
    results['TSNE_Optimized'] = {'embeddings': embeddings_tsne, 'reducer': tsne_reducer}
    
    # ESTRATEGIA 5: Métricas de Similitud Avanzadas
    print("\n" + "="*60)
    print("🔬 ESTRATEGIA 5: MÉTRICAS DE SIMILITUD AVANZADAS")
    print("="*60)
    
    # Calcular métricas avanzadas para embeddings originales
    advanced_metrics = analyzer.calculate_advanced_similarity_metrics(analyzer.original_embeddings)
    results['Advanced_Metrics'] = advanced_metrics
    
    # Evaluar calidad de discriminación
    print("\n🔍 EVALUANDO CALIDAD DE DISCRIMINACIÓN")
    print("-" * 50)
    
    for method_name in ['PCA_Baseline', 'TSNE_Optimized']:
        if method_name in results and 'embeddings' in results[method_name]:
            eval_result = analyzer.evaluate_discrimination_quality(
                results[method_name]['embeddings'], 
                method_name
            )
            evaluation_results.append(eval_result)
    
    # Evaluar embeddings originales como baseline
    baseline_eval = analyzer.evaluate_discrimination_quality(
        analyzer.original_embeddings, 
        'Original_Embeddings'
    )
    evaluation_results.append(baseline_eval)
    
    # Visualizar resultados
    analyzer.visualize_results(results)
    
    # Generar reporte
    analyzer.generate_report(results, evaluation_results)
    
    print("\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE!")
    print("📁 Archivos generados:")
    print("   - simple_intelligent_results.png")
    print("   - simple_intelligent_report.txt")
    
    # Mostrar resumen de resultados
    print("\n📊 RESUMEN DE RESULTADOS:")
    print("-" * 30)
    for eval_result in evaluation_results:
        print(f"   {eval_result['method']}: Silhouette={eval_result['silhouette_score']:.4f}, "
              f"Var_Ratio={eval_result['variance_ratio']:.4f}")

if __name__ == "__main__":
    main()
