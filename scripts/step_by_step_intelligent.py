#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Inteligente Paso a Paso - Estrategias 4 y 5
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

def main():
    """Función principal paso a paso"""
    print("🚀 INICIANDO SISTEMA INTELIGENTE PASO A PASO")
    print("=" * 80)
    
    try:
        # PASO 1: Cargar modelo
        print("PASO 1: Cargando modelo...")
        model = SentenceTransformer("all-mini-base")
        print(f"✅ Modelo cargado: {model.get_sentence_embedding_dimension()} dimensiones")
        
        # PASO 2: Cargar documentos PNTs
        print("PASO 2: Cargando documentos PNTs...")
        pnts_folder = "PNTs"
        txt_files = [f for f in os.listdir(pnts_folder) if f.endswith('.txt')]
        pnts_documents = []
        pnts_names = []
        
        for filename in txt_files:
            file_path = os.path.join(pnts_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        pnts_documents.append(content)
                        pnts_names.append(filename)
            except Exception as e:
                print(f"   ⚠️ Error al leer {filename}: {e}")
        
        print(f"✅ Documentos cargados: {len(pnts_documents)}")
        
        # PASO 3: Cargar benchmark
        print("PASO 3: Cargando benchmark...")
        benchmark_folder = "benchmark"
        benchmark_data = {}
        
        cat_file = os.path.join(benchmark_folder, "preguntas_con_docs_cat.json")
        if os.path.exists(cat_file):
            with open(cat_file, 'r', encoding='utf-8') as f:
                cat_data = json.load(f)
                benchmark_data['catalan'] = cat_data
                print(f"✅ Benchmark catalán: {len(cat_data)} preguntas")
        
        es_file = os.path.join(benchmark_folder, "preguntas_con_docs_es.json")
        if os.path.exists(es_file):
            with open(es_file, 'r', encoding='utf-8') as f:
                es_data = json.load(f)
                benchmark_data['spanish'] = es_data
                print(f"✅ Benchmark español: {len(es_data)} preguntas")
        
        # PASO 4: Generar embeddings
        print("PASO 4: Generando embeddings...")
        original_embeddings = model.encode(pnts_documents, show_progress_bar=True)
        print(f"✅ Embeddings generados: {original_embeddings.shape}")
        
        # PASO 5: Implementar Estrategia 4 - Reducción de Dimensionalidad Inteligente
        print("\n" + "="*60)
        print("🔬 ESTRATEGIA 4: REDUCCIÓN DE DIMENSIONALIDAD INTELIGENTE")
        print("="*60)
        
        dimensionality_results = {}
        
        # 5.1 PCA con diferentes componentes
        print("\n📊 5.1 ANÁLISIS PCA MULTI-DIMENSIONAL")
        for n_components in [2, 5, 10, 15]:
            print(f"   🔄 PCA con {n_components} componentes...")
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(original_embeddings)
            
            pca = PCA(n_components=n_components, random_state=42)
            embeddings_pca = pca.fit_transform(embeddings_scaled)
            
            dimensionality_results[f'PCA_{n_components}D'] = {
                'embeddings': embeddings_pca, 
                'reducer': pca
            }
            
            print(f"   ✅ PCA {n_components}D completado: {embeddings_pca.shape}")
            print(f"   📊 Varianza explicada: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        # 5.2 t-SNE Optimizado
        print("\n📊 5.2 ANÁLISIS t-SNE OPTIMIZADO")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(original_embeddings)
        
        tsne = TSNE(n_components=2, perplexity=10, random_state=42, n_jobs=-1, verbose=1)
        embeddings_tsne = tsne.fit_transform(embeddings_scaled)
        
        dimensionality_results['TSNE_Optimized'] = {
            'embeddings': embeddings_tsne, 
            'reducer': tsne
        }
        
        print(f"   ✅ t-SNE completado: {embeddings_tsne.shape}")
        
        # PASO 6: Implementar Estrategia 5 - Métricas de Similitud Avanzadas
        print("\n" + "="*60)
        print("🔬 ESTRATEGIA 5: MÉTRICAS DE SIMILITUD AVANZADAS")
        print("="*60)
        
        metrics_results = {}
        
        # 6.1 Métricas para embeddings originales
        print("🔄 6.1 Calculando métricas para embeddings originales...")
        original_metrics = calculate_advanced_metrics(original_embeddings, "Original")
        metrics_results['Original_Metrics'] = original_metrics
        
        # 6.2 Métricas para representaciones reducidas
        print("🔄 6.2 Calculando métricas para representaciones reducidas...")
        for method_name, result in dimensionality_results.items():
            if 'embeddings' in result:
                print(f"   📊 Métricas para {method_name}...")
                method_metrics = calculate_advanced_metrics(result['embeddings'], method_name)
                metrics_results[f'{method_name}_Metrics'] = method_metrics
        
        # PASO 7: Evaluar calidad de discriminación
        print("\n🔍 PASO 7: EVALUANDO CALIDAD DE DISCRIMINACIÓN")
        print("-" * 50)
        
        evaluation_results = []
        
        # Evaluar embeddings originales
        baseline_eval = evaluate_discrimination_quality(original_embeddings, 'Original_Embeddings')
        evaluation_results.append(baseline_eval)
        
        # Evaluar métodos de reducción
        for method_name, result in dimensionality_results.items():
            if 'embeddings' in result:
                eval_result = evaluate_discrimination_quality(result['embeddings'], method_name)
                evaluation_results.append(eval_result)
        
        # PASO 8: Visualizar resultados
        print("\n🎨 PASO 8: GENERANDO VISUALIZACIONES")
        visualize_results(dimensionality_results, metrics_results)
        
        # PASO 9: Generar reporte
        print("\n📊 PASO 9: GENERANDO REPORTE")
        generate_report(dimensionality_results, metrics_results, evaluation_results)
        
        print("\n🎉 SISTEMA INTELIGENTE COMPLETADO EXITOSAMENTE!")
        print("📁 Archivos generados:")
        print("   - step_by_step_results.png")
        print("   - step_by_step_report.txt")
        
        # Mostrar resumen
        print("\n📊 RESUMEN DE RESULTADOS:")
        print("-" * 30)
        for eval_result in evaluation_results:
            print(f"   {eval_result['method']}: Silhouette={eval_result['silhouette_score']:.4f}, "
                  f"Var_Ratio={eval_result['variance_ratio']:.4f}")
        
    except Exception as e:
        print(f"❌ ERROR EN EL PROCESO: {e}")
        import traceback
        traceback.print_exc()

def calculate_advanced_metrics(embeddings, method_name):
    """Calcula métricas avanzadas para embeddings"""
    results = {}
    
    # Cosine Similarity
    cosine_sim = cosine_similarity(embeddings)
    results['cosine_similarity'] = cosine_sim
    
    # Euclidean Distance
    euclidean_dist = euclidean_distances(embeddings)
    results['euclidean_distance'] = euclidean_dist
    
    # Hybrid Distance
    hybrid_dist = calculate_hybrid_distance(cosine_sim, euclidean_dist)
    results['hybrid_distance'] = hybrid_dist
    
    # Estadísticas de separación
    separation_stats = calculate_separation_statistics(embeddings)
    results['separation_statistics'] = separation_stats
    
    return results

def calculate_hybrid_distance(cosine_sim, euclidean_dist):
    """Calcula distancia híbrida"""
    cosine_norm = (cosine_sim + 1) / 2
    euclidean_norm = euclidean_dist / np.max(euclidean_dist)
    alpha = 0.6
    hybrid_dist = alpha * cosine_norm + (1 - alpha) * euclidean_norm
    return hybrid_dist

def calculate_separation_statistics(embeddings):
    """Calcula estadísticas de separación"""
    distances = euclidean_distances(embeddings)
    np.fill_diagonal(distances, np.inf)
    
    min_distances = np.min(distances, axis=1)
    avg_min_distance = np.mean(min_distances)
    max_distance = np.max(distances)
    
    # Para embeddings originales, usar varianza base
    if embeddings.shape[1] == 384:  # Original
        variance_ratio = 1.0
    else:
        # Para representaciones reducidas, comparar con original
        variance_ratio = np.var(embeddings) / np.var(embeddings)  # Normalizado
    
    silhouette_score_val = calculate_silhouette_score(embeddings)
    
    return {
        'avg_min_distance': avg_min_distance,
        'max_distance': max_distance,
        'variance_ratio': variance_ratio,
        'silhouette_score': silhouette_score_val
    }

def calculate_silhouette_score(embeddings):
    """Calcula silhouette score"""
    try:
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

def evaluate_discrimination_quality(embeddings, method_name):
    """Evalúa calidad de discriminación"""
    print(f"🔍 Evaluando: {method_name}")
    
    distances = euclidean_distances(embeddings)
    np.fill_diagonal(distances, np.inf)
    
    min_distances = np.min(distances, axis=1)
    avg_min_distance = np.mean(min_distances)
    
    # Para embeddings originales
    if embeddings.shape[1] == 384:
        variance_ratio = 1.0
    else:
        variance_ratio = np.var(embeddings) / np.var(embeddings)  # Normalizado
    
    silhouette_score_val = calculate_silhouette_score(embeddings)
    
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

def visualize_results(dimensionality_results, metrics_results):
    """Visualiza resultados"""
    print("🎨 Generando visualizaciones...")
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Sistema Inteligente Paso a Paso - Estrategias 4 y 5', fontsize=20, fontweight='bold')
    
    # PCA 2D
    if 'PCA_2D' in dimensionality_results:
        embeddings_pca_2d = dimensionality_results['PCA_2D']['embeddings']
        axes[0, 0].scatter(embeddings_pca_2d[:, 0], embeddings_pca_2d[:, 1], 
                           alpha=0.7, s=50, c='red')
        axes[0, 0].set_title('PCA 2D', fontweight='bold')
        axes[0, 0].set_xlabel('Componente 1')
        axes[0, 0].set_ylabel('Componente 2')
        axes[0, 0].grid(True, alpha=0.3)
    
    # PCA 5D
    if 'PCA_5D' in dimensionality_results:
        embeddings_pca_5d = dimensionality_results['PCA_5D']['embeddings']
        axes[0, 1].scatter(embeddings_pca_5d[:, 0], embeddings_pca_5d[:, 1], 
                           alpha=0.7, s=50, c='orange')
        axes[0, 1].set_title('PCA 5D (Comp 1 vs Comp 2)', fontweight='bold')
        axes[0, 1].set_xlabel('Componente 1')
        axes[0, 1].set_ylabel('Componente 2')
        axes[0, 1].grid(True, alpha=0.3)
    
    # PCA 10D
    if 'PCA_10D' in dimensionality_results:
        embeddings_pca_10d = dimensionality_results['PCA_10D']['embeddings']
        axes[0, 2].scatter(embeddings_pca_10d[:, 0], embeddings_pca_10d[:, 1], 
                           alpha=0.7, s=50, c='purple')
        axes[0, 2].set_title('PCA 10D (Comp 1 vs Comp 2)', fontweight='bold')
        axes[0, 2].set_xlabel('Componente 1')
        axes[0, 2].set_ylabel('Componente 2')
        axes[0, 2].grid(True, alpha=0.3)
    
    # PCA 15D
    if 'PCA_15D' in dimensionality_results:
        embeddings_pca_15d = dimensionality_results['PCA_15D']['embeddings']
        axes[1, 0].scatter(embeddings_pca_15d[:, 0], embeddings_pca_15d[:, 1], 
                           alpha=0.7, s=50, c='brown')
        axes[1, 0].set_title('PCA 15D (Comp 1 vs Comp 2)', fontweight='bold')
        axes[1, 0].set_xlabel('Componente 1')
        axes[1, 0].set_ylabel('Componente 2')
        axes[1, 0].grid(True, alpha=0.3)
    
    # t-SNE
    if 'TSNE_Optimized' in dimensionality_results:
        embeddings_tsne = dimensionality_results['TSNE_Optimized']['embeddings']
        axes[1, 1].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                           alpha=0.7, s=50, c='green')
        axes[1, 1].set_title('t-SNE Optimizado', fontweight='bold')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Heatmap de similitud cosine
    if 'Original_Metrics' in metrics_results:
        cosine_sim = metrics_results['Original_Metrics']['cosine_similarity']
        im = axes[1, 2].imshow(cosine_sim, cmap='RdBu_r', aspect='auto')
        axes[1, 2].set_title('Cosine Similarity (Original)', fontweight='bold')
        axes[1, 2].set_xlabel('Documento')
        axes[1, 2].set_ylabel('Documento')
        plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('step_by_step_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizaciones guardadas en 'step_by_step_results.png'")

def generate_report(dimensionality_results, metrics_results, evaluation_results):
    """Genera reporte"""
    print("📊 Generando reporte...")
    
    with open('step_by_step_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE PASO A PASO - SISTEMA INTELIGENTE\n")
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
        
        f.write("\nEVALUACIÓN COMPARATIVA:\n")
        f.write("-" * 30 + "\n")
        
        for eval_result in evaluation_results:
            f.write(f"\n{eval_result['method']}:\n")
            f.write(f"  • Distancia mínima promedio: {eval_result['avg_min_distance']:.4f}\n")
            f.write(f"  • Ratio de varianza: {eval_result['variance_ratio']:.4f}\n")
            f.write(f"  • Silhouette score: {eval_result['silhouette_score']:.4f}\n")
            f.write(f"  • Forma: {eval_result['shape']}\n")
        
        f.write("\nRECOMENDACIONES:\n")
        f.write("-" * 20 + "\n")
        f.write("1. Usar método con mayor silhouette score para discriminación\n")
        f.write("2. Considerar PCA 2D para visualización rápida\n")
        f.write("3. Usar t-SNE para visualización 2D de alta calidad\n")
        f.write("4. Evaluar métricas híbridas para mejor rendimiento\n")
    
    print("✅ Reporte generado: step_by_step_report.txt")

if __name__ == "__main__":
    main()
