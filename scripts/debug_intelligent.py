#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión de Debug para Identificar Problemas
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

def main():
    """Función principal con debug paso a paso"""
    print("🚀 INICIANDO DEBUG - ESTRATEGIAS 4 Y 5")
    print("=" * 80)
    
    try:
        print("PASO 1: Creando analizador...")
        model_path = "all-mini-base"
        pnts_folder = "PNTs"
        benchmark_folder = "benchmark"
        
        print("PASO 2: Cargando modelo...")
        model = SentenceTransformer(model_path)
        print(f"✅ Modelo cargado: {model.get_sentence_embedding_dimension()} dimensiones")
        
        print("PASO 3: Cargando documentos PNTs...")
        if not os.path.exists(pnts_folder):
            print(f"❌ Error: Carpeta {pnts_folder} no encontrada")
            return
        
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
        
        print("PASO 4: Cargando benchmark...")
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
        
        print("PASO 5: Generando embeddings...")
        original_embeddings = model.encode(pnts_documents, show_progress_bar=True)
        print(f"✅ Embeddings generados: {original_embeddings.shape}")
        
        print("PASO 6: Aplicando PCA...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(original_embeddings)
        
        pca = PCA(n_components=10, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        print(f"✅ PCA completado: {embeddings_pca.shape}")
        print(f"📊 Varianza explicada: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        print("PASO 7: Aplicando t-SNE...")
        tsne = TSNE(n_components=2, perplexity=10, random_state=42, n_jobs=-1, verbose=1)
        embeddings_tsne = tsne.fit_transform(embeddings_scaled)
        print(f"✅ t-SNE completado: {embeddings_tsne.shape}")
        
        print("PASO 8: Calculando métricas de similitud...")
        cosine_sim = cosine_similarity(original_embeddings)
        euclidean_dist = euclidean_distances(original_embeddings)
        print(f"✅ Métricas calculadas: cosine={cosine_sim.shape}, euclidean={euclidean_dist.shape}")
        
        print("PASO 9: Evaluando calidad de discriminación...")
        # PCA
        distances_pca = euclidean_distances(embeddings_pca)
        np.fill_diagonal(distances_pca, np.inf)
        min_distances_pca = np.min(distances_pca, axis=1)
        avg_min_distance_pca = np.mean(min_distances_pca)
        variance_ratio_pca = np.var(embeddings_pca) / np.var(original_embeddings)
        
        # t-SNE
        distances_tsne = euclidean_distances(embeddings_tsne)
        np.fill_diagonal(distances_tsne, np.inf)
        min_distances_tsne = np.min(distances_tsne, axis=1)
        avg_min_distance_tsne = np.mean(min_distances_tsne)
        variance_ratio_tsne = np.var(embeddings_tsne) / np.var(original_embeddings)
        
        # Original
        distances_orig = euclidean_distances(original_embeddings)
        np.fill_diagonal(distances_orig, np.inf)
        min_distances_orig = np.min(distances_orig, axis=1)
        avg_min_distance_orig = np.mean(min_distances_orig)
        variance_ratio_orig = 1.0
        
        print("📊 RESULTADOS DE EVALUACIÓN:")
        print(f"   PCA: Dist_Min={avg_min_distance_pca:.4f}, Var_Ratio={variance_ratio_pca:.4f}")
        print(f"   t-SNE: Dist_Min={avg_min_distance_tsne:.4f}, Var_Ratio={variance_ratio_tsne:.4f}")
        print(f"   Original: Dist_Min={avg_min_distance_orig:.4f}, Var_Ratio={variance_ratio_orig:.4f}")
        
        print("PASO 10: Generando visualizaciones...")
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle('Debug - Estrategias 4 y 5', fontsize=20, fontweight='bold')
        
        # Original embeddings (primeras 2 dimensiones)
        axes[0].scatter(original_embeddings[:, 0], original_embeddings[:, 1], alpha=0.7, s=50, c='blue')
        axes[0].set_title('Original Embeddings (Dim 1 vs Dim 2)', fontweight='bold')
        axes[0].set_xlabel('Dimensión 1')
        axes[0].set_ylabel('Dimensión 2')
        axes[0].grid(True, alpha=0.3)
        
        # PCA
        axes[1].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.7, s=50, c='red')
        axes[1].set_title('PCA (Comp 1 vs Comp 2)', fontweight='bold')
        axes[1].set_xlabel('Componente 1')
        axes[1].set_ylabel('Componente 2')
        axes[1].grid(True, alpha=0.3)
        
        # t-SNE
        axes[2].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.7, s=50, c='green')
        axes[2].set_title('t-SNE', fontweight='bold')
        axes[2].set_xlabel('t-SNE 1')
        axes[2].set_ylabel('t-SNE 2')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('debug_intelligent_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("PASO 11: Generando reporte...")
        with open('debug_intelligent_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE DEBUG - ESTRATEGIAS 4 Y 5\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("RESUMEN DE RESULTADOS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"• Embeddings originales: {original_embeddings.shape}\n")
            f.write(f"• PCA (10 componentes): {embeddings_pca.shape}\n")
            f.write(f"• t-SNE (2 componentes): {embeddings_tsne.shape}\n")
            f.write(f"• Varianza explicada PCA: {np.sum(pca.explained_variance_ratio_)*100:.2f}%\n\n")
            
            f.write("EVALUACIÓN DE CALIDAD:\n")
            f.write("-" * 30 + "\n")
            f.write(f"PCA: Dist_Min={avg_min_distance_pca:.4f}, Var_Ratio={variance_ratio_pca:.4f}\n")
            f.write(f"t-SNE: Dist_Min={avg_min_distance_tsne:.4f}, Var_Ratio={variance_ratio_tsne:.4f}\n")
            f.write(f"Original: Dist_Min={avg_min_distance_orig:.4f}, Var_Ratio={variance_ratio_orig:.4f}\n")
        
        print("🎉 DEBUG COMPLETADO EXITOSAMENTE!")
        print("📁 Archivos generados:")
        print("   - debug_intelligent_results.png")
        print("   - debug_intelligent_report.txt")
        
    except Exception as e:
        print(f"❌ ERROR EN EL PROCESO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
