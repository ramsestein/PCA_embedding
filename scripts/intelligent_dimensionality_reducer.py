#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reductor de Dimensionalidad Inteligente con Métricas Avanzadas
Autor: Análisis de Embeddings Médicos
Fecha: 2025
Objetivo: Implementar estrategias 4 y 5 para mejorar discriminación
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
import ot
warnings.filterwarnings('ignore')

# Configuración de matplotlib para español
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class IntelligentDimensionalityReducer:
    """Sistema inteligente de reducción de dimensionalidad con métricas avanzadas"""
    
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
    
    def perform_umap_analysis(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine'):
        """Realiza análisis UMAP optimizado"""
        print(f"🔄 Realizando UMAP con {n_components} componentes...")
        print(f"   Parámetros: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
        
        # Normalización
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.original_embeddings)
        
        # UMAP
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
            verbose=True
        )
        
        embeddings_umap = umap_reducer.fit_transform(embeddings_scaled)
        
        print(f"✅ UMAP completado: {embeddings_umap.shape}")
        return embeddings_umap, umap_reducer
    
    def perform_optimized_tsne(self, n_components=2, perplexity_range=(5, 50), learning_rate_range=(10, 1000)):
        """Realiza t-SNE con parámetros optimizados"""
        print(f"🔄 Realizando t-SNE optimizado con {n_components} componentes...")
        
        # Normalización
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.original_embeddings)
        
        # Grid search de parámetros
        best_tsne = None
        best_score = float('inf')
        best_params = {}
        
        perplexities = np.linspace(perplexity_range[0], perplexity_range[1], 5)
        learning_rates = np.logspace(np.log10(learning_rate_range[0]), np.log10(learning_rate_range[1]), 5)
        
        print("   🔍 Buscando mejores parámetros...")
        
        for perplexity in perplexities:
            for lr in learning_rates:
                try:
                    tsne = TSNE(
                        n_components=n_components,
                        perplexity=perplexity,
                        learning_rate=lr,
                        random_state=42,
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    embeddings_tsne = tsne.fit_transform(embeddings_scaled)
                    
                    # Calcular score de calidad (menor = mejor)
                    # Usar la suma de distancias euclidianas entre puntos cercanos
                    distances = euclidean_distances(embeddings_tsne)
                    np.fill_diagonal(distances, np.inf)
                    min_distances = np.min(distances, axis=1)
                    score = np.sum(min_distances)
                    
                    if score < best_score:
                        best_score = score
                        best_tsne = tsne
                        best_params = {'perplexity': perplexity, 'learning_rate': lr}
                        
                except Exception as e:
                    continue
        
        if best_tsne is None:
            print("❌ Error: No se pudo encontrar parámetros válidos para t-SNE")
            return None, None
        
        print(f"   ✅ Mejores parámetros encontrados: {best_params}")
        print(f"   📊 Score de calidad: {best_score:.4f}")
        
        # Aplicar mejor t-SNE
        embeddings_tsne = best_tsne.fit_transform(embeddings_scaled)
        return embeddings_tsne, best_tsne
    
    def create_variational_autoencoder(self, input_dim, latent_dim=64, hidden_dims=[256, 128]):
        """Crea un autoencoder variacional personalizado"""
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dims):
                super(VAE, self).__init__()
                
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                self.encoder = nn.Sequential(*encoder_layers)
                self.fc_mu = nn.Linear(prev_dim, latent_dim)
                self.fc_var = nn.Linear(prev_dim, latent_dim)
                
                # Decoder
                decoder_layers = []
                prev_dim = latent_dim
                for hidden_dim in reversed(hidden_dims):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
            
            def encode(self, x):
                h = self.encoder(x)
                mu = self.fc_mu(h)
                log_var = self.fc_var(h)
                return mu, log_var
            
            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                return self.decoder(z)
            
            def forward(self, x):
                mu, log_var = self.encode(x)
                z = self.reparameterize(mu, log_var)
                return self.decode(z), mu, log_var
        
        return VAE(input_dim, latent_dim, hidden_dims)
    
    def train_vae(self, embeddings, latent_dim=64, epochs=100, batch_size=32, learning_rate=1e-3):
        """Entrena el autoencoder variacional"""
        print(f"🔄 Entrenando VAE con {latent_dim} dimensiones latentes...")
        
        # Preparar datos
        embeddings_tensor = torch.FloatTensor(embeddings)
        dataset = TensorDataset(embeddings_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Crear modelo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae = self.create_variational_autoencoder(embeddings.shape[1], latent_dim).to(device)
        
        # Optimizador y función de pérdida
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
        
        # Función de pérdida VAE
        def vae_loss(recon_x, x, mu, log_var):
            # Reconstruction loss
            recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            return recon_loss + kl_loss
        
        # Entrenamiento
        vae.train()
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                data = batch[0].to(device)
                
                optimizer.zero_grad()
                recon_batch, mu, log_var = vae(data)
                loss = vae_loss(recon_batch, data, mu, log_var)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print(f"✅ VAE entrenado. Loss final: {losses[-1]:.4f}")
        return vae, losses
    
    def extract_vae_latent_representations(self, vae, embeddings, latent_dim=64):
        """Extrae representaciones latentes del VAE"""
        print(f"🔄 Extrayendo representaciones latentes del VAE...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae.eval()
        
        with torch.no_grad():
            embeddings_tensor = torch.FloatTensor(embeddings).to(device)
            mu, _ = vae.encode(embeddings_tensor)
            latent_repr = mu.cpu().numpy()
        
        print(f"✅ Representaciones latentes extraídas: {latent_repr.shape}")
        return latent_repr
    
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
        
        # 3. Dynamic Time Warping (simplificado para embeddings)
        print("   ⏱️ Calculando Dynamic Time Warping...")
        dtw_distances = self._calculate_dtw_distances(embeddings)
        results['dtw_distance'] = dtw_distances
        
        # 4. Wasserstein Distance
        print("   🌊 Calculando Wasserstein Distance...")
        wasserstein_dist = self._calculate_wasserstein_distances(embeddings)
        results['wasserstein_distance'] = wasserstein_dist
        
        # 5. Optimal Transport Distance
        print("   🚚 Calculando Optimal Transport Distance...")
        ot_distances = self._calculate_optimal_transport_distances(embeddings)
        results['optimal_transport_distance'] = ot_distances
        
        # 6. Hybrid Distance (Cosine + Euclidean)
        print("   🔀 Calculando Hybrid Distance...")
        hybrid_dist = self._calculate_hybrid_distance(cosine_sim, euclidean_dist)
        results['hybrid_distance'] = hybrid_dist
        
        print("✅ Todas las métricas de similitud calculadas")
        return results
    
    def _calculate_dtw_distances(self, embeddings):
        """Calcula distancias DTW entre embeddings"""
        n_docs = embeddings.shape[0]
        dtw_matrix = np.zeros((n_docs, n_docs))
        
        for i in range(n_docs):
            for j in range(i+1, n_docs):
                # DTW simplificado para vectores
                dtw_dist = self._dtw_vector(embeddings[i], embeddings[j])
                dtw_matrix[i, j] = dtw_dist
                dtw_matrix[j, i] = dtw_dist
        
        return dtw_matrix
    
    def _dtw_vector(self, vec1, vec2):
        """Implementación simplificada de DTW para vectores"""
        n, m = len(vec1), len(vec2)
        dtw_matrix = np.full((n+1, m+1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(vec1[i-1] - vec2[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
        
        return dtw_matrix[n, m]
    
    def _calculate_wasserstein_distances(self, embeddings):
        """Calcula distancias de Wasserstein entre embeddings"""
        n_docs = embeddings.shape[0]
        wasserstein_matrix = np.zeros((n_docs, n_docs))
        
        for i in range(n_docs):
            for j in range(i+1, n_docs):
                try:
                    # Normalizar embeddings como distribuciones
                    dist1 = embeddings[i] / np.sum(embeddings[i])
                    dist2 = embeddings[j] / np.sum(embeddings[j])
                    
                    # Calcular Wasserstein distance
                    wass_dist = wasserstein_distance(dist1, dist2)
                    wasserstein_matrix[i, j] = wass_dist
                    wasserstein_matrix[j, i] = wass_dist
                except:
                    wasserstein_matrix[i, j] = np.inf
                    wasserstein_matrix[j, i] = np.inf
        
        return wasserstein_matrix
    
    def _calculate_optimal_transport_distances(self, embeddings):
        """Calcula distancias de Optimal Transport entre embeddings"""
        n_docs = embeddings.shape[0]
        ot_matrix = np.zeros((n_docs, n_docs))
        
        for i in range(n_docs):
            for j in range(i+1, n_docs):
                try:
                    # Normalizar embeddings
                    dist1 = embeddings[i] / np.sum(embeddings[i])
                    dist2 = embeddings[j] / np.sum(embeddings[j])
                    
                    # Matriz de costos (distancia euclidiana entre dimensiones)
                    cost_matrix = np.abs(np.arange(len(dist1))[:, None] - np.arange(len(dist2)))
                    
                    # Calcular Optimal Transport
                    ot_dist = ot.emd2(dist1, dist2, cost_matrix)
                    ot_matrix[i, j] = ot_dist
                    ot_matrix[j, i] = ot_dist
                except:
                    ot_matrix[i, j] = np.inf
                    ot_matrix[j, i] = np.inf
        
        return ot_matrix
    
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
            # Usar clustering K-means para calcular silhouette
            from sklearn.cluster import KMeans
            
            # Determinar número óptimo de clusters
            n_clusters = min(10, len(embeddings) // 2)
            if n_clusters < 2:
                return 0.0
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            return silhouette_avg
        except:
            return 0.0
    
    def visualize_results(self, results_dict):
        """Visualiza todos los resultados"""
        print("🎨 Generando visualizaciones...")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('Análisis de Reducción de Dimensionalidad Inteligente', fontsize=20, fontweight='bold')
        
        methods = list(results_dict.keys())
        
        for i, method in enumerate(methods[:6]):  # Máximo 6 métodos
            row = i // 3
            col = i % 3
            
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
        plt.savefig('intelligent_dimensionality_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizaciones guardadas en 'intelligent_dimensionality_results.png'")
    
    def generate_comprehensive_report(self, results_dict, evaluation_results):
        """Genera reporte completo del análisis"""
        print("📊 Generando reporte completo...")
        
        with open('intelligent_dimensionality_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE REDUCCIÓN DE DIMENSIONALIDAD INTELIGENTE\n")
            f.write("Estrategias 4 y 5: UMAP, t-SNE optimizado, VAE y Métricas Avanzadas\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("RESUMEN DE MÉTODOS IMPLEMENTADOS:\n")
            f.write("-" * 40 + "\n")
            
            for method, result in results_dict.items():
                if 'embeddings' in result:
                    f.write(f"\n{method}:\n")
                    f.write(f"  • Forma: {result['embeddings'].shape}\n")
                    if 'config' in result:
                        f.write(f"  • Configuración: {result['config']}\n")
            
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
            f.write("2. Considerar VAE para representaciones latentes compactas\n")
            f.write("3. Evaluar métricas híbridas para mejor rendimiento\n")
            f.write("4. Probar diferentes configuraciones de UMAP y t-SNE\n")
        
        print("✅ Reporte generado: intelligent_dimensionality_report.txt")

def main():
    """Función principal para ejecutar el análisis completo"""
    print("🚀 INICIANDO ANÁLISIS DE REDUCCIÓN DE DIMENSIONALIDAD INTELIGENTE")
    print("=" * 80)
    print("Implementando Estrategias 4 y 5:")
    print("• Estrategia 4: Técnicas de Reducción de Dimensionalidad Inteligente")
    print("• Estrategia 5: Métricas de Similitud Avanzadas")
    print("=" * 80)
    
    # Crear analizador
    analyzer = IntelligentDimensionalityReducer()
    
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
    
    # 1. UMAP
    print("\n📊 1. ANÁLISIS UMAP")
    embeddings_umap, umap_reducer = analyzer.perform_umap_analysis()
    results['UMAP'] = {'embeddings': embeddings_umap, 'reducer': umap_reducer}
    
    # 2. t-SNE Optimizado
    print("\n📊 2. ANÁLISIS t-SNE OPTIMIZADO")
    embeddings_tsne, tsne_reducer = analyzer.perform_optimized_tsne()
    results['TSNE_Optimized'] = {'embeddings': embeddings_tsne, 'reducer': tsne_reducer}
    
    # 3. VAE
    print("\n📊 3. AUTOENCODER VARIACIONAL (VAE)")
    vae, losses = analyzer.train_vae(analyzer.original_embeddings, latent_dim=128)
    latent_repr = analyzer.extract_vae_latent_representations(vae, analyzer.original_embeddings, 128)
    results['VAE_Latent'] = {'embeddings': latent_repr, 'vae': vae, 'losses': losses}
    
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
    
    for method_name in ['UMAP', 'TSNE_Optimized', 'VAE_Latent']:
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
    analyzer.generate_comprehensive_report(results, evaluation_results)
    
    print("\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE!")
    print("📁 Archivos generados:")
    print("   - intelligent_dimensionality_results.png")
    print("   - intelligent_dimensionality_report.txt")
    
    # Mostrar resumen de resultados
    print("\n📊 RESUMEN DE RESULTADOS:")
    print("-" * 30)
    for eval_result in evaluation_results:
        print(f"   {eval_result['method']}: Silhouette={eval_result['silhouette_score']:.4f}, "
              f"Var_Ratio={eval_result['variance_ratio']:.4f}")

if __name__ == "__main__":
    main()
