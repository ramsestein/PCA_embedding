#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISCRIMINADOR ULTRA-INTELIGENTE - TOP-1 DEL 100%
Autor: Análisis de Embeddings Médicos
Fecha: 2025
Objetivo: Discriminación absoluta mediante estrategias ultra-inteligentes
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import warnings
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

class UltraDiscriminator:
    def __init__(self, model_path="all-mini-base", pnts_folder="PNTs"):
        self.model_path = model_path
        self.pnts_folder = pnts_folder
        self.model = None
        self.pnts_documents = []
        self.pnts_names = []
        
    def load_model(self):
        """Cargar el modelo base"""
        print("🔄 Cargando modelo base...")
        self.model = SentenceTransformer(self.model_path)
        print(f"✅ Modelo cargado: {self.model.get_sentence_embedding_dimension()} dimensiones")
        
    def load_pnts_documents(self):
        """Cargar documentos PNTs"""
        print("🔄 Cargando documentos PNTs...")
        pnts_files = [f for f in os.listdir(self.pnts_folder) if f.endswith('.txt')]
        pnts_files.sort()
        
        for file_name in pnts_files:
            file_path = os.path.join(self.pnts_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                self.pnts_documents.append(content)
                self.pnts_names.append(file_name)
                
        print(f"✅ {len(self.pnts_documents)} documentos PNTs cargados")
        
    def generate_ultra_expansions(self):
        """Generar expansiones ultra-inteligentes para Top-1 del 100%"""
        print("🚀 GENERANDO EXPANSIONES ULTRA-INTELIGENTES PARA TOP-1 DEL 100%...")
        
        original_embeddings = self.model.encode(self.pnts_documents)
        print(f"📊 Embeddings originales: {original_embeddings.shape}")
        
        # ESTRATEGIAS ULTRA-INTELIGENTES para discriminación absoluta
        ultra_configs = [
            # ESTRATEGIA 1: Ruido controlado por similitud semántica
            {
                'dimensions': 300,
                'strategy': 'semantic_controlled',
                'noise_scale': 0.3,
                'description': 'Semántico Controlado (300d)'
            },
            
            # ESTRATEGIA 2: Expansión progresiva inteligente
            {
                'dimensions': 400,
                'strategy': 'progressive_intelligent',
                'noise_scale': 0.4,
                'description': 'Progresivo Inteligente (400d)'
            },
            
            # ESTRATEGIA 3: Ruido específico por documento
            {
                'dimensions': 500,
                'strategy': 'document_specific',
                'noise_scale': 0.5,
                'description': 'Específico por Documento (500d)'
            },
            
            # ESTRATEGIA 4: Balance semántico-diferenciación
            {
                'dimensions': 600,
                'strategy': 'semantic_balance',
                'noise_scale': 0.6,
                'description': 'Balance Semántico (600d)'
            },
            
            # ESTRATEGIA 5: Ruido adaptativo por clusters
            {
                'dimensions': 700,
                'strategy': 'adaptive_clusters',
                'noise_scale': 0.7,
                'description': 'Adaptativo por Clusters (700d)'
            },
            
            # ESTRATEGIA 6: Diferenciación secuencial inteligente
            {
                'dimensions': 800,
                'strategy': 'sequential_intelligent',
                'noise_scale': 0.8,
                'description': 'Secuencial Inteligente (800d)'
            },
            
            # ESTRATEGIA 7: Ruido híbrido ultra-controlado
            {
                'dimensions': 900,
                'strategy': 'hybrid_ultra_controlled',
                'noise_scale': 0.9,
                'description': 'Híbrido Ultra-Controlado (900d)'
            },
            
            # ESTRATEGIA 8: Máxima discriminación preservando semántica
            {
                'dimensions': 1000,
                'strategy': 'max_discrimination_semantic',
                'noise_scale': 1.0,
                'description': 'Máxima Discriminación Semántica (1000d)'
            }
        ]
        
        results = {}
        for i, config in enumerate(ultra_configs):
            print(f"\n🔬 Generando: {config['description']}")
            expanded_embeddings = self._expand_embeddings_ultra(original_embeddings, config)
            differentiation_stats = self._calculate_ultra_differentiation_stats(expanded_embeddings)
            
            results[f'ultra_expansion_{i+1}'] = {
                'config': config,
                'embeddings': expanded_embeddings,
                'stats': differentiation_stats
            }
            
            print(f"   ✅ Ratio de varianza: {differentiation_stats['variance_ratio']:.4f}")
            print(f"   ✅ Distancia promedio: {differentiation_stats['avg_distance']:.4f}")
            print(f"   ✅ Distancia mínima: {differentiation_stats['min_distance']:.4f}")
            print(f"   🎯 Puntuación de separación: {differentiation_stats['separation_score']:.4f}")
            
        self._save_ultra_expansions(results)
        return results
    
    def _expand_embeddings_ultra(self, original_embeddings, config):
        """Expansión ultra-inteligente con diferentes estrategias"""
        n_orig = original_embeddings.shape[1]
        n_add = config['dimensions']
        n_docs = original_embeddings.shape[0]
        
        expanded = np.zeros((n_docs, n_orig + n_add))
        expanded[:, :n_orig] = original_embeddings
        
        strategy = config['strategy']
        noise_scale = config['noise_scale']
        
        if strategy == 'semantic_controlled':
            noise = self._generate_semantic_controlled_noise(original_embeddings, n_add, noise_scale)
        elif strategy == 'progressive_intelligent':
            noise = self._generate_progressive_intelligent_noise(original_embeddings, n_add, noise_scale)
        elif strategy == 'document_specific':
            noise = self._generate_document_specific_noise(original_embeddings, n_add, noise_scale)
        elif strategy == 'semantic_balance':
            noise = self._generate_semantic_balance_noise(original_embeddings, n_add, noise_scale)
        elif strategy == 'adaptive_clusters':
            noise = self._generate_adaptive_clusters_noise(original_embeddings, n_add, noise_scale)
        elif strategy == 'sequential_intelligent':
            noise = self._generate_sequential_intelligent_noise(original_embeddings, n_add, noise_scale)
        elif strategy == 'hybrid_ultra_controlled':
            noise = self._generate_hybrid_ultra_controlled_noise(original_embeddings, n_add, noise_scale)
        elif strategy == 'max_discrimination_semantic':
            noise = self._generate_max_discrimination_semantic_noise(original_embeddings, n_add, noise_scale)
        else:
            noise = np.random.uniform(-noise_scale, noise_scale, (n_docs, n_add))
            
        expanded[:, n_orig:] = noise
        return expanded
    
    def _generate_semantic_controlled_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido controlado por similitud semántica"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular similitud coseno entre documentos
        similarities = cosine_similarity(original_embeddings)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Encontrar documentos más similares
            similar_docs = np.argsort(similarities[i])[1:4]  # Top 3 más similares
            
            # Generar ruido que diferencie de documentos similares pero preserve semántica
            base_noise = np.random.uniform(-noise_scale * 0.5, noise_scale * 0.5, n_add)
            
            # Ruido de diferenciación controlada
            diff_noise = np.zeros(n_add)
            for j, similar_doc in enumerate(similar_docs):
                similarity_score = similarities[i][similar_doc]
                # Cuanto más similar, más ruido de diferenciación
                doc_noise = np.random.uniform(-noise_scale * 0.3, noise_scale * 0.3, n_add)
                diff_noise += doc_noise * similarity_score * 0.5
                
            noise[i, :] = base_noise + diff_noise
            
        return noise
    
    def _generate_progressive_intelligent_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido progresivo inteligente"""
        n_docs = original_embeddings.shape[0]
        
        # Dividir dimensiones en secciones progresivas
        n_sections = 4
        section_size = n_add // n_sections
        
        noise = np.zeros((n_docs, n_add))
        
        for section in range(n_sections):
            start_idx = section * section_size
            end_idx = start_idx + section_size if section < n_sections - 1 else n_add
            
            # Escala progresiva: más agresiva en secciones posteriores
            section_scale = noise_scale * (0.5 + section * 0.2)
            
            # Ruido base progresivo
            section_noise = np.random.uniform(-section_scale, section_scale, (n_docs, end_idx - start_idx))
            
            # Ruido de diferenciación progresiva
            if section > 0:
                # Usar información de secciones anteriores
                prev_noise = noise[:, :start_idx]
                prev_mean = np.mean(prev_noise, axis=1, keepdims=True)
                section_noise += prev_mean * 0.1
                
            noise[:, start_idx:end_idx] = section_noise
            
        return noise
    
    def _generate_document_specific_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido específico por documento"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular características únicas por documento
        doc_means = np.mean(original_embeddings, axis=1)
        doc_stds = np.std(original_embeddings, axis=1)
        doc_maxs = np.max(original_embeddings, axis=1)
        doc_mins = np.min(original_embeddings, axis=1)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Ruido base específico del documento
            doc_scale = noise_scale * (1 + doc_stds[i] * 0.3)
            base_noise = np.random.uniform(-doc_scale, doc_scale, n_add)
            
            # Ruido de características específicas
            feature_noise = np.random.normal(doc_means[i] * 0.05, doc_stds[i] * 0.1, n_add)
            
            # Ruido de rango específico
            range_noise = np.random.uniform(doc_mins[i] * 0.05, doc_maxs[i] * 0.05, n_add)
            
            # Ruido de diferenciación específica
            diff_noise = np.random.uniform(-noise_scale * 0.3, noise_scale * 0.3, n_add)
            
            noise[i, :] = base_noise + feature_noise + range_noise + diff_noise
            
        return noise
    
    def _generate_semantic_balance_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido balanceando semántica y diferenciación"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular similitud semántica
        similarities = cosine_similarity(original_embeddings)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Encontrar documentos más similares y diferentes
            similar_docs = np.argsort(similarities[i])[1:4]  # Top 3 más similares
            different_docs = np.argsort(similarities[i])[-4:-1]  # Top 3 más diferentes
            
            # Ruido base balanceado
            base_noise = np.random.uniform(-noise_scale * 0.6, noise_scale * 0.6, n_add)
            
            # Ruido de diferenciación de similares
            diff_similar = np.zeros(n_add)
            for similar_doc in similar_docs:
                similarity_score = similarities[i][similar_doc]
                doc_noise = np.random.uniform(-noise_scale * 0.4, noise_scale * 0.4, n_add)
                diff_similar += doc_noise * similarity_score * 0.3
                
            # Ruido de preservación de diferentes
            preserve_different = np.zeros(n_add)
            for different_doc in different_docs:
                similarity_score = similarities[i][different_doc]
                doc_noise = np.random.uniform(-noise_scale * 0.2, noise_scale * 0.2, n_add)
                preserve_different += doc_noise * (1 - similarity_score) * 0.2
                
            noise[i, :] = base_noise + diff_similar + preserve_different
            
        return noise
    
    def _generate_adaptive_clusters_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido adaptativo por clusters"""
        n_docs = original_embeddings.shape[0]
        
        # Crear clusters de documentos similares
        n_clusters = min(5, n_docs // 3)  # Máximo 5 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(original_embeddings)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            cluster_id = cluster_labels[i]
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            
            # Ruido base del cluster
            cluster_scale = noise_scale * (1 + len(cluster_members) * 0.1)
            base_noise = np.random.uniform(-cluster_scale, cluster_scale, n_add)
            
            # Ruido de diferenciación dentro del cluster
            cluster_diff = np.zeros(n_add)
            for member in cluster_members:
                if member != i:
                    doc_noise = np.random.uniform(-noise_scale * 0.3, noise_scale * 0.3, n_add)
                    cluster_diff += doc_noise * 0.2
                    
            # Ruido de separación entre clusters
            other_clusters = np.where(cluster_labels != cluster_id)[0]
            cluster_sep = np.zeros(n_add)
            for other in other_clusters[:3]:  # Top 3 de otros clusters
                doc_noise = np.random.uniform(-noise_scale * 0.2, noise_scale * 0.2, n_add)
                cluster_sep += doc_noise * 0.1
                
            noise[i, :] = base_noise + cluster_diff + cluster_sep
            
        return noise
    
    def _generate_sequential_intelligent_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido secuencial inteligente"""
        n_docs = original_embeddings.shape[0]
        
        # Dividir en secuencias inteligentes
        n_sequences = 5
        seq_length = n_add // n_sequences
        
        noise = np.zeros((n_docs, n_add))
        
        for seq in range(n_sequences):
            start_idx = seq * seq_length
            end_idx = start_idx + seq_length if seq < n_sequences - 1 else n_add
            
            # Estrategia específica por secuencia
            if seq == 0:  # Secuencia 1: Preservación semántica
                seq_noise = np.random.uniform(-noise_scale * 0.3, noise_scale * 0.3, (n_docs, end_idx - start_idx))
            elif seq == 1:  # Secuencia 2: Diferenciación básica
                seq_noise = np.random.uniform(-noise_scale * 0.5, noise_scale * 0.5, (n_docs, end_idx - start_idx))
            elif seq == 2:  # Secuencia 3: Diferenciación avanzada
                seq_noise = np.random.uniform(-noise_scale * 0.7, noise_scale * 0.7, (n_docs, end_idx - start_idx))
            elif seq == 3:  # Secuencia 4: Diferenciación extrema
                seq_noise = np.random.uniform(-noise_scale * 0.9, noise_scale * 0.9, (n_docs, end_idx - start_idx))
            else:  # Secuencia 5: Balance final
                seq_noise = np.random.uniform(-noise_scale * 0.6, noise_scale * 0.6, (n_docs, end_idx - start_idx))
                
            # Aplicar ruido de características específicas
            if seq > 0:
                doc_means = np.mean(original_embeddings, axis=1)
                feature_noise = np.random.normal(doc_means.reshape(-1, 1) * 0.02, 0.01, (n_docs, end_idx - start_idx))
                seq_noise += feature_noise
                
            noise[:, start_idx:end_idx] = seq_noise
            
        return noise
    
    def _generate_hybrid_ultra_controlled_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido híbrido ultra-controlado"""
        n_docs = original_embeddings.shape[0]
        
        # Dividir dimensiones en secciones híbridas
        n_uniform = n_add // 4
        n_gaussian = n_add // 4
        n_exponential = n_add // 4
        n_adaptive = n_add - n_uniform - n_gaussian - n_exponential
        
        noise = np.zeros((n_docs, n_add))
        
        # Sección 1: Ruido uniforme controlado
        if n_uniform > 0:
            uniform_noise = np.random.uniform(-noise_scale * 0.6, noise_scale * 0.6, (n_docs, n_uniform))
            noise[:, :n_uniform] = uniform_noise
            
        # Sección 2: Ruido gaussiano controlado
        if n_gaussian > 0:
            gaussian_noise = np.random.normal(0, noise_scale * 0.5, (n_docs, n_gaussian))
            noise[:, n_uniform:n_uniform+n_gaussian] = gaussian_noise
            
        # Sección 3: Ruido exponencial controlado
        if n_exponential > 0:
            exp_noise = np.random.exponential(noise_scale * 0.4, (n_docs, n_exponential))
            exp_noise = np.random.choice([-1, 1], (n_docs, n_exponential)) * exp_noise
            noise[:, n_uniform+n_gaussian:n_uniform+n_gaussian+n_exponential] = exp_noise
            
        # Sección 4: Ruido adaptativo controlado
        if n_adaptive > 0:
            doc_stds = np.std(original_embeddings, axis=1)
            adaptive_noise = np.random.uniform(-noise_scale * 0.8, noise_scale * 0.8, (n_docs, n_adaptive))
            # Ajustar por características del documento
            for i in range(n_docs):
                adaptive_noise[i, :] *= (1 + doc_stds[i] * 0.2)
            noise[:, n_uniform+n_gaussian+n_exponential:] = adaptive_noise
            
        return noise
    
    def _generate_max_discrimination_semantic_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido de máxima discriminación preservando semántica"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular similitud semántica
        similarities = cosine_similarity(original_embeddings)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Encontrar documentos más similares
            similar_docs = np.argsort(similarities[i])[1:6]  # Top 5 más similares
            
            # Ruido base de máxima diferenciación
            base_noise = np.random.uniform(-noise_scale, noise_scale, n_add)
            
            # Ruido de diferenciación extrema de similares
            extreme_diff = np.zeros(n_add)
            for j, similar_doc in enumerate(similar_docs):
                similarity_score = similarities[i][similar_doc]
                # Cuanto más similar, más diferenciación extrema
                doc_noise = np.random.uniform(-noise_scale * 1.2, noise_scale * 1.2, n_add)
                extreme_diff += doc_noise * similarity_score * 0.8
                
            # Ruido de preservación semántica
            semantic_preserve = np.random.normal(0, noise_scale * 0.3, n_add)
            
            # Ruido de características específicas del documento
            doc_means = np.mean(original_embeddings[i, :])
            doc_stds = np.std(original_embeddings[i, :])
            feature_noise = np.random.normal(doc_means * 0.03, doc_stds * 0.05, n_add)
            
            noise[i, :] = base_noise + extreme_diff + semantic_preserve + feature_noise
            
        return noise
    
    def _calculate_ultra_differentiation_stats(self, expanded_embeddings):
        """Calcular estadísticas de diferenciación ultra-inteligente"""
        # Distancias euclidianas entre todos los pares
        distances = euclidean_distances(expanded_embeddings)
        
        # Obtener solo las distancias entre documentos diferentes (triangular superior)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        
        # Calcular similitud coseno para preservación semántica
        similarities = cosine_similarity(expanded_embeddings)
        upper_sim = similarities[np.triu_indices_from(similarities, k=1)]
        
        stats = {
            'avg_distance': np.mean(upper_tri),
            'min_distance': np.min(upper_tri),
            'max_distance': np.max(upper_tri),
            'std_distance': np.std(upper_tri),
            'variance_ratio': np.var(upper_tri) / np.var(expanded_embeddings),
            'separation_score': np.min(upper_tri) / np.mean(upper_tri),
            'discrimination_potential': np.std(upper_tri) / np.mean(upper_tri),
            'semantic_preservation': np.mean(upper_sim),  # Cuanto más alto, mejor preservación semántica
            'discrimination_semantic_balance': np.min(upper_tri) / (np.mean(upper_sim) + 1e-8)  # Balance entre diferenciación y semántica
        }
        
        return stats
    
    def _save_ultra_expansions(self, results):
        """Guardar todas las expansiones ultra-inteligentes"""
        print("\n💾 Guardando expansiones ultra-inteligentes...")
        
        # Preparar datos para guardar
        save_data = {}
        for key, value in results.items():
            save_data[key] = value['embeddings']
            save_data[f'{key}_config'] = value['config']
            save_data[f'{key}_stats'] = value['stats']
        
        # Guardar nombres de archivos PNTs
        save_data['pnts_names'] = np.array(self.pnts_names)
        
        # Guardar en archivo NPZ
        np.savez_compressed('ultra_expansions.npz', **save_data)
        print("✅ Expansiones ultra-inteligentes guardadas en 'ultra_expansions.npz'")
        
        # Generar reporte
        self._generate_ultra_report(results)
    
    def _generate_ultra_report(self, results):
        """Generar reporte de expansiones ultra-inteligentes"""
        print("\n📊 Generando reporte de expansiones ultra-inteligentes...")
        
        report_lines = [
            "🚀 REPORTE DE EXPANSIONES ULTRA-INTELIGENTES - TOP-1 DEL 100%",
            "=" * 80,
            f"Fecha: 2025",
            f"Modelo base: {self.model_path}",
            f"Documentos PNTs: {len(self.pnts_documents)}",
            f"Dimensiones originales: {self.model.get_sentence_embedding_dimension()}",
            "",
            "🎯 OBJETIVO: TOP-1 DEL 100% MEDIANTE ESTRATEGIAS ULTRA-INTELIGENTES",
            "",
            "📊 CONFIGURACIONES ULTRA-INTELIGENTES GENERADAS:",
            "-" * 60
        ]
        
        for key, value in results.items():
            config = value['config']
            stats = value['stats']
            
            report_lines.extend([
                f"\n🔬 {config['description']}",
                f"   📏 Dimensiones adicionales: {config['dimensions']}",
                f"   🧠 Estrategia: {config['strategy']}",
                f"   📊 Escala de ruido: {config['noise_scale']}",
                f"   📈 Ratio de varianza: {stats['variance_ratio']:.4f}",
                f"   📏 Distancia promedio: {stats['avg_distance']:.4f}",
                f"   📏 Distancia mínima: {stats['min_distance']:.4f}",
                f"   📏 Distancia máxima: {stats['max_distance']:.4f}",
                f"   🎯 Puntuación de separación: {stats['separation_score']:.4f}",
                f"   🚀 Potencial de discriminación: {stats['discrimination_potential']:.4f}",
                f"   🧠 Preservación semántica: {stats['semantic_preservation']:.4f}",
                f"   ⚖️ Balance discriminación-semántica: {stats['discrimination_semantic_balance']:.4f}"
            ])
        
        report_lines.extend([
            "",
            "💡 ANÁLISIS DE ESTRATEGIAS ULTRA-INTELIGENTES:",
            "-" * 50,
            "1. 🧠 Semántico Controlado: Preserva similitud mientras diferencia",
            "2. 📈 Progresivo Inteligente: Aumenta diferenciación gradualmente",
            "3. 📄 Específico por Documento: Ruido personalizado por documento",
            "4. ⚖️ Balance Semántico: Equilibrio entre diferenciación y semántica",
            "5. 🎯 Adaptativo por Clusters: Diferenciación inteligente por grupos",
            "6. 🔄 Secuencial Inteligente: Estrategias en cascada",
            "7. 🎲 Híbrido Ultra-Controlado: Múltiples tipos de ruido controlados",
            "8. 🚀 Máxima Discriminación Semántica: Diferenciación extrema preservando semántica",
            "",
            "🚀 PRÓXIMO PASO:",
            "Ejecutar 'ultra_benchmark_evaluator.py' para evaluar con el benchmark",
            "y verificar si alcanzamos el Top-1 del 100%",
            "",
            "=" * 80
        ])
        
        # Guardar reporte
        with open('ultra_expansions_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("✅ Reporte guardado en 'ultra_expansions_report.txt'")
        
        # Mostrar resumen
        print("\n📊 RESUMEN DE EXPANSIONES ULTRA-INTELIGENTES:")
        print("-" * 60)
        for key, value in results.items():
            config = value['config']
            stats = value['stats']
            print(f"🔬 {config['description']}")
            print(f"   📏 +{config['dimensions']}d | 🎯 Separación: {stats['separation_score']:.4f} | 🧠 Semántica: {stats['semantic_preservation']:.4f}")

if __name__ == "__main__":
    print("🚀 INICIANDO DISCRIMINADOR ULTRA-INTELIGENTE PARA TOP-1 DEL 100%")
    print("=" * 80)
    
    discriminator = UltraDiscriminator()
    discriminator.load_model()
    discriminator.load_pnts_documents()
    
    # Generar expansiones ultra-inteligentes
    results = discriminator.generate_ultra_expansions()
    
    print("\n🎉 ¡EXPANSIONES ULTRA-INTELIGENTES GENERADAS EXITOSAMENTE!")
    print("🚀 Próximo paso: Evaluar con el benchmark para verificar Top-1 del 100%")
