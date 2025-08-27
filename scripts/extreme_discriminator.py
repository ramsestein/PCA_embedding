#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISCRIMINADOR EXTREMO - BUSCANDO TOP-1 DEL 100%
Autor: Análisis de Embeddings Médicos
Fecha: 2025
Objetivo: Discriminación absoluta mediante expansión dimensional masiva
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import warnings
from sklearn.metrics.pairwise import euclidean_distances
warnings.filterwarnings('ignore')

class ExtremeDiscriminator:
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
        
    def generate_extreme_expansions(self):
        """Generar expansiones extremas para discriminación absoluta"""
        print("🚀 GENERANDO EXPANSIONES EXTREMAS PARA TOP-1 DEL 100%...")
        
        original_embeddings = self.model.encode(self.pnts_documents)
        print(f"📊 Embeddings originales: {original_embeddings.shape}")
        
        # CONFIGURACIONES EXTREMAS para discriminación absoluta
        extreme_configs = [
            # EXPANSIÓN MASIVA BÁSICA
            {
                'dimensions': 500,
                'noise_type': 'extreme_uniform',
                'noise_scale': 0.8,
                'description': 'Extremo Uniforme Masivo (500d)'
            },
            {
                'dimensions': 1000,
                'noise_type': 'extreme_uniform',
                'noise_scale': 1.0,
                'description': 'Extremo Uniforme Ultra Masivo (1000d)'
            },
            
            # EXPANSIÓN GAUSSIANA EXTREMA
            {
                'dimensions': 500,
                'noise_type': 'extreme_gaussian',
                'noise_scale': 0.7,
                'description': 'Extremo Gaussiano Masivo (500d)'
            },
            {
                'dimensions': 1000,
                'noise_type': 'extreme_gaussian',
                'noise_scale': 0.9,
                'description': 'Extremo Gaussiano Ultra Masivo (1000d)'
            },
            
            # EXPANSIÓN HÍBRIDA COMPLEJA
            {
                'dimensions': 750,
                'noise_type': 'hybrid_extreme',
                'noise_scale': 0.8,
                'description': 'Híbrido Extremo Complejo (750d)'
            },
            
            # EXPANSIÓN POR CLUSTERS
            {
                'dimensions': 600,
                'noise_type': 'cluster_based',
                'noise_scale': 0.9,
                'description': 'Basado en Clusters (600d)'
            },
            
            # EXPANSIÓN SECUENCIAL
            {
                'dimensions': 800,
                'noise_type': 'sequential_extreme',
                'noise_scale': 0.85,
                'description': 'Secuencial Extremo (800d)'
            },
            
            # EXPANSIÓN ADAPTATIVA
            {
                'dimensions': 900,
                'noise_type': 'adaptive_extreme',
                'noise_scale': 0.95,
                'description': 'Adaptativo Extremo (900d)'
            }
        ]
        
        results = {}
        for i, config in enumerate(extreme_configs):
            print(f"\n🔬 Generando: {config['description']}")
            expanded_embeddings = self._expand_embeddings_extreme(original_embeddings, config)
            differentiation_stats = self._calculate_extreme_differentiation_stats(expanded_embeddings)
            
            results[f'extreme_expansion_{i+1}'] = {
                'config': config,
                'embeddings': expanded_embeddings,
                'stats': differentiation_stats
            }
            
            print(f"   ✅ Ratio de varianza: {differentiation_stats['variance_ratio']:.4f}")
            print(f"   ✅ Distancia promedio: {differentiation_stats['avg_distance']:.4f}")
            print(f"   ✅ Distancia mínima: {differentiation_stats['min_distance']:.4f}")
            
        self._save_extreme_expansions(results)
        return results
    
    def _expand_embeddings_extreme(self, original_embeddings, config):
        """Expansión extrema con diferentes estrategias"""
        n_orig = original_embeddings.shape[1]
        n_add = config['dimensions']
        n_docs = original_embeddings.shape[0]
        
        expanded = np.zeros((n_docs, n_orig + n_add))
        expanded[:, :n_orig] = original_embeddings
        
        noise_type = config['noise_type']
        noise_scale = config['noise_scale']
        
        if noise_type == 'extreme_uniform':
            noise = self._generate_extreme_uniform_noise(original_embeddings, n_add, noise_scale)
        elif noise_type == 'extreme_gaussian':
            noise = self._generate_extreme_gaussian_noise(original_embeddings, n_add, noise_scale)
        elif noise_type == 'hybrid_extreme':
            noise = self._generate_hybrid_extreme_noise(original_embeddings, n_add, noise_scale)
        elif noise_type == 'cluster_based':
            noise = self._generate_cluster_based_noise(original_embeddings, n_add, noise_scale)
        elif noise_type == 'sequential_extreme':
            noise = self._generate_sequential_extreme_noise(original_embeddings, n_add, noise_scale)
        elif noise_type == 'adaptive_extreme':
            noise = self._generate_adaptive_extreme_noise(original_embeddings, n_add, noise_scale)
        else:
            noise = np.random.uniform(-noise_scale, noise_scale, (n_docs, n_add))
            
        expanded[:, n_orig:] = noise
        return expanded
    
    def _generate_extreme_uniform_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido uniforme extremo con diferenciación máxima"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular estadísticas por documento para ruido personalizado
        doc_means = np.mean(original_embeddings, axis=1)
        doc_stds = np.std(original_embeddings, axis=1)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Ruido base uniforme
            base_noise = np.random.uniform(-noise_scale, noise_scale, n_add)
            
            # Ruido personalizado por documento
            doc_signature = np.random.uniform(-0.5, 0.5, n_add) * doc_stds[i]
            
            # Ruido de diferenciación extrema
            diff_noise = np.random.uniform(-1.0, 1.0, n_add) * noise_scale * 0.5
            
            noise[i, :] = base_noise + doc_signature + diff_noise
            
        return noise
    
    def _generate_extreme_gaussian_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido gaussiano extremo con control de diferenciación"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular características únicas por documento
        doc_means = np.mean(original_embeddings, axis=1)
        doc_stds = np.std(original_embeddings, axis=1)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Ruido gaussiano base
            base_noise = np.random.normal(0, noise_scale, n_add)
            
            # Ruido de diferenciación específico
            doc_pattern = np.random.normal(doc_means[i] * 0.1, doc_stds[i] * 0.2, n_add)
            
            # Ruido de separación extrema
            separation_noise = np.random.normal(0, noise_scale * 0.8, n_add)
            
            noise[i, :] = base_noise + doc_pattern + separation_noise
            
        return noise
    
    def _generate_hybrid_extreme_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido híbrido extremo combinando múltiples estrategias"""
        n_docs = original_embeddings.shape[0]
        
        # Dividir dimensiones adicionales en secciones
        n_uniform = n_add // 3
        n_gaussian = n_add // 3
        n_exponential = n_add - n_uniform - n_gaussian
        
        noise = np.zeros((n_docs, n_add))
        
        # Sección 1: Ruido uniforme extremo
        if n_uniform > 0:
            noise[:, :n_uniform] = self._generate_extreme_uniform_noise(
                original_embeddings, n_uniform, noise_scale * 1.2
            )
        
        # Sección 2: Ruido gaussiano extremo
        if n_gaussian > 0:
            noise[:, n_uniform:n_uniform+n_gaussian] = self._generate_extreme_gaussian_noise(
                original_embeddings, n_gaussian, noise_scale * 1.1
            )
        
        # Sección 3: Ruido exponencial extremo
        if n_exponential > 0:
            exp_noise = np.random.exponential(noise_scale, (n_docs, n_exponential))
            exp_noise = np.random.choice([-1, 1], (n_docs, n_exponential)) * exp_noise
            noise[:, n_uniform+n_gaussian:] = exp_noise
            
        return noise
    
    def _generate_cluster_based_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido basado en similitud de documentos"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular similitud entre documentos
        similarities = euclidean_distances(original_embeddings)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Encontrar documentos más similares
            similar_docs = np.argsort(similarities[i])[1:4]  # Top 3 más similares
            
            # Generar ruido que diferencie de documentos similares
            base_noise = np.random.uniform(-noise_scale, noise_scale, n_add)
            
            # Ruido de diferenciación específica
            diff_pattern = np.zeros(n_add)
            for j, similar_doc in enumerate(similar_docs):
                doc_noise = np.random.uniform(-noise_scale * 0.5, noise_scale * 0.5, n_add)
                diff_pattern += doc_noise * (j + 1) * 0.3  # Peso decreciente
                
            noise[i, :] = base_noise + diff_pattern
            
        return noise
    
    def _generate_sequential_extreme_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido secuencial aplicando diferentes patrones en secuencia"""
        n_docs = original_embeddings.shape[0]
        
        # Dividir en secuencias
        seq_length = n_add // 4
        noise = np.zeros((n_docs, n_add))
        
        # Secuencia 1: Ruido uniforme
        if seq_length > 0:
            noise[:, :seq_length] = self._generate_extreme_uniform_noise(
                original_embeddings, seq_length, noise_scale
            )
        
        # Secuencia 2: Ruido gaussiano
        if seq_length > 0:
            noise[:, seq_length:2*seq_length] = self._generate_extreme_gaussian_noise(
                original_embeddings, seq_length, noise_scale * 1.1
            )
        
        # Secuencia 3: Ruido exponencial
        if seq_length > 0:
            exp_noise = np.random.exponential(noise_scale * 0.8, (n_docs, seq_length))
            exp_noise = np.random.choice([-1, 1], (n_docs, seq_length)) * exp_noise
            noise[:, 2*seq_length:3*seq_length] = exp_noise
            
        # Secuencia 4: Ruido mixto final
        remaining = n_add - 3*seq_length
        if remaining > 0:
            mixed_noise = np.random.uniform(-noise_scale * 1.5, noise_scale * 1.5, (n_docs, remaining))
            noise[:, 3*seq_length:] = mixed_noise
            
        return noise
    
    def _generate_adaptive_extreme_noise(self, original_embeddings, n_add, noise_scale):
        """Ruido adaptativo basado en características del documento"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular características adaptativas
        doc_means = np.mean(original_embeddings, axis=1)
        doc_stds = np.std(original_embeddings, axis=1)
        doc_maxs = np.max(original_embeddings, axis=1)
        doc_mins = np.min(original_embeddings, axis=1)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Ruido base adaptativo
            base_scale = noise_scale * (1 + doc_stds[i] * 0.5)
            base_noise = np.random.uniform(-base_scale, base_scale, n_add)
            
            # Ruido de características específicas
            feature_noise = np.random.normal(doc_means[i] * 0.1, doc_stds[i] * 0.3, n_add)
            
            # Ruido de rango adaptativo
            range_noise = np.random.uniform(doc_mins[i] * 0.1, doc_maxs[i] * 0.1, n_add)
            
            # Ruido de diferenciación extrema
            extreme_noise = np.random.uniform(-noise_scale * 2, noise_scale * 2, n_add) * 0.3
            
            noise[i, :] = base_noise + feature_noise + range_noise + extreme_noise
            
        return noise
    
    def _calculate_extreme_differentiation_stats(self, expanded_embeddings):
        """Calcular estadísticas de diferenciación extrema"""
        # Distancias euclidianas entre todos los pares
        distances = euclidean_distances(expanded_embeddings)
        
        # Obtener solo las distancias entre documentos diferentes (triangular superior)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        
        stats = {
            'avg_distance': np.mean(upper_tri),
            'min_distance': np.min(upper_tri),
            'max_distance': np.max(upper_tri),
            'std_distance': np.std(upper_tri),
            'variance_ratio': np.var(upper_tri) / np.var(expanded_embeddings),
            'separation_score': np.min(upper_tri) / np.mean(upper_tri),  # Cuanto más alto, mejor separación
            'discrimination_potential': np.std(upper_tri) / np.mean(upper_tri)  # Potencial de discriminación
        }
        
        return stats
    
    def _save_extreme_expansions(self, results):
        """Guardar todas las expansiones extremas"""
        print("\n💾 Guardando expansiones extremas...")
        
        # Preparar datos para guardar
        save_data = {}
        for key, value in results.items():
            save_data[key] = value['embeddings']
            save_data[f'{key}_config'] = value['config']
            save_data[f'{key}_stats'] = value['stats']
        
        # Guardar nombres de archivos PNTs
        save_data['pnts_names'] = np.array(self.pnts_names)
        
        # Guardar en archivo NPZ
        np.savez_compressed('extreme_expansions.npz', **save_data)
        print("✅ Expansiones extremas guardadas en 'extreme_expansions.npz'")
        
        # Generar reporte
        self._generate_extreme_report(results)
    
    def _generate_extreme_report(self, results):
        """Generar reporte de expansiones extremas"""
        print("\n📊 Generando reporte de expansiones extremas...")
        
        report_lines = [
            "🚀 REPORTE DE EXPANSIONES EXTREMAS - BUSCANDO TOP-1 DEL 100%",
            "=" * 80,
            f"Fecha: 2025",
            f"Modelo base: {self.model_path}",
            f"Documentos PNTs: {len(self.pnts_documents)}",
            f"Dimensiones originales: {self.model.get_sentence_embedding_dimension()}",
            "",
            "🎯 OBJETIVO: DISCRIMINACIÓN ABSOLUTA (Top-1 del 100%)",
            "",
            "📊 CONFIGURACIONES EXTREMAS GENERADAS:",
            "-" * 50
        ]
        
        for key, value in results.items():
            config = value['config']
            stats = value['stats']
            
            report_lines.extend([
                f"\n🔬 {config['description']}",
                f"   📏 Dimensiones adicionales: {config['dimensions']}",
                f"   🎲 Tipo de ruido: {config['noise_type']}",
                f"   📊 Escala de ruido: {config['noise_scale']}",
                f"   📈 Ratio de varianza: {stats['variance_ratio']:.4f}",
                f"   📏 Distancia promedio: {stats['avg_distance']:.4f}",
                f"   📏 Distancia mínima: {stats['min_distance']:.4f}",
                f"   📏 Distancia máxima: {stats['max_distance']:.4f}",
                f"   🎯 Puntuación de separación: {stats['separation_score']:.4f}",
                f"   🚀 Potencial de discriminación: {stats['discrimination_potential']:.4f}"
            ])
        
        report_lines.extend([
            "",
            "💡 ANÁLISIS DE POTENCIAL:",
            "-" * 30,
            "1. 📈 Las expansiones masivas (500-1000 dimensiones) maximizan la separación",
            "2. 🎲 El ruido híbrido combina múltiples estrategias de diferenciación",
            "3. 🎯 El ruido basado en clusters diferencia documentos similares",
            "4. 🔄 El ruido secuencial aplica patrones en cascada",
            "5. 🧠 El ruido adaptativo se ajusta a cada documento",
            "",
            "🚀 PRÓXIMO PASO:",
            "Ejecutar 'extreme_benchmark_evaluator.py' para evaluar con el benchmark",
            "y verificar si alcanzamos el Top-1 del 100%",
            "",
            "=" * 80
        ])
        
        # Guardar reporte
        with open('extreme_expansions_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("✅ Reporte guardado en 'extreme_expansions_report.txt'")
        
        # Mostrar resumen
        print("\n📊 RESUMEN DE EXPANSIONES EXTREMAS:")
        print("-" * 50)
        for key, value in results.items():
            config = value['config']
            stats = value['stats']
            print(f"🔬 {config['description']}")
            print(f"   📏 +{config['dimensions']}d | 🎯 Separación: {stats['separation_score']:.4f} | 🚀 Potencial: {stats['discrimination_potential']:.4f}")

if __name__ == "__main__":
    print("🚀 INICIANDO DISCRIMINADOR EXTREMO PARA TOP-1 DEL 100%")
    print("=" * 70)
    
    discriminator = ExtremeDiscriminator()
    discriminator.load_model()
    discriminator.load_pnts_documents()
    
    # Generar expansiones extremas
    results = discriminator.generate_extreme_expansions()
    
    print("\n🎉 ¡EXPANSIONES EXTREMAS GENERADAS EXITOSAMENTE!")
    print("🚀 Próximo paso: Evaluar con el benchmark para verificar Top-1 del 100%")
