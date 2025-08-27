#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISCRIMINADOR FINAL - TOP-1 DEL 100% MEDIANTE ESTRATEGIA EQUILIBRADA
Autor: Análisis de Embeddings Médicos
Fecha: 2025
Objetivo: Discriminación absoluta preservando semántica al 100%
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import warnings
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
warnings.filterwarnings('ignore')

class FinalDiscriminator:
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
        
    def generate_final_expansions(self):
        """Generar expansiones finales para Top-1 del 100%"""
        print("🚀 GENERANDO EXPANSIONES FINALES PARA TOP-1 DEL 100%...")
        
        original_embeddings = self.model.encode(self.pnts_documents)
        print(f"📊 Embeddings originales: {original_embeddings.shape}")
        
        # ESTRATEGIAS FINALES para discriminación absoluta
        final_configs = [
            # ESTRATEGIA 1: Diferenciación por similitud semántica
            {
                'dimensions': 200,
                'strategy': 'semantic_similarity_differentiation',
                'noise_scale': 0.1,
                'description': 'Diferenciación por Similitud Semántica (200d)'
            },
            
            # ESTRATEGIA 2: Huella digital única por documento
            {
                'dimensions': 300,
                'strategy': 'unique_document_fingerprint',
                'noise_scale': 0.15,
                'description': 'Huella Digital Única (300d)'
            },
            
            # ESTRATEGIA 3: Diferenciación progresiva inteligente
            {
                'dimensions': 400,
                'strategy': 'progressive_intelligent_differentiation',
                'noise_scale': 0.2,
                'description': 'Diferenciación Progresiva Inteligente (400d)'
            },
            
            # ESTRATEGIA 4: Balance perfecto semántica-diferenciación
            {
                'dimensions': 500,
                'strategy': 'perfect_semantic_differentiation_balance',
                'noise_scale': 0.25,
                'description': 'Balance Perfecto Semántica-Diferenciación (500d)'
            },
            
            # ESTRATEGIA 5: Diferenciación adaptativa por contenido
            {
                'dimensions': 600,
                'strategy': 'content_adaptive_differentiation',
                'noise_scale': 0.3,
                'description': 'Diferenciación Adaptativa por Contenido (600d)'
            }
        ]
        
        results = {}
        for i, config in enumerate(final_configs):
            print(f"\n🔬 Generando: {config['description']}")
            expanded_embeddings = self._expand_embeddings_final(original_embeddings, config)
            differentiation_stats = self._calculate_final_differentiation_stats(expanded_embeddings)
            
            results[f'final_expansion_{i+1}'] = {
                'config': config,
                'embeddings': expanded_embeddings,
                'stats': differentiation_stats
            }
            
            print(f"   ✅ Ratio de varianza: {differentiation_stats['variance_ratio']:.4f}")
            print(f"   ✅ Distancia promedio: {differentiation_stats['avg_distance']:.4f}")
            print(f"   ✅ Distancia mínima: {differentiation_stats['min_distance']:.4f}")
            print(f"   🎯 Puntuación de separación: {differentiation_stats['separation_score']:.4f}")
            print(f"   🧠 Preservación semántica: {differentiation_stats['semantic_preservation']:.4f}")
            
        self._save_final_expansions(results)
        return results
    
    def _expand_embeddings_final(self, original_embeddings, config):
        """Expansión final con estrategias equilibradas"""
        n_orig = original_embeddings.shape[1]
        n_add = config['dimensions']
        n_docs = original_embeddings.shape[0]
        
        expanded = np.zeros((n_docs, n_orig + n_add))
        expanded[:, :n_orig] = original_embeddings  # Preservar embeddings originales al 100%
        
        strategy = config['strategy']
        noise_scale = config['noise_scale']
        
        if strategy == 'semantic_similarity_differentiation':
            noise = self._generate_semantic_similarity_differentiation(original_embeddings, n_add, noise_scale)
        elif strategy == 'unique_document_fingerprint':
            noise = self._generate_unique_document_fingerprint(original_embeddings, n_add, noise_scale)
        elif strategy == 'progressive_intelligent_differentiation':
            noise = self._generate_progressive_intelligent_differentiation(original_embeddings, n_add, noise_scale)
        elif strategy == 'perfect_semantic_differentiation_balance':
            noise = self._generate_perfect_semantic_differentiation_balance(original_embeddings, n_add, noise_scale)
        elif strategy == 'content_adaptive_differentiation':
            noise = self._generate_content_adaptive_differentiation(original_embeddings, n_add, noise_scale)
        else:
            noise = np.random.uniform(-noise_scale, noise_scale, (n_docs, n_add))
            
        expanded[:, n_orig:] = noise
        return expanded
    
    def _generate_semantic_similarity_differentiation(self, original_embeddings, n_add, noise_scale):
        """Diferenciación basada en similitud semántica"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular similitud coseno entre documentos
        similarities = cosine_similarity(original_embeddings)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Encontrar documentos más similares
            similar_docs = np.argsort(similarities[i])[1:4]  # Top 3 más similares
            
            # Generar ruido que diferencie de documentos similares
            base_noise = np.random.uniform(-noise_scale * 0.3, noise_scale * 0.3, n_add)
            
            # Ruido de diferenciación específica
            diff_noise = np.zeros(n_add)
            for j, similar_doc in enumerate(similar_docs):
                similarity_score = similarities[i][similar_doc]
                # Cuanto más similar, más diferenciación
                doc_noise = np.random.uniform(-noise_scale * 0.4, noise_scale * 0.4, n_add)
                diff_noise += doc_noise * similarity_score * 0.6
                
            noise[i, :] = base_noise + diff_noise
            
        return noise
    
    def _generate_unique_document_fingerprint(self, original_embeddings, n_add, noise_scale):
        """Huella digital única por documento"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular características únicas por documento
        doc_means = np.mean(original_embeddings, axis=1)
        doc_stds = np.std(original_embeddings, axis=1)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Ruido base único del documento
            doc_scale = noise_scale * (1 + doc_stds[i] * 0.2)
            base_noise = np.random.uniform(-doc_scale, doc_scale, n_add)
            
            # Ruido de características específicas
            feature_noise = np.random.normal(doc_means[i] * 0.02, doc_stds[i] * 0.05, n_add)
            
            # Ruido de diferenciación única
            unique_noise = np.random.uniform(-noise_scale * 0.5, noise_scale * 0.5, n_add)
            
            noise[i, :] = base_noise + feature_noise + unique_noise
            
        return noise
    
    def _generate_progressive_intelligent_differentiation(self, original_embeddings, n_add, noise_scale):
        """Diferenciación progresiva inteligente"""
        n_docs = original_embeddings.shape[0]
        
        # Dividir dimensiones en secciones progresivas
        n_sections = 5
        section_size = n_add // n_sections
        
        noise = np.zeros((n_docs, n_add))
        
        for section in range(n_sections):
            start_idx = section * section_size
            end_idx = start_idx + section_size if section < n_sections - 1 else n_add
            
            # Escala progresiva: más diferenciación en secciones posteriores
            section_scale = noise_scale * (0.3 + section * 0.15)
            
            # Ruido base progresivo
            section_noise = np.random.uniform(-section_scale, section_scale, (n_docs, end_idx - start_idx))
            
            # Ruido de diferenciación progresiva
            if section > 0:
                # Usar información de secciones anteriores
                prev_noise = noise[:, :start_idx]
                prev_mean = np.mean(prev_noise, axis=1, keepdims=True)
                section_noise += prev_mean * 0.05
                
            noise[:, start_idx:end_idx] = section_noise
            
        return noise
    
    def _generate_perfect_semantic_differentiation_balance(self, original_embeddings, n_add, noise_scale):
        """Balance perfecto entre semántica y diferenciación"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular similitud semántica
        similarities = cosine_similarity(original_embeddings)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Encontrar documentos más similares y diferentes
            similar_docs = np.argsort(similarities[i])[1:4]  # Top 3 más similares
            different_docs = np.argsort(similarities[i])[-4:-1]  # Top 3 más diferentes
            
            # Ruido base balanceado
            base_noise = np.random.uniform(-noise_scale * 0.4, noise_scale * 0.4, n_add)
            
            # Ruido de diferenciación de similares
            diff_similar = np.zeros(n_add)
            for similar_doc in similar_docs:
                similarity_score = similarities[i][similar_doc]
                doc_noise = np.random.uniform(-noise_scale * 0.5, noise_scale * 0.5, n_add)
                diff_similar += doc_noise * similarity_score * 0.4
                
            # Ruido de preservación de diferentes
            preserve_different = np.zeros(n_add)
            for different_doc in different_docs:
                similarity_score = similarities[i][different_doc]
                doc_noise = np.random.uniform(-noise_scale * 0.2, noise_scale * 0.2, n_add)
                preserve_different += doc_noise * (1 - similarity_score) * 0.3
                
            noise[i, :] = base_noise + diff_similar + preserve_different
            
        return noise
    
    def _generate_content_adaptive_differentiation(self, original_embeddings, n_add, noise_scale):
        """Diferenciación adaptativa por contenido"""
        n_docs = original_embeddings.shape[0]
        
        # Calcular características del contenido
        doc_means = np.mean(original_embeddings, axis=1)
        doc_stds = np.std(original_embeddings, axis=1)
        doc_maxs = np.max(original_embeddings, axis=1)
        doc_mins = np.min(original_embeddings, axis=1)
        
        # Calcular similitud semántica
        similarities = cosine_similarity(original_embeddings)
        
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Encontrar documentos más similares
            similar_docs = np.argsort(similarities[i])[1:4]
            
            # Ruido base adaptativo
            base_scale = noise_scale * (1 + doc_stds[i] * 0.3)
            base_noise = np.random.uniform(-base_scale, base_scale, n_add)
            
            # Ruido de características específicas
            feature_noise = np.random.normal(doc_means[i] * 0.03, doc_stds[i] * 0.08, n_add)
            
            # Ruido de diferenciación adaptativa
            diff_noise = np.zeros(n_add)
            for similar_doc in similar_docs:
                similarity_score = similarities[i][similar_doc]
                doc_noise = np.random.uniform(-noise_scale * 0.6, noise_scale * 0.6, n_add)
                diff_noise += doc_noise * similarity_score * 0.5
                
            # Ruido de rango adaptativo
            range_noise = np.random.uniform(doc_mins[i] * 0.02, doc_maxs[i] * 0.02, n_add)
            
            noise[i, :] = base_noise + feature_noise + diff_noise + range_noise
            
        return noise
    
    def _calculate_final_differentiation_stats(self, expanded_embeddings):
        """Calcular estadísticas de diferenciación final"""
        # Distancias euclidianas entre todos los pares
        distances = euclidean_distances(expanded_embeddings)
        
        # Obtener solo las distancias entre documentos diferentes (triangular superior)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        
        # Calcular similitud coseno para preservación semántica
        similarities = cosine_similarity(expanded_embeddings)
        upper_sim = similarities[np.triu_indices_from(similarities, k=1)]
        
        # Calcular similitud de los embeddings originales (primeras 384 dimensiones)
        original_embeddings = expanded_embeddings[:, :384]
        original_similarities = cosine_similarity(original_embeddings)
        original_upper_sim = original_similarities[np.triu_indices_from(original_similarities, k=1)]
        
        stats = {
            'avg_distance': np.mean(upper_tri),
            'min_distance': np.min(upper_tri),
            'max_distance': np.max(upper_tri),
            'std_distance': np.std(upper_tri),
            'variance_ratio': np.var(upper_tri) / np.var(expanded_embeddings),
            'separation_score': np.min(upper_tri) / np.mean(upper_tri),
            'discrimination_potential': np.std(upper_tri) / np.mean(upper_tri),
            'semantic_preservation': np.mean(upper_sim),  # Similitud en espacio expandido
            'original_semantic_preservation': np.mean(original_upper_sim),  # Similitud original
            'semantic_consistency': np.corrcoef(upper_sim, original_upper_sim)[0, 1],  # Correlación entre similitudes
            'differentiation_efficiency': np.min(upper_tri) / (np.mean(upper_sim) + 1e-8)  # Eficiencia de diferenciación
        }
        
        return stats
    
    def _save_final_expansions(self, results):
        """Guardar todas las expansiones finales"""
        print("\n💾 Guardando expansiones finales...")
        
        # Preparar datos para guardar
        save_data = {}
        for key, value in results.items():
            save_data[key] = value['embeddings']
            save_data[f'{key}_config'] = value['config']
            save_data[f'{key}_stats'] = value['stats']
        
        # Guardar nombres de archivos PNTs
        save_data['pnts_names'] = np.array(self.pnts_names)
        
        # Guardar en archivo NPZ
        np.savez_compressed('final_expansions.npz', **save_data)
        print("✅ Expansiones finales guardadas en 'final_expansions.npz'")
        
        # Generar reporte
        self._generate_final_report(results)
    
    def _generate_final_report(self, results):
        """Generar reporte de expansiones finales"""
        print("\n📊 Generando reporte de expansiones finales...")
        
        report_lines = [
            "🚀 REPORTE DE EXPANSIONES FINALES - TOP-1 DEL 100%",
            "=" * 80,
            f"Fecha: 2025",
            f"Modelo base: {self.model_path}",
            f"Documentos PNTs: {len(self.pnts_documents)}",
            f"Dimensiones originales: {self.model.get_sentence_embedding_dimension()}",
            "",
            "🎯 OBJETIVO: TOP-1 DEL 100% MEDIANTE ESTRATEGIA EQUILIBRADA",
            "",
            "📊 CONFIGURACIONES FINALES GENERADAS:",
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
                f"   🧠 Preservación semántica original: {stats['original_semantic_preservation']:.4f}",
                f"   🔗 Consistencia semántica: {stats['semantic_consistency']:.4f}",
                f"   ⚡ Eficiencia de diferenciación: {stats['differentiation_efficiency']:.4f}"
            ])
        
        report_lines.extend([
            "",
            "💡 ANÁLISIS DE ESTRATEGIAS FINALES:",
            "-" * 40,
            "1. 🧠 Diferenciación por Similitud Semántica: Diferenciar solo lo similar",
            "2. 🎯 Huella Digital Única: Identidad única por documento",
            "3. 📈 Diferenciación Progresiva Inteligente: Aumento gradual y controlado",
            "4. ⚖️ Balance Perfecto: Equilibrio entre semántica y diferenciación",
            "5. 🔄 Diferenciación Adaptativa: Ajuste automático por contenido",
            "",
            "🚀 PRÓXIMO PASO:",
            "Ejecutar 'final_benchmark_evaluator.py' para evaluar con el benchmark",
            "y verificar si alcanzamos el Top-1 del 100%",
            "",
            "=" * 80
        ])
        
        # Guardar reporte
        with open('final_expansions_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("✅ Reporte guardado en 'final_expansions_report.txt'")
        
        # Mostrar resumen
        print("\n📊 RESUMEN DE EXPANSIONES FINALES:")
        print("-" * 60)
        for key, value in results.items():
            config = value['config']
            stats = value['stats']
            print(f"🔬 {config['description']}")
            print(f"   📏 +{config['dimensions']}d | 🎯 Separación: {stats['separation_score']:.4f} | 🧠 Semántica: {stats['semantic_preservation']:.4f}")

if __name__ == "__main__":
    print("🚀 INICIANDO DISCRIMINADOR FINAL PARA TOP-1 DEL 100%")
    print("=" * 80)
    
    discriminator = FinalDiscriminator()
    discriminator.load_model()
    discriminator.load_pnts_documents()
    
    # Generar expansiones finales
    results = discriminator.generate_final_expansions()
    
    print("\n🎉 ¡EXPANSIONES FINALES GENERADAS EXITOSAMENTE!")
    print("🚀 Próximo paso: Evaluar con el benchmark para verificar Top-1 del 100%")
