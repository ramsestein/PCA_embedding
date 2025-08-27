#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expansor Dimensional Agresivo para Hiper-Especialización
Autor: Análisis de Embeddings Médicos
Fecha: 2025
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import warnings
warnings.filterwarnings('ignore')

class AggressiveDimensionalExpander:
    """Sistema de expansión dimensional agresiva para hiper-especialización"""
    
    def __init__(self, model_path="all-mini-base", pnts_folder="PNTs"):
        self.model_path = model_path
        self.pnts_folder = pnts_folder
        self.model = None
        self.pnts_documents = []
        self.pnts_names = []
        
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
        
        # Cargar todos los archivos .txt
        txt_files = [f for f in os.listdir(self.pnts_folder) if f.endswith('.txt')]
        
        for filename in txt_files:
            file_path = os.path.join(self.pnts_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Solo documentos no vacíos
                        self.pnts_documents.append(content)
                        self.pnts_names.append(filename)
            except Exception as e:
                print(f"   ⚠️ Error al leer {filename}: {e}")
        
        print(f"✅ Documentos cargados: {len(self.pnts_documents)}")
        return len(self.pnts_documents) > 0
    
    def generate_aggressive_expansions(self):
        """Genera expansiones dimensionales agresivas"""
        print("\n🚀 GENERANDO EXPANSIONES DIMENSIONALES AGRESIVAS")
        print("=" * 60)
        
        # Generar embeddings originales
        print("📊 Generando embeddings originales...")
        original_embeddings = self.model.encode(self.pnts_documents)
        print(f"✅ Embeddings originales: {original_embeddings.shape}")
        
        # Configuraciones de expansión agresiva
        expansion_configs = [
            # Expansión moderada pero agresiva
            {'dimensions': 100, 'noise_type': 'uniform', 'noise_scale': 0.3, 'description': 'Uniforme Muy Agresivo (100d)'},
            {'dimensions': 150, 'noise_type': 'uniform', 'noise_scale': 0.4, 'description': 'Uniforme Ultra Agresivo (150d)'},
            {'dimensions': 200, 'noise_type': 'uniform', 'noise_scale': 0.5, 'description': 'Uniforme Extremo (200d)'},
            
            # Expansión con ruido gaussiano agresivo
            {'dimensions': 100, 'noise_type': 'gaussian', 'noise_scale': 0.25, 'description': 'Gaussiano Muy Agresivo (100d)'},
            {'dimensions': 150, 'noise_type': 'gaussian', 'noise_scale': 0.35, 'description': 'Gaussiano Ultra Agresivo (150d)'},
            {'dimensions': 200, 'noise_type': 'gaussian', 'noise_scale': 0.45, 'description': 'Gaussiano Extremo (200d)'},
            
            # Expansión mixta (combinación de tipos)
            {'dimensions': 150, 'noise_type': 'mixed', 'noise_scale': 0.3, 'description': 'Mixto Agresivo (150d)'},
            {'dimensions': 200, 'noise_type': 'mixed', 'noise_scale': 0.4, 'description': 'Mixto Ultra Agresivo (200d)'},
        ]
        
        results = {}
        
        for i, config in enumerate(expansion_configs):
            print(f"\n🔬 EXPANSIÓN {i+1}/{len(expansion_configs)}: {config['description']}")
            print("-" * 50)
            
            # Generar expansión
            expanded_embeddings = self._expand_embeddings_aggressive(
                original_embeddings, 
                config
            )
            
            # Calcular estadísticas de diferenciación
            differentiation_stats = self._calculate_differentiation_stats(expanded_embeddings)
            
            # Guardar resultados
            results[f'expansion_{i+1}'] = {
                'config': config,
                'embeddings': expanded_embeddings,
                'stats': differentiation_stats
            }
            
            print(f"   ✅ Expansión completada: {expanded_embeddings.shape}")
            print(f"   📊 Ratio de varianza: {differentiation_stats['variance_ratio']:.4f}")
            print(f"   📏 Distancia promedio: {differentiation_stats['avg_distance']:.4f}")
        
        # Guardar todos los resultados
        self._save_aggressive_expansions(results)
        
        return results
    
    def _expand_embeddings_aggressive(self, original_embeddings, config):
        """Expande embeddings de forma agresiva"""
        n_orig = original_embeddings.shape[1]  # 384
        n_add = config['dimensions']
        n_docs = original_embeddings.shape[0]
        
        # Crear matriz expandida
        expanded = np.zeros((n_docs, n_orig + n_add))
        
        # Copiar dimensiones originales
        expanded[:, :n_orig] = original_embeddings
        
        # Generar dimensiones adicionales con ruido agresivo
        if config['noise_type'] == 'mixed':
            # Combinar diferentes tipos de ruido
            half = n_add // 2
            expanded[:, n_orig:n_orig+half] = self._generate_uniform_noise(
                original_embeddings, half, config['noise_scale']
            )
            expanded[:, n_orig+half:] = self._generate_gaussian_noise(
                original_embeddings, n_add - half, config['noise_scale']
            )
        else:
            # Ruido uniforme o gaussiano
            if config['noise_type'] == 'uniform':
                expanded[:, n_orig:] = self._generate_uniform_noise(
                    original_embeddings, n_add, config['noise_scale']
                )
            else:  # gaussian
                expanded[:, n_orig:] = self._generate_gaussian_noise(
                    original_embeddings, n_add, config['noise_scale']
                )
        
        return expanded
    
    def _generate_uniform_noise(self, original_embeddings, n_add, noise_scale):
        """Genera ruido uniforme agresivo"""
        n_docs = original_embeddings.shape[0]
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Calcular estadísticas del documento
            doc_std = np.std(original_embeddings[i])
            doc_mean = np.mean(original_embeddings[i])
            
            # Ruido uniforme agresivo
            scaled_range = doc_std * noise_scale
            noise[i] = np.random.uniform(
                low=doc_mean - scaled_range,
                high=doc_mean + scaled_range,
                size=n_add
            )
        
        return noise
    
    def _generate_gaussian_noise(self, original_embeddings, n_add, noise_scale):
        """Genera ruido gaussiano agresivo"""
        n_docs = original_embeddings.shape[0]
        noise = np.zeros((n_docs, n_add))
        
        for i in range(n_docs):
            # Calcular estadísticas del documento
            doc_std = np.std(original_embeddings[i])
            doc_mean = np.mean(original_embeddings[i])
            
            # Ruido gaussiano agresivo
            scaled_std = doc_std * noise_scale
            noise[i] = np.random.normal(
                loc=doc_mean,
                scale=scaled_std,
                size=n_add
            )
        
        return noise
    
    def _calculate_differentiation_stats(self, expanded_embeddings):
        """Calcula estadísticas de diferenciación"""
        # Calcular distancias entre todos los pares de documentos
        from sklearn.metrics.pairwise import euclidean_distances
        
        distances = euclidean_distances(expanded_embeddings)
        
        # Estadísticas de diferenciación
        avg_distance = np.mean(distances)
        min_distance = np.min(distances[distances > 0])  # Excluir diagonal
        max_distance = np.max(distances)
        
        # Ratio de varianza (mayor = mejor diferenciación)
        variance_ratio = np.var(expanded_embeddings) / np.var(expanded_embeddings[:, :384])
        
        return {
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'variance_ratio': variance_ratio
        }
    
    def _save_aggressive_expansions(self, results):
        """Guarda todas las expansiones agresivas"""
        print("\n💾 Guardando expansiones agresivas...")
        
        # Guardar en archivo NPZ
        save_data = {
            'pnts_names': self.pnts_names,
            'original_embeddings': self.model.encode(self.pnts_documents)
        }
        
        # Agregar cada expansión
        for key, result in results.items():
            save_data[f'{key}_embeddings'] = result['embeddings']
            save_data[f'{key}_config'] = result['config']
            save_data[f'{key}_stats'] = result['stats']
        
        np.savez_compressed(
            'aggressive_expansions.npz',
            **save_data
        )
        
        # Generar reporte
        self._generate_aggressive_report(results)
        
        print("✅ Expansiones agresivas guardadas en: aggressive_expansions.npz")
    
    def _generate_aggressive_report(self, results):
        """Genera reporte de expansiones agresivas"""
        with open('aggressive_expansions_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE EXPANSIONES DIMENSIONALES AGRESIVAS\n")
            f.write("OBJETIVO: Hiper-especialización para benchmark PNTs\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("CONFIGURACIONES DE EXPANSIÓN:\n")
            f.write("-" * 40 + "\n")
            
            for key, result in results.items():
                config = result['config']
                stats = result['stats']
                
                f.write(f"\n{config['description']}:\n")
                f.write(f"  • Dimensiones adicionales: {config['dimensions']}\n")
                f.write(f"  • Tipo de ruido: {config['noise_type']}\n")
                f.write(f"  • Escala de ruido: {config['noise_scale']}\n")
                f.write(f"  • Dimensiones totales: {384 + config['dimensions']}\n")
                f.write(f"  • Ratio de varianza: {stats['variance_ratio']:.4f}\n")
                f.write(f"  • Distancia promedio: {stats['avg_distance']:.4f}\n")
                f.write(f"  • Distancia mínima: {stats['min_distance']:.4f}\n")
                f.write(f"  • Distancia máxima: {stats['max_distance']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMENDACIONES PARA HIPER-ESPECIALIZACIÓN:\n")
            f.write("-" * 50 + "\n")
            f.write("1. Usar expansiones con mayor ratio de varianza\n")
            f.write("2. Priorizar configuraciones con mayor distancia promedio\n")
            f.write("3. Evaluar con benchmark para confirmar mejoras\n")
            f.write("4. Considerar expansiones de 150-200 dimensiones\n")
        
        print("✅ Reporte generado: aggressive_expansions_report.txt")

def main():
    """Función principal para expansión agresiva"""
    print("🚀 INICIANDO EXPANSIÓN DIMENSIONAL AGRESIVA")
    print("=" * 60)
    
    # Crear expandidor
    expander = AggressiveDimensionalExpander()
    
    # Cargar modelo y documentos
    if not expander.load_model():
        return
    
    if not expander.load_pnts_documents():
        return
    
    # Generar expansiones agresivas
    results = expander.generate_aggressive_expansions()
    
    print("\n🎉 EXPANSIÓN AGRESIVA COMPLETADA!")
    print(f"\n📊 Total de expansiones generadas: {len(results)}")
    print(f"📁 Archivos generados:")
    print(f"   - aggressive_expansions.npz")
    print(f"   - aggressive_expansions_report.txt")
    
    print(f"\n💡 Próximo paso: Evaluar estas expansiones con el benchmark")

if __name__ == "__main__":
    main()
