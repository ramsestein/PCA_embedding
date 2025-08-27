#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador de Augmentación Semántica - Estrategia 2
Implementa técnicas de augmentación semántica para mejorar la discriminación
"""

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import time
from pathlib import Path
import random

class SemanticAugmentationEvaluator:
    """
    Implementa la Estrategia 2: Técnicas de Augmentación Semántica
    - Paráfrasis semántica
    - Expansión de términos relacionados
    - Generación de variaciones contextuales
    - Evaluación contra benchmark real
    """
    
    def __init__(self, model_path: str = "all-mini-base"):
        self.model_path = model_path
        self.model = None
        self.pnts_documents = []
        self.pnts_filenames = []
        self.benchmark_data = []
        
        # Resultados de evaluación
        self.baseline_results = {}
        self.augmentation_results = {}
        
        # Diccionario de términos médicos relacionados
        self.medical_terms = {
            'paciente': ['usuario', 'usuario', 'individuo', 'persona'],
            'medicamento': ['fármaco', 'medicina', 'droga', 'tratamiento'],
            'hospitalización': ['ingreso', 'internamiento', 'estancia hospitalaria'],
            'domicilio': ['casa', 'hogar', 'residencia', 'vivienda'],
            'tratamiento': ['terapia', 'cuidado', 'intervención', 'procedimiento'],
            'enfermedad': ['patología', 'afección', 'condición', 'síndrome'],
            'síntoma': ['manifestación', 'signo', 'indicador', 'señal'],
            'diagnóstico': ['evaluación', 'valoración', 'análisis', 'determinación'],
            'medicina': ['farmacología', 'terapéutica', 'clínica', 'asistencial'],
            'cuidado': ['atención', 'asistencia', 'vigilancia', 'supervisión']
        }
    
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
    
    def generate_semantic_paraphrases(self, text: str, num_variations: int = 3) -> List[str]:
        """Genera paráfrasis semánticas del texto"""
        variations = [text]  # Incluir texto original
        
        # Técnica 1: Reemplazo de términos médicos relacionados
        for _ in range(num_variations):
            paraphrased = text
            for term, alternatives in self.medical_terms.items():
                if term in paraphrased.lower():
                    alternative = random.choice(alternatives)
                    paraphrased = paraphrased.replace(term, alternative)
                    paraphrased = paraphrased.replace(term.capitalize(), alternative.capitalize())
            if paraphrased != text:
                variations.append(paraphrased)
        
        # Técnica 2: Reestructuración de frases
        sentences = text.split('. ')
        if len(sentences) > 1:
            for _ in range(min(2, num_variations)):
                shuffled_sentences = sentences.copy()
                random.shuffle(shuffled_sentences)
                restructured = '. '.join(shuffled_sentences)
                if restructured != text:
                    variations.append(restructured)
        
        # Técnica 3: Sinónimos y variaciones léxicas
        synonyms = {
            'cómo': ['de qué manera', 'de qué forma', 'cuál es la forma'],
            'qué': ['cuál', 'qué tipo de', 'qué clase de'],
            'cuándo': ['en qué momento', 'en qué instante', 'a qué hora'],
            'dónde': ['en qué lugar', 'en qué sitio', 'en qué ubicación'],
            'quién': ['qué persona', 'qué profesional', 'qué individuo']
        }
        
        for _ in range(min(2, num_variations)):
            paraphrased = text
            for word, alternatives in synonyms.items():
                if word in paraphrased.lower():
                    alternative = random.choice(alternatives)
                    paraphrased = paraphrased.replace(word, alternative)
                    paraphrased = paraphrased.replace(word.capitalize(), alternative.capitalize())
            if paraphrased != text and paraphrased not in variations:
                variations.append(paraphrased)
        
        # Asegurar que no haya duplicados y limitar el número
        unique_variations = list(dict.fromkeys(variations))[:num_variations + 1]
        return unique_variations
    
    def generate_contextual_variations(self, text: str, num_variations: int = 2) -> List[str]:
        """Genera variaciones contextuales del texto"""
        variations = [text]
        
        # Técnica 1: Añadir contexto médico
        medical_contexts = [
            "En el contexto médico, ",
            "Desde la perspectiva clínica, ",
            "En términos de asistencia sanitaria, ",
            "Considerando la práctica médica, ",
            "En el ámbito de la salud, "
        ]
        
        for _ in range(min(2, num_variations)):
            context = random.choice(medical_contexts)
            variation = context + text
            if variation not in variations:
                variations.append(variation)
        
        # Técnica 2: Formular como pregunta directa vs indirecta
        if text.endswith('?'):
            # Convertir pregunta directa a indirecta
            indirect_variations = [
                f"Necesito saber {text.lower()}",
                f"Me gustaría conocer {text.lower()}",
                f"Quisiera que me expliques {text.lower()}"
            ]
            for variation in indirect_variations:
                if variation not in variations:
                    variations.append(variation)
        else:
            # Convertir a pregunta directa
            direct_variations = [
                f"¿{text}?",
                f"¿Podrías explicarme {text.lower()}?",
                f"¿Cuál es la información sobre {text.lower()}?"
            ]
            for variation in direct_variations:
                if variation not in variations:
                    variations.append(variation)
        
        return variations[:num_variations + 1]
    
    def apply_semantic_augmentation(self):
        """Aplica técnicas de augmentación semántica a los documentos PNTs"""
        print("🔄 Aplicando augmentación semántica...")
        
        augmented_documents = []
        augmented_filenames = []
        
        for i, (doc, filename) in enumerate(zip(self.pnts_documents, self.pnts_filenames)):
            # Documento original
            augmented_documents.append(doc)
            augmented_filenames.append(f"{filename} (Original)")
            
            # Generar paráfrasis semánticas
            paraphrases = self.generate_semantic_paraphrases(doc, num_variations=2)
            for j, paraphrase in enumerate(paraphrases[1:], 1):  # Excluir el original
                augmented_documents.append(paraphrase)
                augmented_filenames.append(f"{filename} (Paráfrasis {j})")
            
            # Generar variaciones contextuales
            contextual_vars = self.generate_contextual_variations(doc, num_variations=1)
            for j, context_var in enumerate(contextual_vars[1:], 1):  # Excluir el original
                augmented_documents.append(context_var)
                augmented_filenames.append(f"{filename} (Contexto {j})")
        
        self.augmented_documents = augmented_documents
        self.augmented_filenames = augmented_filenames
        
        print(f"✅ Documentos aumentados: {len(self.augmented_documents)} (originales: {len(self.pnts_documents)})")
        print(f"   - Aumento: {len(self.augmented_documents) - len(self.pnts_documents)} variaciones generadas")
    
    def generate_augmented_embeddings(self):
        """Genera embeddings para documentos originales y aumentados"""
        print("🔄 Generando embeddings aumentados...")
        
        # Embeddings de documentos originales
        self.original_embeddings = self.model.encode(
            self.pnts_documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✅ Embeddings originales: {self.original_embeddings.shape}")
        
        # Embeddings de documentos aumentados
        self.augmented_embeddings = self.model.encode(
            self.augmented_documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✅ Embeddings aumentados: {self.augmented_embeddings.shape}")
        
        # Embeddings de consultas del benchmark
        self.query_embeddings = self.model.encode(
            [item['query'] for item in self.benchmark_data],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✅ Embeddings de consultas: {self.query_embeddings.shape}")
    
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
                
                # Distancia híbrida
                hybrid_dist = (1 - cos_sim) + (eucl_dist / np.max(eucl_dist)) if np.max(eucl_dist) > 0 else (1 - cos_sim)
                
                # Determinar el nombre del archivo correctamente
                if 'augmented' in method_name or len(doc_emb) > len(self.pnts_filenames):
                    # Usar nombres de archivos aumentados
                    filename = self.augmented_filenames[j] if j < len(self.augmented_filenames) else f"Document_{j}"
                else:
                    # Usar nombres de archivos originales
                    filename = self.pnts_filenames[j] if j < len(self.pnts_filenames) else f"Document_{j}"
                
                query_similarities.append({
                    'document_index': j,
                    'filename': filename,
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
                    # Determinar el nombre del archivo correcto
                    if 'augmented' in method_name or len(doc_emb) > len(self.pnts_filenames):
                        # Usar nombres de archivos aumentados
                        filename = self.augmented_filenames[doc_idx] if doc_idx < len(self.augmented_filenames) else f"Document_{doc_idx}"
                        
                        # CORRECCIÓN: Buscar coincidencia basada en el nombre base del archivo
                        # El benchmark espera "filename.txt", pero tenemos "filename.txt (Paráfrasis 1)"
                        base_filename = filename.split(' (')[0]  # Extraer solo la parte antes del paréntesis
                        if base_filename == expected_doc:
                            expected_found = True
                    else:
                        # Usar nombres de archivos originales
                        filename = self.pnts_filenames[doc_idx] if doc_idx < len(self.pnts_filenames) else f"Document_{doc_idx}"
                        if filename == expected_doc:
                            expected_found = True
                    
                    if expected_found:
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
    
    def run_semantic_augmentation_evaluation(self):
        """Ejecuta la evaluación completa de augmentación semántica"""
        print("🚀 INICIANDO EVALUACIÓN DE AUGMENTACIÓN SEMÁNTICA")
        print("=" * 80)
        
        try:
            # PASO 1: Cargar modelo
            self.load_model()
            
            # PASO 2: Cargar documentos PNTs
            self.load_pnts_documents()
            
            # PASO 3: Cargar benchmark
            self.load_benchmark()
            
            # PASO 4: Aplicar augmentación semántica
            print("\n📊 APLICANDO ESTRATEGIA 2: Augmentación Semántica")
            print("-" * 50)
            self.apply_semantic_augmentation()
            
            # PASO 5: Generar embeddings
            self.generate_augmented_embeddings()
            
            # PASO 6: Evaluar baseline (embeddings originales)
            print("\n📊 EVALUANDO BASELINE (Documentos Originales)")
            print("-" * 50)
            self.baseline_results = self.evaluate_retrieval_performance(
                self.original_embeddings, 
                self.query_embeddings, 
                "Baseline_Original"
            )
            
            # PASO 7: Evaluar augmentación semántica
            print("\n📊 EVALUANDO AUGMENTACIÓN SEMÁNTICA")
            print("-" * 50)
            self.augmentation_results = self.evaluate_retrieval_performance(
                self.augmented_embeddings,
                self.query_embeddings,
                "Semantic_Augmentation"
            )
            
            # PASO 8: Generar reporte
            print("\n📊 GENERANDO REPORTE")
            print("-" * 50)
            self.generate_augmentation_report()
            
            print("\n🎉 EVALUACIÓN DE AUGMENTACIÓN SEMÁNTICA COMPLETADA!")
            
        except Exception as e:
            print(f"❌ ERROR EN LA EVALUACIÓN: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_augmentation_report(self):
        """Genera un reporte de los resultados de augmentación semántica"""
        print("🔄 Generando reporte de augmentación...")
        
        # Preparar datos para el reporte
        report_data = []
        
        # Baseline
        baseline = self.baseline_results
        report_data.append({
            'Método': 'Baseline (Original)',
            'Documentos': len(self.pnts_documents),
            'Dimensiones': self.original_embeddings.shape[1],
            'Top-1 Accuracy': f"{baseline['top1_accuracy']:.4f}",
            'Top-3 Accuracy': f"{baseline['top3_accuracy']:.4f}",
            'Top-5 Accuracy': f"{baseline['top5_accuracy']:.4f}",
            'MRR': f"{baseline['mrr']:.4f}"
        })
        
        # Augmentación semántica
        augmentation = self.augmentation_results
        report_data.append({
            'Método': 'Augmentación Semántica',
            'Documentos': len(self.augmented_documents),
            'Dimensiones': self.augmented_embeddings.shape[1],
            'Top-1 Accuracy': f"{augmentation['top1_accuracy']:.4f}",
            'Top-3 Accuracy': f"{augmentation['top3_accuracy']:.4f}",
            'Top-5 Accuracy': f"{augmentation['top5_accuracy']:.4f}",
            'MRR': f"{augmentation['mrr']:.4f}"
        })
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(report_data)
        
        # Guardar reporte CSV
        csv_path = "semantic_augmentation_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Reporte CSV guardado: {csv_path}")
        
        # Guardar reporte detallado en texto
        txt_path = "semantic_augmentation_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE AUGMENTACIÓN SEMÁNTICA - ESTRATEGIA 2\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total consultas evaluadas: {len(self.benchmark_data)}\n")
            f.write(f"Documentos originales: {len(self.pnts_documents)}\n")
            f.write(f"Documentos aumentados: {len(self.augmented_documents)}\n")
            f.write(f"Variaciones generadas: {len(self.augmented_documents) - len(self.pnts_documents)}\n\n")
            
            f.write("RESULTADOS DE EVALUACIÓN:\n")
            f.write("-" * 40 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Análisis detallado
            f.write("ANÁLISIS DETALLADO:\n")
            f.write("-" * 40 + "\n")
            
            # Comparación con baseline
            baseline_mrr = float(baseline['mrr'])
            augmentation_mrr = float(augmentation['mrr'])
            improvement = ((augmentation_mrr - baseline_mrr) / baseline_mrr) * 100
            
            f.write(f"📊 COMPARACIÓN CON BASELINE:\n")
            f.write(f"   Baseline MRR: {baseline_mrr:.4f}\n")
            f.write(f"   Augmentación MRR: {augmentation_mrr:.4f}\n")
            f.write(f"   Mejora MRR: {improvement:+.2f}%\n\n")
            
            f.write(f"📊 MÉTRICAS DE ACCURACY:\n")
            f.write(f"   Top-1: {baseline['top1_accuracy']:.4f} → {augmentation['top1_accuracy']:.4f}\n")
            f.write(f"   Top-3: {baseline['top3_accuracy']:.4f} → {augmentation['top3_accuracy']:.4f}\n")
            f.write(f"   Top-5: {baseline['top5_accuracy']:.4f} → {augmentation['top5_accuracy']:.4f}\n\n")
            
            if improvement > 0:
                f.write("🎉 CONCLUSIÓN: La augmentación semántica MEJORA el rendimiento!\n")
            else:
                f.write("⚠️  CONCLUSIÓN: La augmentación semántica NO mejora el rendimiento.\n")
        
        print(f"✅ Reporte detallado guardado: {txt_path}")
        
        # Generar visualización
        self.generate_augmentation_visualization(df)
    
    def generate_augmentation_visualization(self, df):
        """Genera visualizaciones de los resultados de augmentación"""
        print("🔄 Generando visualizaciones...")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Evaluación de Augmentación Semántica - Estrategia 2', fontsize=16, fontweight='bold')
        
        # 1. Comparación de MRR
        ax1 = axes[0, 0]
        methods = df['Método']
        mrr_values = [float(x) for x in df['MRR']]
        colors = ['#2E86AB', '#A23B72']
        
        bars1 = ax1.bar(methods, mrr_values, color=colors, alpha=0.8)
        ax1.set_title('Mean Reciprocal Rank (MRR)', fontweight='bold')
        ax1.set_ylabel('MRR')
        ax1.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, value in zip(bars1, mrr_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=12)
        
        # 2. Comparación de Top-1 Accuracy
        ax2 = axes[0, 1]
        top1_values = [float(x) for x in df['Top-1 Accuracy']]
        bars2 = ax2.bar(methods, top1_values, color=colors, alpha=0.8)
        ax2.set_title('Top-1 Accuracy', fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, top1_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=12)
        
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
        
        # 4. Número de documentos
        ax4 = axes[1, 1]
        doc_counts = df['Documentos']
        bars5 = ax4.bar(methods, doc_counts, color=colors, alpha=0.8)
        ax4.set_title('Número de Documentos', fontweight='bold')
        ax4.set_ylabel('Cantidad')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars5, doc_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        # Guardar visualización
        viz_path = "semantic_augmentation_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualización guardada: {viz_path}")
        
        plt.show()

def main():
    """Función principal"""
    print("🚀 EVALUADOR DE AUGMENTACIÓN SEMÁNTICA")
    print("Implementa la Estrategia 2: Técnicas de Augmentación Semántica")
    print("=" * 80)
    
    try:
        # Crear evaluador
        evaluator = SemanticAugmentationEvaluator()
        
        # Ejecutar evaluación
        evaluator.run_semantic_augmentation_evaluation()
        
        print("\n🎯 EVALUACIÓN COMPLETADA")
        print("Revisa los archivos generados:")
        print("  📊 semantic_augmentation_results.csv - Resultados en formato tabla")
        print("  📋 semantic_augmentation_report.txt - Reporte detallado")
        print("  🖼️  semantic_augmentation_visualization.png - Visualizaciones")
        
    except Exception as e:
        print(f"❌ ERROR EN LA EJECUCIÓN: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
