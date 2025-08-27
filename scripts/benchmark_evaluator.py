#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador de Benchmark para Modelo de Embeddings PNTs
Autor: Análisis de Embeddings Médicos
Fecha: 2025
"""

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
warnings.filterwarnings('ignore')

class BenchmarkEvaluator:
    """Evaluador del benchmark para medir rendimiento del modelo de embeddings"""
    
    def __init__(self, model_path="all-mini-base", pnts_folder="PNTs", benchmark_folder="benchmark"):
        """
        Inicializa el evaluador del benchmark
        
        Args:
            model_path (str): Ruta al modelo de embeddings
            pnts_folder (str): Carpeta que contiene los documentos PNTs
            benchmark_folder (str): Carpeta que contiene los archivos de benchmark
        """
        self.model_path = model_path
        self.pnts_folder = pnts_folder
        self.benchmark_folder = benchmark_folder
        self.model = None
        self.pnts_embeddings = None
        self.pnts_names = None
        self.benchmark_data = {}
        
    def load_model(self):
        """Carga el modelo de embeddings"""
        print("🔄 Cargando modelo de embeddings...")
        try:
            self.model = SentenceTransformer(self.model_path)
            print(f"✅ Modelo cargado exitosamente desde: {self.model_path}")
            print(f"📊 Dimensiones del embedding: {self.model.get_sentence_embedding_dimension()}")
            return True
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            return False
    
    def load_pnts_documents(self):
        """Carga todos los documentos PNTs y genera embeddings"""
        print(f"📁 Cargando documentos PNTs desde: {self.pnts_folder}")
        
        if not os.path.exists(self.pnts_folder):
            print(f"❌ Error: La carpeta {self.pnts_folder} no existe")
            return False
        
        documents = []
        document_names = []
        
        # Obtener todos los archivos .txt en la carpeta PNTs
        txt_files = [f for f in os.listdir(self.pnts_folder) if f.endswith('.txt')]
        
        if not txt_files:
            print(f"❌ Error: No se encontraron archivos .txt en {self.pnts_folder}")
            return False
        
        print(f"📄 Encontrados {len(txt_files)} documentos PNTs")
        
        for filename in txt_files:
            file_path = os.path.join(self.pnts_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                    # Solo procesar archivos con contenido
                    if content and len(content) > 10:
                        documents.append(content)
                        document_names.append(filename)
                        print(f"   ✅ {filename} ({len(content)} caracteres)")
                    else:
                        print(f"   ⚠️ {filename} - archivo vacío o muy corto")
                        
            except Exception as e:
                print(f"   ❌ Error al leer {filename}: {e}")
        
        if not documents:
            print("❌ Error: No se pudieron cargar documentos válidos")
            return False
        
        # Generar embeddings para todos los documentos PNTs
        print("🔄 Generando embeddings para documentos PNTs...")
        try:
            self.pnts_embeddings = self.model.encode(documents, show_progress_bar=True)
            self.pnts_names = document_names
            print(f"✅ Embeddings generados: {self.pnts_embeddings.shape}")
            return True
        except Exception as e:
            print(f"❌ Error al generar embeddings: {e}")
            return False
    
    def load_benchmark_data(self):
        """Carga los datos del benchmark en ambos idiomas"""
        print(f"📊 Cargando datos del benchmark desde: {self.benchmark_folder}")
        
        benchmark_files = {
            'catalan': 'preguntas_con_docs_cat.json',
            'spanish': 'preguntas_con_docs_es.json'
        }
        
        for language, filename in benchmark_files.items():
            file_path = os.path.join(self.benchmark_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.benchmark_data[language] = json.load(f)
                print(f"   ✅ {language.capitalize()}: {len(self.benchmark_data[language])} preguntas")
            except Exception as e:
                print(f"   ❌ Error al cargar {filename}: {e}")
                return False
        
        return True
    
    def evaluate_benchmark(self, language):
        """Evalúa el benchmark para un idioma específico"""
        if language not in self.benchmark_data:
            print(f"❌ Error: Idioma {language} no encontrado en el benchmark")
            return None
        
        print(f"🔍 Evaluando benchmark en {language.capitalize()}...")
        
        questions = self.benchmark_data[language]
        results = []
        
        for i, qa_pair in enumerate(questions):
            query = qa_pair['query']
            expected_doc = qa_pair['document_expected']
            
            # Generar embedding para la pregunta
            query_embedding = self.model.encode([query])
            
            # Calcular similitud con todos los documentos PNTs
            similarities = cosine_similarity(query_embedding, self.pnts_embeddings)[0]
            
            # Obtener ranking de documentos
            ranking = np.argsort(similarities)[::-1]  # Orden descendente
            
            # Encontrar posición del documento esperado
            expected_position = None
            for pos, idx in enumerate(ranking):
                if self.pnts_names[idx] == expected_doc:
                    expected_position = pos + 1
                    break
            
            # Calcular métricas
            top1_correct = expected_position == 1
            top3_correct = expected_position <= 3 if expected_position else False
            mrr = 1.0 / expected_position if expected_position else 0.0
            
            # Obtener top 3 documentos
            top3_docs = [self.pnts_names[idx] for idx in ranking[:3]]
            top3_scores = [similarities[idx] for idx in ranking[:3]]
            
            result = {
                'question_id': i + 1,
                'query': query,
                'expected_document': expected_doc,
                'expected_position': expected_position,
                'top1_correct': top1_correct,
                'top3_correct': top3_correct,
                'mrr': mrr,
                'top3_documents': top3_docs,
                'top3_scores': top3_scores,
                'max_similarity': similarities[ranking[0]]
            }
            
            results.append(result)
            
            # Mostrar progreso cada 10 preguntas
            if (i + 1) % 10 == 0:
                print(f"   Procesadas {i + 1}/{len(questions)} preguntas...")
        
        return results
    
    def calculate_metrics(self, results):
        """Calcula métricas agregadas del benchmark"""
        if not results:
            return None
        
        total_questions = len(results)
        
        # Métricas básicas
        top1_accuracy = sum(1 for r in results if r['top1_correct']) / total_questions
        top3_accuracy = sum(1 for r in results if r['top3_correct']) / total_questions
        mrr_score = np.mean([r['mrr'] for r in results])
        
        # Análisis de posiciones
        positions = [r['expected_position'] for r in results if r['expected_position']]
        avg_position = np.mean(positions) if positions else 0
        
        # Distribución de posiciones
        position_distribution = {}
        for pos in positions:
            position_distribution[pos] = position_distribution.get(pos, 0) + 1
        
        # Análisis de similitud
        max_similarities = [r['max_similarity'] for r in results]
        avg_max_similarity = np.mean(max_similarities)
        
        metrics = {
            'total_questions': total_questions,
            'top1_accuracy': top1_accuracy,
            'top3_accuracy': top3_accuracy,
            'mrr_score': mrr_score,
            'average_position': avg_position,
            'position_distribution': position_distribution,
            'average_max_similarity': avg_max_similarity,
            'results': results
        }
        
        return metrics
    
    def generate_report(self, metrics_cat, metrics_es):
        """Genera un reporte completo del benchmark"""
        print("📊 Generando reporte del benchmark...")
        
        # Crear reporte de texto
        with open('benchmark_baseline_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE BENCHMARK - MODELO BASELINE (384 dimensiones)\n")
            f.write("MODELO: all-MiniLM-L6-v2\n")
            f.write("=" * 80 + "\n\n")
            
            # Métricas en Catalán
            f.write("MÉTRICAS EN CATALÁN:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total de preguntas: {metrics_cat['total_questions']}\n")
            f.write(f"Top-1 Accuracy: {metrics_cat['top1_accuracy']:.4f} ({metrics_cat['top1_accuracy']*100:.2f}%)\n")
            f.write(f"Top-3 Accuracy: {metrics_cat['top3_accuracy']:.4f} ({metrics_cat['top3_accuracy']*100:.2f}%)\n")
            f.write(f"MRR Score: {metrics_cat['mrr_score']:.4f}\n")
            f.write(f"Posición promedio: {metrics_cat['average_position']:.2f}\n")
            f.write(f"Similitud máxima promedio: {metrics_cat['average_max_similarity']:.4f}\n\n")
            
            # Métricas en Español
            f.write("MÉTRICAS EN ESPAÑOL:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total de preguntas: {metrics_es['total_questions']}\n")
            f.write(f"Top-1 Accuracy: {metrics_es['top1_accuracy']:.4f} ({metrics_es['top1_accuracy']*100:.2f}%)\n")
            f.write(f"Top-3 Accuracy: {metrics_es['top3_accuracy']:.4f} ({metrics_es['top3_accuracy']*100:.2f}%)\n")
            f.write(f"MRR Score: {metrics_es['mrr_score']:.4f}\n")
            f.write(f"Posición promedio: {metrics_es['average_position']:.2f}\n")
            f.write(f"Similitud máxima promedio: {metrics_es['average_max_similarity']:.4f}\n\n")
            
            # Comparación entre idiomas
            f.write("COMPARACIÓN ENTRE IDIOMAS:\n")
            f.write("-" * 35 + "\n")
            f.write(f"Diferencia Top-1: {(metrics_cat['top1_accuracy'] - metrics_es['top1_accuracy'])*100:.2f} puntos porcentuales\n")
            f.write(f"Diferencia Top-3: {(metrics_cat['top3_accuracy'] - metrics_es['top3_accuracy'])*100:.2f} puntos porcentuales\n")
            f.write(f"Diferencia MRR: {metrics_cat['mrr_score'] - metrics_es['mrr_score']:.4f}\n\n")
            
            # Análisis de posiciones
            f.write("DISTRIBUCIÓN DE POSICIONES:\n")
            f.write("-" * 35 + "\n")
            f.write("Catalán:\n")
            for pos, count in sorted(metrics_cat['position_distribution'].items()):
                f.write(f"  Posición {pos}: {count} preguntas\n")
            f.write("\nEspañol:\n")
            for pos, count in sorted(metrics_es['position_distribution'].items()):
                f.write(f"  Posición {pos}: {count} preguntas\n")
        
        # Crear CSV con resultados detallados
        self._save_detailed_results(metrics_cat, metrics_es)
        
        print("✅ Reporte del benchmark generado:")
        print("   - benchmark_baseline_report.txt")
        print("   - benchmark_detailed_results.csv")
        
        return True
    
    def _save_detailed_results(self, metrics_cat, metrics_es):
        """Guarda resultados detallados en CSV"""
        # Preparar datos para CSV
        rows = []
        
        # Resultados en catalán
        for result in metrics_cat['results']:
            rows.append({
                'Idioma': 'Catalán',
                'ID_Pregunta': result['question_id'],
                'Pregunta': result['query'],
                'Documento_Esperado': result['expected_document'],
                'Posicion_Encontrada': result['expected_position'],
                'Top1_Correcto': result['top1_correct'],
                'Top3_Correcto': result['top3_correct'],
                'MRR': result['mrr'],
                'Top1_Documento': result['top3_documents'][0] if result['top3_documents'] else '',
                'Top1_Score': result['top3_scores'][0] if result['top3_scores'] else 0,
                'Similitud_Maxima': result['max_similarity']
            })
        
        # Resultados en español
        for result in metrics_es['results']:
            rows.append({
                'Idioma': 'Español',
                'ID_Pregunta': result['question_id'],
                'Pregunta': result['query'],
                'Documento_Esperado': result['expected_document'],
                'Posicion_Encontrada': result['expected_position'],
                'Top1_Correcto': result['top1_correct'],
                'Top3_Correcto': result['top3_correct'],
                'MRR': result['mrr'],
                'Top1_Documento': result['top3_documents'][0] if result['top3_documents'] else '',
                'Top1_Score': result['top3_scores'][0] if result['top3_scores'] else 0,
                'Similitud_Maxima': result['max_similarity']
            })
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(rows)
        df.to_csv('benchmark_detailed_results.csv', index=False, encoding='utf-8')

def main():
    """Función principal para ejecutar la evaluación del benchmark"""
    print("🚀 INICIANDO EVALUACIÓN DEL BENCHMARK - MODELO BASELINE")
    print("=" * 70)
    
    # Crear evaluador
    evaluator = BenchmarkEvaluator()
    
    # Cargar modelo
    if not evaluator.load_model():
        return
    
    # Cargar documentos PNTs
    if not evaluator.load_pnts_documents():
        return
    
    # Cargar datos del benchmark
    if not evaluator.load_benchmark_data():
        return
    
    # Evaluar benchmark en catalán
    print("\n🔍 EVALUANDO BENCHMARK EN CATALÁN...")
    results_cat = evaluator.evaluate_benchmark('catalan')
    metrics_cat = evaluator.calculate_metrics(results_cat)
    
    # Evaluar benchmark en español
    print("\n🔍 EVALUANDO BENCHMARK EN ESPAÑOL...")
    results_es = evaluator.evaluate_benchmark('spanish')
    metrics_es = evaluator.calculate_metrics(results_es)
    
    # Generar reporte
    evaluator.generate_report(metrics_cat, metrics_es)
    
    # Mostrar resumen
    print("\n🎉 EVALUACIÓN DEL BENCHMARK COMPLETADA!")
    print("\n📊 RESUMEN DE RESULTADOS:")
    print(f"Catalán - Top-1: {metrics_cat['top1_accuracy']*100:.2f}%, Top-3: {metrics_cat['top3_accuracy']*100:.2f}%, MRR: {metrics_cat['mrr_score']:.4f}")
    print(f"Español  - Top-1: {metrics_es['top1_accuracy']*100:.2f}%, Top-3: {metrics_es['top3_accuracy']*100:.2f}%, MRR: {metrics_es['mrr_score']:.4f}")
    print(f"\n💡 Este baseline te permitirá:")
    print("   • Comparar el rendimiento antes y después de la expansión dimensional")
    print("   • Identificar áreas de mejora específicas")
    print("   • Validar que la expansión no degrada el rendimiento")
    print("   • Medir el impacto real de la diferenciación mejorada")

if __name__ == "__main__":
    main()
