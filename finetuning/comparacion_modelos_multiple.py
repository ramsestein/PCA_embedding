#!/usr/bin/env python3
"""
comparacion_modelos_serie.py - Script para comparar múltiples modelos en serie
Compara all-MiniLM-base contra todos los modelos en la carpeta models
"""

import json
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import openpyxl

# Importar tu cargador de modelos
from cargador_models import UnifiedEmbeddingAdapter

class BatchEmbeddingComparator:
    """Comparador de embeddings en lote"""
    
    def __init__(self, base_model_config, models_folder, documents_path, test_queries_path):
        self.base_model_config = base_model_config
        self.models_folder = Path(models_folder)
        self.documents_path = Path(documents_path)
        self.test_queries_path = test_queries_path
        self.all_results = {}
        
        # Cargar modelo base
        print("📦 Cargando modelo base...")
        self.base_model = UnifiedEmbeddingAdapter(
            model_path=base_model_config['path'],
            model_name=base_model_config['name'],
            pooling_strategy=base_model_config.get('pooling', 'mean')
        )
        
        # Crear embeddings del modelo base una sola vez
        print("📚 Creando embeddings del modelo base...")
        self.base_doc_chunks, self.base_chunk_to_doc = self.create_document_embeddings(self.base_model)
        
        # Cargar queries de prueba
        with open(self.test_queries_path, 'r', encoding='utf-8') as f:
            self.test_cases = json.load(f)
    
    def find_all_models(self):
        """Encuentra todos los modelos en la carpeta models"""
        models_found = []
        
        # Buscar en cada subcarpeta
        for model_type_folder in self.models_folder.iterdir():
            if model_type_folder.is_dir():
                # Buscar modelos dentro de cada subcarpeta
                for model_folder in model_type_folder.iterdir():
                    if model_folder.is_dir() and model_folder.name.startswith('model-'):
                        models_found.append({
                            'path': str(model_folder),
                            'name': f"{model_type_folder.name}/{model_folder.name}",
                            'type': model_type_folder.name,
                            'version': model_folder.name
                        })
        
        print(f"\n🔍 Encontrados {len(models_found)} modelos para comparar:")
        for model in sorted(models_found, key=lambda x: x['name']):
            print(f"   • {model['name']}")
        
        return models_found
    
    def chunk_text(self, text, chunk_size=384, chunk_overlap=128):
        """Divide texto en chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_document_embeddings(self, model):
        """Crea embeddings para todos los documentos"""
        doc_embeddings = {}
        all_chunks = []
        chunk_to_doc = {}
        
        # Procesar cada documento
        doc_files = list(self.documents_path.glob("*.txt"))
        
        for doc_file in tqdm(doc_files, desc="Indexando documentos", leave=False):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                continue
            
            # Dividir en chunks
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                # Generar embedding
                embedding = model.embed(chunk)
                
                # Guardar información
                chunk_id = f"{doc_file.name}_{i}"
                all_chunks.append({
                    'id': chunk_id,
                    'text': chunk,
                    'embedding': embedding,
                    'doc_name': doc_file.name,
                    'chunk_pos': i
                })
                chunk_to_doc[chunk_id] = doc_file.name
        
        return all_chunks, chunk_to_doc
    
    def search_similar_chunks(self, query_embedding, doc_chunks, top_k=5):
        """Busca chunks similares"""
        # Calcular similitudes
        similarities = []
        
        for chunk_info in doc_chunks:
            # Similitud coseno
            cos_sim = np.dot(query_embedding, chunk_info['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_info['embedding'])
            )
            similarity = cos_sim * 100
            
            similarities.append({
                'doc_name': chunk_info['doc_name'],
                'chunk_text': chunk_info['text'],
                'similarity': similarity,
                'chunk_pos': chunk_info['chunk_pos']
            })
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def evaluate_model(self, model_config):
        """Evalúa un modelo específico"""
        try:
            # Cargar modelo
            model = UnifiedEmbeddingAdapter(
                model_path=model_config['path'],
                model_name=model_config['name'],
                pooling_strategy='mean'
            )
            
            # Crear embeddings de documentos
            doc_chunks, chunk_to_doc = self.create_document_embeddings(model)
            
            results = []
            total_time = 0
            
            for case in tqdm(self.test_cases, desc=f"Evaluando {model_config['name']}", leave=False):
                query = case['query']
                expected_doc = case['document_expected']
                
                # Generar embedding de la query
                start_time = time.time()
                query_embedding = model.embed(query)
                embed_time = time.time() - start_time
                
                # Buscar
                start_time = time.time()
                similar_chunks = self.search_similar_chunks(query_embedding, doc_chunks)
                search_time = time.time() - start_time
                
                total_time += embed_time + search_time
                
                # Verificar si encontró el documento esperado
                found = False
                position = -1
                similarity = 0
                top_doc = similar_chunks[0]['doc_name'] if similar_chunks else None
                
                for i, chunk in enumerate(similar_chunks):
                    if chunk['doc_name'] == expected_doc:
                        found = True
                        position = i + 1
                        similarity = chunk['similarity']
                        break
                
                results.append({
                    'query': query,
                    'expected_doc': expected_doc,
                    'found': found,
                    'position': position,
                    'similarity': similarity,
                    'top_doc': top_doc
                })
            
            # Calcular métricas
            metrics = self.calculate_metrics(results)
            metrics['avg_query_time'] = total_time / len(self.test_cases)
            
            return {
                'metrics': metrics,
                'results': results,
                'model_info': {
                    'name': model_config['name'],
                    'path': model_config['path'],
                    'dimension': model.get_dimension()
                }
            }
            
        except Exception as e:
            print(f"\n❌ Error al evaluar {model_config['name']}: {str(e)}")
            return None
    
    def calculate_metrics(self, results):
        """Calcula métricas de evaluación"""
        total = len(results)
        
        acc_at_1 = sum(1 for r in results if r['position'] == 1) / total
        acc_at_5 = sum(1 for r in results if r['found']) / total
        
        mrr = sum(1/r['position'] for r in results if r['position'] > 0) / total
        
        found_results = [r for r in results if r['found']]
        avg_similarity = sum(r['similarity'] for r in found_results) / len(found_results) if found_results else 0
        
        return {
            'accuracy_at_1': acc_at_1,
            'accuracy_at_5': acc_at_5,
            'mrr': mrr,
            'avg_similarity': avg_similarity,
            'not_found': sum(1 for r in results if not r['found']),
            'total_queries': total
        }
    
    def run_batch_comparison(self):
        """Ejecuta la comparación en lote"""
        print("\n" + "="*60)
        print("COMPARACIÓN EN LOTE DE EMBEDDINGS")
        print("="*60)
        
        # Evaluar modelo base
        print(f"\n📊 Evaluando modelo base: {self.base_model_config['name']}")
        base_results = {
            'metrics': {},
            'results': [],
            'model_info': {
                'name': self.base_model_config['name'],
                'path': self.base_model_config['path'],
                'dimension': self.base_model.get_dimension()
            }
        }
        
        # Evaluar queries con modelo base
        results = []
        total_time = 0
        
        for case in tqdm(self.test_cases, desc="Evaluando modelo base"):
            query = case['query']
            expected_doc = case['document_expected']
            
            start_time = time.time()
            query_embedding = self.base_model.embed(query)
            embed_time = time.time() - start_time
            
            start_time = time.time()
            similar_chunks = self.search_similar_chunks(query_embedding, self.base_doc_chunks)
            search_time = time.time() - start_time
            
            total_time += embed_time + search_time
            
            found = False
            position = -1
            similarity = 0
            top_doc = similar_chunks[0]['doc_name'] if similar_chunks else None
            
            for i, chunk in enumerate(similar_chunks):
                if chunk['doc_name'] == expected_doc:
                    found = True
                    position = i + 1
                    similarity = chunk['similarity']
                    break
            
            results.append({
                'query': query,
                'expected_doc': expected_doc,
                'found': found,
                'position': position,
                'similarity': similarity,
                'top_doc': top_doc
            })
        
        base_results['results'] = results
        base_results['metrics'] = self.calculate_metrics(results)
        base_results['metrics']['avg_query_time'] = total_time / len(self.test_cases)
        
        self.all_results['base_model'] = base_results
        
        # Encontrar todos los modelos
        models_to_compare = self.find_all_models()
        
        # Evaluar cada modelo
        print(f"\n🔄 Evaluando {len(models_to_compare)} modelos...")
        
        for i, model_config in enumerate(models_to_compare, 1):
            print(f"\n[{i}/{len(models_to_compare)}] Evaluando: {model_config['name']}")
            
            result = self.evaluate_model(model_config)
            if result:
                self.all_results[model_config['name']] = result
        
        return self.all_results
    
    def get_top_models(self, n=5, metric='accuracy_at_1'):
        """Obtiene los N mejores modelos según una métrica"""
        # Ordenar modelos por métrica
        model_scores = []
        
        for model_name, data in self.all_results.items():
            if model_name != 'base_model':  # Excluir modelo base del ranking
                score = data['metrics'][metric]
                model_scores.append({
                    'name': model_name,
                    'score': score,
                    'metrics': data['metrics'],
                    'info': data['model_info']
                })
        
        # Ordenar por score descendente
        model_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return model_scores[:n]
    
    def print_final_report(self):
        """Imprime el reporte final con todos los modelos"""
        print("\n" + "="*100)
        print("REPORTE FINAL - COMPARACIÓN DE MODELOS")
        print("="*100)
        
        # Modelo base
        base_metrics = self.all_results['base_model']['metrics']
        print(f"\n📊 MODELO BASE: {self.base_model_config['name']}")
        print(f"   • Accuracy@1: {base_metrics['accuracy_at_1']:.1%}")
        print(f"   • Accuracy@5: {base_metrics['accuracy_at_5']:.1%}")
        print(f"   • MRR: {base_metrics['mrr']:.3f}")
        print(f"   • Similitud promedio: {base_metrics['avg_similarity']:.1f}%")
        print(f"   • Velocidad: {base_metrics['avg_query_time']:.4f}s/query")
        
        # Obtener TODOS los modelos ordenados
        all_models = self.get_top_models(n=len(self.all_results)-1, metric='accuracy_at_1')
        
        # Tabla de todos los modelos
        print("\n" + "="*100)
        print("📊 RANKING COMPLETO DE MODELOS (ordenados por Accuracy@1)")
        print("="*100)
        
        # Encabezado de tabla
        print(f"\n{'#':<4} {'Modelo':<35} {'Acc@1':<10} {'vs Base':<12} {'Acc@5':<10} {'MRR':<8} {'Sim Avg':<10} {'Vel (s)':<10}")
        print("-"*100)
        
        for i, model in enumerate(all_models, 1):
            metrics = model['metrics']
            
            # Calcular mejora respecto al modelo base
            improvement = (metrics['accuracy_at_1'] - base_metrics['accuracy_at_1']) * 100
            improvement_str = f"{'+' if improvement > 0 else ''}{improvement:.1f} pp"
            
            # Color coding en la terminal (opcional)
            if improvement > 10:
                marker = "🟢"  # Mejora significativa
            elif improvement > 5:
                marker = "🟡"  # Mejora moderada
            elif improvement > 0:
                marker = "⚪"  # Mejora leve
            else:
                marker = "🔴"  # Sin mejora
            
            print(f"{marker} {i:<2} {model['name']:<35} "
                  f"{metrics['accuracy_at_1']:<10.1%} "
                  f"{improvement_str:<12} "
                  f"{metrics['accuracy_at_5']:<10.1%} "
                  f"{metrics['mrr']:<8.3f} "
                  f"{metrics['avg_similarity']:<10.1f} "
                  f"{metrics['avg_query_time']:<10.4f}")
        
        # Estadísticas generales
        print("\n" + "="*100)
        print("📈 ESTADÍSTICAS GENERALES")
        print("="*100)
        
        # Calcular estadísticas
        all_acc1 = [m['metrics']['accuracy_at_1'] for m in all_models]
        all_mrr = [m['metrics']['mrr'] for m in all_models]
        
        models_better_than_base = sum(1 for m in all_models 
                                     if m['metrics']['accuracy_at_1'] > base_metrics['accuracy_at_1'])
        
        print(f"\n• Total de modelos evaluados: {len(all_models)}")
        print(f"• Modelos que superan al base: {models_better_than_base} ({models_better_than_base/len(all_models)*100:.1f}%)")
        print(f"• Rango de Accuracy@1: {min(all_acc1):.1%} - {max(all_acc1):.1%}")
        print(f"• Promedio de Accuracy@1: {np.mean(all_acc1):.1%}")
        print(f"• Promedio de MRR: {np.mean(all_mrr):.3f}")
        
        # Top 5 con detalles
        print("\n" + "="*100)
        print("🏆 TOP 5 MEJORES MODELOS - ANÁLISIS DETALLADO")
        print("="*100)
        
        top_5 = all_models[:5]
        
        for i, model in enumerate(top_5, 1):
            metrics = model['metrics']
            improvement = (metrics['accuracy_at_1'] - base_metrics['accuracy_at_1']) * 100
            speed_ratio = base_metrics['avg_query_time'] / metrics['avg_query_time']
            
            print(f"\n{i}. {model['name']}")
            print(f"   Ruta: {model['info']['path']}")
            print(f"   • Accuracy@1: {metrics['accuracy_at_1']:.1%} ({'+' if improvement > 0 else ''}{improvement:.1f} pp vs base)")
            print(f"   • Accuracy@5: {metrics['accuracy_at_5']:.1%}")
            print(f"   • MRR: {metrics['mrr']:.3f}")
            print(f"   • Similitud promedio: {metrics['avg_similarity']:.1f}%")
            print(f"   • Velocidad: {speed_ratio:.2f}x {'más rápido' if speed_ratio > 1 else 'más lento'} que el base")
            print(f"   • Queries no encontradas: {metrics['not_found']}/{metrics['total_queries']}")
    
    def save_full_results(self, output_dir="resultados_comparacion_batch"):
        """Guarda todos los resultados"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar resumen de todos los modelos
        summary_data = []
        
        for model_name, data in self.all_results.items():
            metrics = data['metrics']
            summary_data.append({
                'model': model_name,
                'path': data['model_info'].get('path', 'N/A'),
                'accuracy_at_1': metrics['accuracy_at_1'],
                'accuracy_at_5': metrics['accuracy_at_5'],
                'mrr': metrics['mrr'],
                'avg_similarity': metrics['avg_similarity'],
                'not_found': metrics['not_found'],
                'total_queries': metrics['total_queries'],
                'avg_query_time': metrics['avg_query_time'],
                'dimension': data['model_info']['dimension']
            })
        
        # Crear DataFrame y ordenar por accuracy@1
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('accuracy_at_1', ascending=False)
        
        # Guardar CSV completo
        df_summary.to_csv(f"{output_dir}/resumen_todos_modelos_{timestamp}.csv", index=False)
        
        # Guardar Excel con formato (si openpyxl está disponible)
        excel_saved = False
        try:
            with pd.ExcelWriter(f"{output_dir}/resultados_completos_{timestamp}.xlsx", engine='openpyxl') as writer:
                # Hoja 1: Resumen
                df_summary.to_excel(writer, sheet_name='Resumen', index=False)
                
                # Hoja 2: Top 10
                df_summary.head(10).to_excel(writer, sheet_name='Top 10', index=False)
                
                # Hoja 3: Comparación con base
                df_comp = df_summary.copy()
                base_metrics = self.all_results['base_model']['metrics']
                df_comp['mejora_acc1_pp'] = (df_comp['accuracy_at_1'] - base_metrics['accuracy_at_1']) * 100
                df_comp['mejora_mrr'] = df_comp['mrr'] - base_metrics['mrr']
                df_comp = df_comp[df_comp['model'] != 'base_model']
                df_comp.to_excel(writer, sheet_name='Comparacion con Base', index=False)
            excel_saved = True
        except:
            print("   ⚠️  No se pudo crear archivo Excel (instalar openpyxl)")
        
        # Guardar reporte completo de texto
        with open(f"{output_dir}/reporte_completo_{timestamp}.txt", 'w', encoding='utf-8') as f:
            # Redirigir la salida del print_final_report
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            self.print_final_report()
            sys.stdout = old_stdout
        
        # Guardar un archivo markdown con formato bonito
        with open(f"{output_dir}/reporte_completo_{timestamp}.md", 'w', encoding='utf-8') as f:
            f.write("# Reporte de Comparación de Modelos de Embeddings\n\n")
            f.write(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Modelo base
            base_metrics = self.all_results['base_model']['metrics']
            f.write("## Modelo Base\n\n")
            f.write(f"**Nombre**: {self.base_model_config['name']}\n\n")
            f.write("| Métrica | Valor |\n")
            f.write("|---------|-------|\n")
            f.write(f"| Accuracy@1 | {base_metrics['accuracy_at_1']:.1%} |\n")
            f.write(f"| Accuracy@5 | {base_metrics['accuracy_at_5']:.1%} |\n")
            f.write(f"| MRR | {base_metrics['mrr']:.3f} |\n")
            f.write(f"| Similitud promedio | {base_metrics['avg_similarity']:.1f}% |\n")
            f.write(f"| Velocidad | {base_metrics['avg_query_time']:.4f}s/query |\n\n")
            
            # Tabla completa
            f.write("## Ranking Completo de Modelos\n\n")
            f.write("| # | Modelo | Acc@1 | vs Base | Acc@5 | MRR | Sim Avg | Vel (s) |\n")
            f.write("|---|--------|-------|---------|-------|-----|---------|-------|\n")
            
            all_models = self.get_top_models(n=len(self.all_results)-1, metric='accuracy_at_1')
            
            for i, model in enumerate(all_models, 1):
                metrics = model['metrics']
                improvement = (metrics['accuracy_at_1'] - base_metrics['accuracy_at_1']) * 100
                improvement_str = f"{'+' if improvement > 0 else ''}{improvement:.1f}pp"
                
                f.write(f"| {i} | {model['name']} | "
                       f"{metrics['accuracy_at_1']:.1%} | "
                       f"{improvement_str} | "
                       f"{metrics['accuracy_at_5']:.1%} | "
                       f"{metrics['mrr']:.3f} | "
                       f"{metrics['avg_similarity']:.1f}% | "
                       f"{metrics['avg_query_time']:.4f} |\n")
            
            # Top 5 detallado
            f.write("\n## Top 5 Mejores Modelos - Análisis Detallado\n\n")
            top_5 = all_models[:5]
            
            for i, model in enumerate(top_5, 1):
                metrics = model['metrics']
                improvement = (metrics['accuracy_at_1'] - base_metrics['accuracy_at_1']) * 100
                speed_ratio = base_metrics['avg_query_time'] / metrics['avg_query_time']
                
                f.write(f"### {i}. {model['name']}\n\n")
                f.write(f"**Ruta**: `{model['info']['path']}`\n\n")
                f.write("| Métrica | Valor | Comparación con Base |\n")
                f.write("|---------|-------|---------------------|\n")
                f.write(f"| Accuracy@1 | {metrics['accuracy_at_1']:.1%} | "
                       f"{'+' if improvement > 0 else ''}{improvement:.1f} pp |\n")
                f.write(f"| Accuracy@5 | {metrics['accuracy_at_5']:.1%} | - |\n")
                f.write(f"| MRR | {metrics['mrr']:.3f} | "
                       f"{'+' if metrics['mrr'] > base_metrics['mrr'] else ''}"
                       f"{metrics['mrr'] - base_metrics['mrr']:.3f} |\n")
                f.write(f"| Velocidad | {metrics['avg_query_time']:.4f}s | "
                       f"{speed_ratio:.2f}x {'más rápido' if speed_ratio > 1 else 'más lento'} |\n")
                f.write(f"| Queries no encontradas | {metrics['not_found']}/{metrics['total_queries']} | - |\n\n")
        
        # Crear gráficos comparativos
        self.create_comparison_chart(df_summary, output_dir, timestamp)
        
        print(f"\n📁 Resultados guardados en: {output_dir}/")
        print(f"   • resumen_todos_modelos_{timestamp}.csv")
        if excel_saved:
            print(f"   • resultados_completos_{timestamp}.xlsx")
        print(f"   • reporte_completo_{timestamp}.txt")
        print(f"   • reporte_completo_{timestamp}.md")
    
    def create_comparison_chart(self, df_summary, output_dir, timestamp):
        """Crea gráfico comparativo de todos los modelos"""
        # Limitar a máximo 20 modelos para que sea legible
        max_models = min(20, len(df_summary))
        df_plot = df_summary.head(max_models)
        
        # Crear figura más grande
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Gráfico 1: Accuracy@1 de todos los modelos
        colors = ['#2ecc71' if 'base' in name.lower() else '#3498db' for name in df_plot['model']]
        bars1 = ax1.barh(range(len(df_plot)), df_plot['accuracy_at_1'], color=colors)
        ax1.set_yticks(range(len(df_plot)))
        ax1.set_yticklabels(df_plot['model'], fontsize=8)
        ax1.set_xlabel('Accuracy@1')
        ax1.set_title(f'Top {max_models} Modelos - Accuracy@1')
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Añadir valores
        for i, (bar, val) in enumerate(zip(bars1, df_plot['accuracy_at_1'])):
            ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1%}', va='center', fontsize=8)
        
        # Línea del modelo base
        base_acc = df_summary[df_summary['model'] == 'base_model']['accuracy_at_1'].values
        if len(base_acc) > 0:
            ax1.axvline(x=base_acc[0], color='red', linestyle='--', alpha=0.5, label='Modelo Base')
            ax1.legend()
        
        # Gráfico 2: Comparación múltiple de métricas
        x = np.arange(len(df_plot))
        width = 0.25
        
        bars_acc1 = ax2.bar(x - width, df_plot['accuracy_at_1'], width, label='Accuracy@1', color='#3498db')
        bars_acc5 = ax2.bar(x, df_plot['accuracy_at_5'], width, label='Accuracy@5', color='#2ecc71')
        bars_mrr = ax2.bar(x + width, df_plot['mrr'], width, label='MRR', color='#e74c3c')
        
        ax2.set_xlabel('Modelos')
        ax2.set_ylabel('Score')
        ax2.set_title('Comparación de Métricas - Todos los Modelos')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.split('/')[-1] for m in df_plot['model']], rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparacion_todos_modelos_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Crear gráfico adicional de scatter para velocidad vs accuracy
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Usar todos los modelos para el scatter
        scatter = ax.scatter(df_summary['avg_query_time'], 
                           df_summary['accuracy_at_1'], 
                           s=df_summary['mrr']*300,  # Tamaño basado en MRR
                           c=['#2ecc71' if 'base' in name.lower() else '#3498db' 
                             for name in df_summary['model']], 
                           alpha=0.6, edgecolors='black', linewidth=1)
        
        ax.set_xlabel('Tiempo promedio por query (segundos)')
        ax.set_ylabel('Accuracy@1')
        ax.set_title('Velocidad vs Precisión (tamaño = MRR)')
        ax.grid(True, alpha=0.3)
        
        # Añadir etiquetas para top 10 y modelo base
        for i in range(min(10, len(df_summary))):
            ax.annotate(df_summary.iloc[i]['model'].split('/')[-1], 
                       (df_summary.iloc[i]['avg_query_time'], df_summary.iloc[i]['accuracy_at_1']),
                       xytext=(5, 5), textcoords='offset points', fontsize=7)
        
        # Resaltar modelo base
        base_data = df_summary[df_summary['model'] == 'base_model']
        if not base_data.empty:
            ax.annotate('BASE', 
                       (base_data.iloc[0]['avg_query_time'], base_data.iloc[0]['accuracy_at_1']),
                       xytext=(10, -10), textcoords='offset points', 
                       fontsize=9, fontweight='bold', color='green',
                       arrowprops=dict(arrowstyle='->', color='green'))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/velocidad_vs_precision_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gráficos guardados:")
        print(f"   • comparacion_todos_modelos_{timestamp}.png")
        print(f"   • velocidad_vs_precision_{timestamp}.png")


def main():
    print("🚀 Iniciando comparación en lote de embeddings\n")
    
    # Configuración del modelo base
    base_model_config = {
        'path': './all-mini-base',
        'name': 'all-MiniLM-base',
        'pooling': 'mean'
    }
    
    # Paths
    models_folder = './models'
    documents_path = './PNTs'
    test_queries_path = './preguntas_con_docs_es.json'
    
    # Verificar que existen los paths
    if not Path(base_model_config['path']).exists():
        print(f"❌ No se encuentra modelo base: {base_model_config['path']}")
        return
    
    if not Path(models_folder).exists():
        print(f"❌ No se encuentra carpeta de modelos: {models_folder}")
        return
    
    if not Path(documents_path).exists():
        print(f"❌ No se encuentra carpeta de documentos: {documents_path}")
        return
        
    if not Path(test_queries_path).exists():
        print(f"❌ No se encuentra archivo de pruebas: {test_queries_path}")
        return
    
    # Crear comparador
    comparator = BatchEmbeddingComparator(
        base_model_config=base_model_config,
        models_folder=models_folder,
        documents_path=documents_path,
        test_queries_path=test_queries_path
    )
    
    # Ejecutar comparación en lote
    start_time = time.time()
    results = comparator.run_batch_comparison()
    
    # Mostrar reporte final
    comparator.print_final_report()
    
    # Guardar todos los resultados
    comparator.save_full_results()
    
    total_time = time.time() - start_time
    print(f"\n✅ Comparación completada en {total_time/60:.1f} minutos")
    print(f"   Modelos evaluados: {len(results) - 1}")  # -1 por el modelo base


if __name__ == "__main__":
    main()