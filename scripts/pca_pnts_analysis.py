#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis PCA de Embeddings usando Documentos PNTs Reales
Autor: Análisis de Embeddings Médicos
Fecha: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib para español
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class PNTsEmbeddingAnalyzer:
    """Analizador de embeddings para documentos PNTs médicos"""
    
    def __init__(self, model_path="all-mini-base", pnts_folder="PNTs"):
        """
        Inicializa el analizador para documentos PNTs
        
        Args:
            model_path (str): Ruta al modelo de embeddings
            pnts_folder (str): Carpeta que contiene los documentos PNTs
        """
        self.model_path = model_path
        self.pnts_folder = pnts_folder
        self.model = None
        self.embeddings = None
        self.documents = None
        self.document_names = None
        self.pca = None
        self.scaler = None
        
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
        """Carga todos los documentos PNTs de la carpeta"""
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
                        # Limpiar el nombre del archivo para mejor visualización
                        clean_name = filename.replace('_limpio.txt', '').replace('_', ' ')
                        document_names.append(clean_name)
                        print(f"   ✅ {clean_name} ({len(content)} caracteres)")
                    else:
                        print(f"   ⚠️ {filename} - archivo vacío o muy corto")
                        
            except Exception as e:
                print(f"   ❌ Error al leer {filename}: {e}")
        
        if not documents:
            print("❌ Error: No se pudieron cargar documentos válidos")
            return False
        
        self.documents = documents
        self.document_names = document_names
        
        print(f"✅ {len(self.documents)} documentos PNTs cargados exitosamente")
        return True
    
    def segment_documents_into_sentences(self, min_length=20, max_length=500):
        """Segmenta los documentos en oraciones para análisis más granular"""
        print("✂️ Segmentando documentos en oraciones...")
        
        sentences = []
        sentence_sources = []
        
        for doc_idx, document in enumerate(self.documents):
            # Dividir por puntos, signos de exclamación e interrogación
            # Mantener solo oraciones con longitud razonable
            raw_sentences = re.split(r'[.!?]+', document)
            
            for sentence in raw_sentences:
                sentence = sentence.strip()
                # Filtrar oraciones por longitud y contenido
                if (len(sentence) >= min_length and 
                    len(sentence) <= max_length and
                    not sentence.isdigit() and
                    len(sentence.split()) >= 3):  # Al menos 3 palabras
                    
                    sentences.append(sentence)
                    sentence_sources.append(self.document_names[doc_idx])
        
        print(f"✅ {len(sentences)} oraciones extraídas de {len(self.documents)} documentos")
        
        # Mostrar algunas estadísticas
        sentence_lengths = [len(s.split()) for s in sentences]
        print(f"📊 Estadísticas de oraciones:")
        print(f"   • Longitud promedio: {np.mean(sentence_lengths):.1f} palabras")
        print(f"   • Longitud mínima: {min(sentence_lengths)} palabras")
        print(f"   • Longitud máxima: {max(sentence_lengths)} palabras")
        
        return sentences, sentence_sources
    
    def generate_embeddings_from_sentences(self, sentences):
        """Genera embeddings para las oraciones segmentadas"""
        if self.model is None:
            print("❌ Error: Modelo no cargado")
            return False
        
        if not sentences:
            print("❌ Error: No hay oraciones para procesar")
            return False
        
        print(f"🔄 Generando embeddings para {len(sentences)} oraciones...")
        try:
            self.embeddings = self.model.encode(sentences, show_progress_bar=True)
            print(f"✅ Embeddings generados: {self.embeddings.shape}")
            return True
        except Exception as e:
            print(f"❌ Error al generar embeddings: {e}")
            return False
    
    def perform_pca_analysis(self, n_components=20):
        """Realiza análisis de componentes principales"""
        if self.embeddings is None:
            print("❌ Error: No hay embeddings para analizar")
            return False
        
        print(f"🔄 Realizando PCA con {n_components} componentes...")
        
        # Normalización de los datos
        self.scaler = StandardScaler()
        embeddings_scaled = self.scaler.fit_transform(self.embeddings)
        
        # Aplicar PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        embeddings_pca = self.pca.fit_transform(embeddings_scaled)
        
        print(f"✅ PCA completado")
        print(f"📊 Varianza explicada por componente:")
        for i, var in enumerate(self.pca.explained_variance_ratio_):
            print(f"   Componente {i+1}: {var:.4f} ({var*100:.2f}%)")
        
        print(f"📈 Varianza acumulada: {np.sum(self.pca.explained_variance_ratio_):.4f} ({np.sum(self.pca.explained_variance_ratio_)*100:.2f}%)")
        
        return embeddings_pca
    
    def analyze_component_importance(self):
        """Analiza la importancia de cada componente"""
        if self.pca is None:
            print("❌ Error: PCA no realizado")
            return None
        
        # Crear DataFrame con información de componentes
        component_info = pd.DataFrame({
            'Componente': range(1, len(self.pca.explained_variance_ratio_) + 1),
            'Varianza_Explicada': self.pca.explained_variance_ratio_,
            'Varianza_Acumulada': np.cumsum(self.pca.explained_variance_ratio_),
            'Valor_Singular': np.sqrt(self.pca.explained_variance_)
        })
        
        return component_info
    
    def visualize_pca_results(self, embeddings_pca, sentence_sources):
        """Visualiza los resultados del PCA con información de documentos"""
        if self.pca is None or embeddings_pca is None:
            print("❌ Error: PCA no realizado")
            return
        
        print("🎨 Generando visualizaciones...")
        
        # Configurar el estilo de las gráficas
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Análisis PCA de Documentos PNTs - Modelo all-MiniLM-L6-v2', 
                     fontsize=20, fontweight='bold')
        
        # 1. Scree plot - Varianza explicada por componente
        component_info = self.analyze_component_importance()
        
        axes[0, 0].bar(range(1, len(component_info) + 1), 
                       component_info['Varianza_Explicada'], 
                       color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Varianza Explicada por Componente', fontweight='bold')
        axes[0, 0].set_xlabel('Componente Principal')
        axes[0, 0].set_ylabel('Varianza Explicada')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Varianza acumulada
        axes[0, 1].plot(range(1, len(component_info) + 1), 
                        component_info['Varianza_Acumulada'], 
                        marker='o', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_title('Varianza Acumulada', fontweight='bold')
        axes[0, 1].set_xlabel('Número de Componentes')
        axes[0, 1].set_ylabel('Varianza Acumulada')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% de varianza')
        axes[0, 1].axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% de varianza')
        axes[0, 1].legend()
        
        # 3. Proyección 2D con colores por documento fuente
        axes[1, 0].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                           alpha=0.6, s=30, c='purple')
        axes[1, 0].set_title('Proyección 2D: Componente 1 vs Componente 2', fontweight='bold')
        axes[1, 0].set_xlabel(f'Componente 1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[1, 0].set_ylabel(f'Componente 2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Proyección 3D de los primeros tres componentes
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        scatter = ax_3d.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], embeddings_pca[:, 2], 
                               alpha=0.6, s=30, c='red')
        ax_3d.set_title('Proyección 3D: Componentes 1, 2 y 3', fontweight='bold')
        ax_3d.set_xlabel(f'Comp 1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax_3d.set_ylabel(f'Comp 2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax_3d.set_zlabel(f'Comp 3 ({self.pca.explained_variance_ratio_[2]*100:.1f}%)')
        
        plt.tight_layout()
        plt.savefig('pca_pnts_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizaciones guardadas en 'pca_pnts_analysis_results.png'")
    
    def analyze_feature_importance(self):
        """Analiza la importancia de las características originales en cada componente"""
        if self.pca is None:
            print("❌ Error: PCA no realizado")
            return None
        
        print("🔍 Analizando importancia de características...")
        
        # Obtener los loadings (coeficientes) de cada componente
        loadings = self.pca.components_
        
        # Crear DataFrame con los loadings
        feature_importance = pd.DataFrame(
            loadings.T,
            columns=[f'PC{i+1}' for i in range(loadings.shape[0])],
            index=[f'Dim_{i+1}' for i in range(loadings.shape[1])]
        )
        
        # Visualizar heatmap de importancia de características
        plt.figure(figsize=(15, 10))
        sns.heatmap(feature_importance, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0, 
                   fmt='.3f',
                   cbar_kws={'label': 'Coeficiente de Carga'})
        plt.title('Importancia de Características por Componente Principal - PNTs', fontweight='bold', fontsize=16)
        plt.xlabel('Componente Principal', fontweight='bold')
        plt.ylabel('Dimensión Original', fontweight='bold')
        plt.tight_layout()
        plt.savefig('pnts_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Análisis de importancia de características completado")
        return feature_importance
    
    def perform_tsne_analysis(self, n_components=2, perplexity=30):
        """Realiza análisis t-SNE para visualización de clusters"""
        if self.embeddings is None:
            print("❌ Error: No hay embeddings para analizar")
            return None
        
        print(f"🔄 Realizando t-SNE con {n_components} componentes y perplexity {perplexity}...")
        
        # Aplicar t-SNE
        tsne = TSNE(n_components=n_components, 
                    perplexity=perplexity, 
                    random_state=42, 
                    n_jobs=-1)
        
        embeddings_tsne = tsne.fit_transform(self.embeddings)
        
        # Visualizar resultados t-SNE
        plt.figure(figsize=(12, 10))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                   alpha=0.6, s=30, c='green')
        plt.title('Visualización t-SNE de Embeddings PNTs', fontweight='bold', fontsize=16)
        plt.xlabel('t-SNE 1', fontweight='bold')
        plt.ylabel('t-SNE 2', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pnts_tsne_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Análisis t-SNE completado")
        return embeddings_tsne
    
    def generate_comprehensive_report(self, sentence_sources):
        """Genera un reporte completo del análisis de PNTs"""
        if self.pca is None:
            print("❌ Error: PCA no realizado")
            return
        
        print("📊 Generando reporte completo...")
        
        # Información del modelo y documentos
        model_info = {
            'Modelo': self.model_path,
            'Dimensiones_Originales': self.embeddings.shape[1] if self.embeddings is not None else 'N/A',
            'Número_Documentos_PNTs': len(self.documents) if self.documents is not None else 'N/A',
            'Número_Oraciones': len(sentence_sources) if sentence_sources is not None else 'N/A',
            'Componentes_PCA': self.pca.n_components_ if self.pca else 'N/A'
        }
        
        # Estadísticas de componentes
        component_info = self.analyze_component_importance()
        
        # Guardar reporte en CSV
        component_info.to_csv('pnts_pca_component_analysis.csv', index=False)
        
        # Crear reporte de texto
        with open('pnts_pca_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE ANÁLISIS PCA - DOCUMENTOS PNTS MÉDICOS\n")
            f.write("MODELO: all-MiniLM-L6-v2\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("INFORMACIÓN DEL MODELO Y DOCUMENTOS:\n")
            f.write("-" * 40 + "\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nDOCUMENTOS PNTs ANALIZADOS:\n")
            f.write("-" * 35 + "\n")
            for i, doc_name in enumerate(self.document_names):
                f.write(f"{i+1:2d}. {doc_name}\n")
            
            f.write("\nANÁLISIS DE COMPONENTES PRINCIPALES:\n")
            f.write("-" * 40 + "\n")
            f.write(component_info.to_string())
            
            f.write(f"\n\nRESUMEN:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• Varianza total explicada: {np.sum(self.pca.explained_variance_ratio_)*100:.2f}%\n")
            f.write(f"• Componentes para 80% varianza: {np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= 0.8) + 1}\n")
            f.write(f"• Componentes para 90% varianza: {np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= 0.9) + 1}\n")
            f.write(f"• Componentes para 95% varianza: {np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= 0.95) + 1}\n")
            
            f.write(f"\nRECOMENDACIONES PARA SISTEMAS MÉDICOS:\n")
            f.write("-" * 45 + "\n")
            f.write("• Para búsqueda semántica en PNTs: usar 15-20 componentes principales\n")
            f.write("• Para clustering de procedimientos: usar 7-10 componentes principales\n")
            f.write("• Para sistemas de recomendación médica: usar 10-15 componentes principales\n")
            f.write("• Para análisis de similitud de procedimientos: usar 5-7 componentes principales\n")
        
        print("✅ Reporte completo generado:")
        print("   - pnts_pca_component_analysis.csv")
        print("   - pnts_pca_analysis_report.txt")
        
        return component_info

def main():
    """Función principal para ejecutar el análisis PCA de PNTs"""
    print("🚀 INICIANDO ANÁLISIS PCA DE DOCUMENTOS PNTS MÉDICOS")
    print("=" * 70)
    
    # Crear analizador
    analyzer = PNTsEmbeddingAnalyzer()
    
    # Cargar modelo
    if not analyzer.load_model():
        return
    
    # Cargar documentos PNTs
    if not analyzer.load_pnts_documents():
        return
    
    # Segmentar documentos en oraciones
    sentences, sentence_sources = analyzer.segment_documents_into_sentences()
    
    # Generar embeddings
    if not analyzer.generate_embeddings_from_sentences(sentences):
        return
    
    # Realizar análisis PCA
    embeddings_pca = analyzer.perform_pca_analysis(n_components=20)
    
    # Visualizar resultados
    analyzer.visualize_pca_results(embeddings_pca, sentence_sources)
    
    # Analizar importancia de características
    feature_importance = analyzer.analyze_feature_importance()
    
    # Realizar análisis t-SNE
    embeddings_tsne = analyzer.perform_tsne_analysis()
    
    # Generar reporte completo
    analyzer.generate_comprehensive_report(sentence_sources)
    
    print("\n🎉 ANÁLISIS PCA DE PNTS COMPLETADO EXITOSAMENTE!")
    print("📁 Archivos generados:")
    print("   - pca_pnts_analysis_results.png")
    print("   - pnts_feature_importance_heatmap.png")
    print("   - pnts_tsne_visualization.png")
    print("   - pnts_pca_component_analysis.csv")
    print("   - pnts_pca_analysis_report.txt")
    print("\n💡 Este análisis te permitirá:")
    print("   • Optimizar sistemas de búsqueda en documentación médica")
    print("   • Clasificar procedimientos de enfermería por similitud")
    print("   • Reducir dimensiones para aplicaciones médicas en tiempo real")
    print("   • Identificar patrones semánticos en procedimientos médicos")

if __name__ == "__main__":
    main()
