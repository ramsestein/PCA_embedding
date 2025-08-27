#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de Componentes Principales (PCA) del Modelo de Embeddings all-MiniLM-L6-v2
Autor: Análisis de Embeddings
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
import warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib para español
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class EmbeddingAnalyzer:
    """Clase para analizar embeddings usando PCA y técnicas de reducción de dimensionalidad"""
    
    def __init__(self, model_path="all-mini-base"):
        """
        Inicializa el analizador de embeddings
        
        Args:
            model_path (str): Ruta al modelo de embeddings
        """
        self.model_path = model_path
        self.model = None
        self.embeddings = None
        self.sentences = None
        self.pca = None
        self.scaler = None
        
    def load_model(self):
        """Carga el modelo de embeddings"""
        print("🔄 Cargando modelo de embeddings...")
        try:
            self.model = SentenceTransformer(self.model_path)
            print(f"✅ Modelo cargado exitosamente desde: {self.model_path}")
            print(f"📊 Dimensiones del embedding: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            return False
        return True
    
    def generate_sample_sentences(self, num_sentences=1000):
        """Genera oraciones de muestra para el análisis"""
        print(f"📝 Generando {num_sentences} oraciones de muestra...")
        
        # Categorías de oraciones para análisis diverso
        categories = {
            'tecnología': [
                "Python es un lenguaje de programación versátil",
                "Machine learning requiere datos de calidad",
                "La inteligencia artificial transforma industrias",
                "El deep learning usa redes neuronales",
                "Los algoritmos optimizan procesos complejos"
            ],
            'ciencia': [
                "La física cuántica describe partículas subatómicas",
                "La evolución es un proceso biológico fundamental",
                "La química estudia la composición de la materia",
                "Las matemáticas son el lenguaje del universo",
                "La astronomía explora el cosmos"
            ],
            'cultura': [
                "El arte expresa emociones humanas",
                "La literatura refleja la sociedad",
                "La música conecta culturas diferentes",
                "La historia nos enseña del pasado",
                "La filosofía cuestiona la existencia"
            ],
            'negocios': [
                "El marketing digital es esencial hoy",
                "La innovación impulsa el crecimiento",
                "Los datos guían decisiones estratégicas",
                "La sostenibilidad es clave empresarial",
                "El liderazgo inspira equipos"
            ],
            'salud': [
                "El ejercicio mejora la salud cardiovascular",
                "La nutrición equilibrada es fundamental",
                "El sueño reparador fortalece el sistema inmune",
                "La meditación reduce el estrés",
                "La prevención es mejor que la curación"
            ]
        }
        
        sentences = []
        for category, examples in categories.items():
            # Generar variaciones de las oraciones base
            for base_sentence in examples:
                for i in range(num_sentences // len(categories) // len(examples)):
                    # Crear variaciones semánticas
                    variations = [
                        base_sentence,
                        f"Variación {i+1}: {base_sentence}",
                        f"Otro aspecto: {base_sentence}",
                        f"Perspectiva diferente: {base_sentence}",
                        f"Ejemplo concreto: {base_sentence}"
                    ]
                    sentences.extend(variations[:num_sentences // len(categories) // len(examples)])
        
        # Asegurar que tenemos exactamente el número solicitado
        self.sentences = sentences[:num_sentences]
        print(f"✅ {len(self.sentences)} oraciones generadas")
        return self.sentences
    
    def generate_embeddings(self):
        """Genera embeddings para las oraciones de muestra"""
        if self.model is None:
            print("❌ Error: Modelo no cargado")
            return False
        
        if self.sentences is None:
            print("❌ Error: No hay oraciones para procesar")
            return False
        
        print("🔄 Generando embeddings...")
        try:
            self.embeddings = self.model.encode(self.sentences, show_progress_bar=True)
            print(f"✅ Embeddings generados: {self.embeddings.shape}")
            return True
        except Exception as e:
            print(f"❌ Error al generar embeddings: {e}")
            return False
    
    def perform_pca_analysis(self, n_components=10):
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
    
    def visualize_pca_results(self, embeddings_pca):
        """Visualiza los resultados del PCA"""
        if self.pca is None or embeddings_pca is None:
            print("❌ Error: PCA no realizado")
            return
        
        print("🎨 Generando visualizaciones...")
        
        # Configurar el estilo de las gráficas
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Análisis de Componentes Principales - Modelo all-MiniLM-L6-v2', 
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
        
        # 3. Proyección 2D de los primeros dos componentes
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
        plt.savefig('pca_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizaciones guardadas en 'pca_analysis_results.png'")
    
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
        plt.title('Importancia de Características por Componente Principal', fontweight='bold', fontsize=16)
        plt.xlabel('Componente Principal', fontweight='bold')
        plt.ylabel('Dimensión Original', fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
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
        plt.title('Visualización t-SNE de Embeddings', fontweight='bold', fontsize=16)
        plt.xlabel('t-SNE 1', fontweight='bold')
        plt.ylabel('t-SNE 2', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Análisis t-SNE completado")
        return embeddings_tsne
    
    def generate_comprehensive_report(self):
        """Genera un reporte completo del análisis"""
        if self.pca is None:
            print("❌ Error: PCA no realizado")
            return
        
        print("📊 Generando reporte completo...")
        
        # Información del modelo
        model_info = {
            'Modelo': self.model_path,
            'Dimensiones_Originales': self.embeddings.shape[1] if self.embeddings is not None else 'N/A',
            'Número_Oraciones': len(self.sentences) if self.sentences is not None else 'N/A',
            'Componentes_PCA': self.pca.n_components_ if self.pca else 'N/A'
        }
        
        # Estadísticas de componentes
        component_info = self.analyze_component_importance()
        
        # Guardar reporte en CSV
        component_info.to_csv('pca_component_analysis.csv', index=False)
        
        # Crear reporte de texto
        with open('pca_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE ANÁLISIS PCA - MODELO all-MiniLM-L6-v2\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("INFORMACIÓN DEL MODELO:\n")
            f.write("-" * 30 + "\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nANÁLISIS DE COMPONENTES PRINCIPALES:\n")
            f.write("-" * 40 + "\n")
            f.write(component_info.to_string())
            
            f.write(f"\n\nRESUMEN:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• Varianza total explicada: {np.sum(self.pca.explained_variance_ratio_)*100:.2f}%\n")
            f.write(f"• Componentes para 80% varianza: {np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= 0.8) + 1}\n")
            f.write(f"• Componentes para 90% varianza: {np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= 0.9) + 1}\n")
            f.write(f"• Componentes para 95% varianza: {np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= 0.95) + 1}\n")
        
        print("✅ Reporte completo generado:")
        print("   - pca_component_analysis.csv")
        print("   - pca_analysis_report.txt")
        
        return component_info

def main():
    """Función principal para ejecutar el análisis completo"""
    print("🚀 INICIANDO ANÁLISIS PCA DE EMBEDDINGS")
    print("=" * 60)
    
    # Crear analizador
    analyzer = EmbeddingAnalyzer()
    
    # Cargar modelo
    if not analyzer.load_model():
        return
    
    # Generar oraciones de muestra
    analyzer.generate_sample_sentences(num_sentences=500)
    
    # Generar embeddings
    if not analyzer.generate_embeddings():
        return
    
    # Realizar análisis PCA
    embeddings_pca = analyzer.perform_pca_analysis(n_components=20)
    
    # Visualizar resultados
    analyzer.visualize_pca_results(embeddings_pca)
    
    # Analizar importancia de características
    feature_importance = analyzer.analyze_feature_importance()
    
    # Realizar análisis t-SNE
    embeddings_tsne = analyzer.perform_tsne_analysis()
    
    # Generar reporte completo
    analyzer.generate_comprehensive_report()
    
    print("\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE!")
    print("📁 Archivos generados:")
    print("   - pca_analysis_results.png")
    print("   - feature_importance_heatmap.png")
    print("   - tsne_visualization.png")
    print("   - pca_component_analysis.csv")
    print("   - pca_analysis_report.txt")

if __name__ == "__main__":
    main()
