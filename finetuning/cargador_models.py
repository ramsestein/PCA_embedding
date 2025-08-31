#!/usr/bin/env python3
"""
Script para descargar y preparar los modelos de embedding
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import Dict
import json

class UnifiedEmbeddingAdapter:
    """Adaptador unificado para cualquier modelo de Hugging Face"""
    
    def __init__(self, model_path: str, model_name: str = None, pooling_strategy: str = 'mean'):
        """
        Args:
            model_path: Path al modelo local o nombre en HuggingFace
            model_name: Nombre descriptivo del modelo
            pooling_strategy: 'mean', 'cls', o 'max' pooling
        """
        self.model_path = model_path
        self.model_name = model_name or Path(model_path).name
        self.pooling_strategy = pooling_strategy
        
        print(f"🔄 Cargando modelo: {self.model_name}")
        
        # Cargar tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
        # Configurar dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Detectar dimensión
        self._detect_dimension()
        
        print(f"✅ Modelo cargado: {self.model_name} (dim={self.dimension})")
        
    def _detect_dimension(self):
        """Detecta automáticamente la dimensión del embedding"""
        with torch.no_grad():
            # Generar un embedding de prueba
            test_input = self.tokenizer("test", return_tensors='pt').to(self.device)
            outputs = self.model(**test_input)
            
            if self.pooling_strategy == 'cls':
                test_embedding = outputs.last_hidden_state[:, 0, :]
            else:  # mean pooling por defecto
                test_embedding = outputs.last_hidden_state.mean(dim=1)
                
            self.dimension = test_embedding.shape[-1]
    
    def embed(self, text: str) -> np.ndarray:
        """Genera embedding para un texto"""
        # Tokenizar
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Generar embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Aplicar estrategia de pooling
            if self.pooling_strategy == 'cls':
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif self.pooling_strategy == 'max':
                embeddings = outputs.last_hidden_state.max(dim=1)[0]
            else:  # mean pooling
                # Considerar attention mask para mean pooling correcto
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
                sum_mask = mask_expanded.sum(1)
                embeddings = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        
        return embeddings.cpu().numpy().squeeze()
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def get_name(self) -> str:
        return self.model_name


def download_all_minilm(target_dir: str = "./models/all-MiniLM-L6-v2"):
    """Descarga el modelo all-MiniLM-L6-v2 de Hugging Face"""
    from huggingface_hub import snapshot_download
    
    print("📥 Descargando all-MiniLM-L6-v2 desde Hugging Face...")
    
    # Crear directorio si no existe
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Descargar modelo
        snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Modelo descargado en: {target_dir}")
        
        # Verificar archivos descargados
        required_files = ['config.json', 'tokenizer_config.json', 'pytorch_model.bin']
        files_ok = all((Path(target_dir) / f).exists() for f in required_files)
        
        if files_ok:
            print("✅ Todos los archivos necesarios están presentes")
        else:
            print("⚠️  Algunos archivos pueden faltar, verificar manualmente")
            
        return True
        
    except Exception as e:
        print(f"❌ Error descargando modelo: {e}")
        return False


def prepare_models_for_comparison():
    """Prepara ambos modelos para la comparación"""
    
    print("="*60)
    print("PREPARACIÓN DE MODELOS PARA COMPARACIÓN")
    print("="*60)
    
    # 1. Verificar/descargar all-MiniLM
    all_minilm_path = "./models/all-MiniLM-L6-v2"
    
    if not Path(all_minilm_path).exists():
        print("\n1. all-MiniLM-L6-v2 no encontrado localmente")
        download_all_minilm(all_minilm_path)
    else:
        print("\n1. ✅ all-MiniLM-L6-v2 ya existe localmente")
    
    # 2. Verificar RoBERTa
    roberta_path = "./models/roberta-ca-embeddings"  # Ajusta según tu path
    
    print(f"\n2. Verificando RoBERTa en: {roberta_path}")
    if Path(roberta_path).exists():
        print("✅ RoBERTa-CA encontrado")
    else:
        print(f"❌ No se encuentra RoBERTa en: {roberta_path}")
        print("   Por favor, asegúrate de que tu modelo esté en esa ubicación")
        return False
    
    # 3. Test de carga
    print("\n3. Probando carga de modelos...")
    
    try:
        # Probar all-MiniLM
        print("\n   Cargando all-MiniLM...")
        minilm_adapter = UnifiedEmbeddingAdapter(
            all_minilm_path, 
            model_name="all-MiniLM-L6-v2",
            pooling_strategy='mean'
        )
        test_embedding = minilm_adapter.embed("Prueba de texto")
        print(f"   ✅ Embedding generado: shape={test_embedding.shape}")
        
        # Probar RoBERTa
        print("\n   Cargando RoBERTa-CA...")
        roberta_adapter = UnifiedEmbeddingAdapter(
            roberta_path,
            model_name="RoBERTa-CA-v2",
            pooling_strategy='mean'  # Ajusta según tu modelo
        )
        test_embedding = roberta_adapter.embed("Prova de text en català")
        print(f"   ✅ Embedding generado: shape={test_embedding.shape}")
        
        print("\n✅ Ambos modelos funcionan correctamente")
        return True
        
    except Exception as e:
        print(f"\n❌ Error cargando modelos: {e}")
        import traceback
        traceback.print_exc()
        return False


# Actualización del comparador para usar el adaptador unificado
class EmbeddingSystemComparatorUnified:
    """Versión simplificada del comparador usando adaptadores unificados"""
    
    def __init__(self, model_paths: Dict[str, dict], documents_path: str, 
                 test_queries_path: str, db_connection_string: str = None):
        """
        Args:
            model_paths: Dict con configuración de modelos, ej:
                {
                    'minilm': {
                        'path': './models/all-MiniLM-L6-v2',
                        'name': 'all-MiniLM-L6-v2',
                        'pooling': 'mean'
                    },
                    'roberta': {
                        'path': './models/roberta-ca-embeddings',
                        'name': 'RoBERTa-CA-v2',
                        'pooling': 'mean'
                    }
                }
        """
        from config import PGVECTOR_CONNECTION_STRING
        
        self.documents_path = Path(documents_path)
        self.test_queries_path = test_queries_path
        self.db_connection_string = db_connection_string or PGVECTOR_CONNECTION_STRING
        
        # Cargar adaptadores
        self.adapters = {}
        for key, config in model_paths.items():
            self.adapters[key] = UnifiedEmbeddingAdapter(
                model_path=config['path'],
                model_name=config.get('name'),
                pooling_strategy=config.get('pooling', 'mean')
            )
        
        self.results = {}
    
    # El resto de métodos serían idénticos a EmbeddingSystemComparator
    # pero usando self.adapters[key].embed() y self.adapters[key].get_dimension()


# Script de configuración rápida
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preparar modelos para comparación')
    parser.add_argument('--download-minilm', action='store_true',
                       help='Forzar descarga de all-MiniLM')
    parser.add_argument('--roberta-path', default='./models/roberta-ca-embeddings',
                       help='Path al modelo RoBERTa')
    
    args = parser.parse_args()
    
    if args.download_minilm:
        download_all_minilm()
    
    # Preparar y verificar modelos
    if prepare_models_for_comparison():
        print("\n" + "="*60)
        print("✅ MODELOS LISTOS PARA COMPARACIÓN")
        print("="*60)
        print("\nPuedes ejecutar la comparación con:")
        print("python run_embedding_comparison_unified.py")
    else:
        print("\n❌ Hubo problemas preparando los modelos")