#!/usr/bin/env python3
"""
Entrenamiento en serie de múltiples modelos de embeddings para corpus clínico cerrado
Optimizado para Windows 11 + RTX 4060 (8GB VRAM)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import psutil
import argparse
import traceback
import gc

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_series.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configuración para Windows y GPU
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# CONFIGURACIÓN DE MODELOS A ENTRENAR
MODELS_TO_TRAIN = [
    # Tier 1: Máximo potencial
    {
        "model_name": "michiyasunaga/BioLinkBERT-base",
        "model_type": "biolinkbert",
        "dimensions": 768,
        "batch_size": 16,
        "learning_rate": 1e-5,
        "num_epochs": 15,
        "warmup_ratio": 0.1,
        "temperature": 0.07,
        "margin": 0.3,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "model_type": "bluebert",
        "dimensions": 768,
        "batch_size": 24,
        "learning_rate": 2e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.15,
        "temperature": 0.05,
        "margin": 0.3,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "model_type": "sapbert-pubmed",
        "dimensions": 768,
        "batch_size": 24,
        "learning_rate": 1e-5,  # Más conservador, ya está fine-tuneado
        "num_epochs": 15,
        "warmup_ratio": 0.2,
        "temperature": 0.07,
        "margin": 0.5,
        "max_length": 256,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "ncbi/MedCPT-Query-Encoder",
        "model_type": "medcpt",
        "dimensions": 768,
        "batch_size": 24,
        "learning_rate": 5e-6,  # Muy conservador, ya está fine-tuneado
        "num_epochs": 10,
        "warmup_ratio": 0.1,
        "temperature": 0.05,
        "margin": 0.5,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "dmis-lab/biobert-base-cased-v1.2",
        "model_type": "biobert-v1.2",
        "dimensions": 768,
        "batch_size": 28,
        "learning_rate": 2e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.15,
        "temperature": 0.05,
        "margin": 0.3,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    
    {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "model_type": "bioclinicalbert",
        "dimensions": 768,
        "batch_size": 28,
        "learning_rate": 2e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.15,
        "temperature": 0.05,
        "margin": 0.3,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "dmis-lab/biobert-v1.1",
        "model_type": "biobert",
        "dimensions": 768,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.15,
        "temperature": 0.05,
        "margin": 0.3,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "allenai/scibert_scivocab_uncased",
        "model_type": "scibert",
        "dimensions": 768,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.15,
        "temperature": 0.05,
        "margin": 0.3,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "model_type": "pubmedbert-ms",
        "dimensions": 768,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.15,
        "temperature": 0.05,
        "margin": 0.3,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        "model_type": "sapbert-umls",
        "dimensions": 768,
        "batch_size": 24,
        "learning_rate": 1e-5,
        "num_epochs": 15,
        "warmup_ratio": 0.2,
        "temperature": 0.07,
        "margin": 0.5,
        "max_length": 256,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT",
        "model_type": "pubmedbert-marco",
        "dimensions": 768,
        "batch_size": 24,
        "learning_rate": 5e-6,
        "num_epochs": 10,
        "warmup_ratio": 0.1,
        "temperature": 0.05,
        "margin": 0.5,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "PlanTL-GOB-ES/roberta-base-biomedical-es",
        "model_type": "biomedical-roberta",
        "dimensions": 768,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.1,
        "temperature": 0.05,
        "margin": 0.2,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "model_type": "bioclinicalbert",
        "dimensions": 768,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.1,
        "temperature": 0.05,
        "margin": 0.2,
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    
    {
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "model_type": "sapbert-pubmed",
        "dimensions": 768,
        "batch_size": 32,
        "learning_rate": 2e-5,  # Más conservador porque ya está fine-tuneado para embeddings
        "num_epochs": 15,  # Menos épocas, ya está especializado
        "warmup_ratio": 0.1,
        "temperature": 0.05,
        "margin": 0.3,  # Margin más alto para mejor separación
        "max_length": 384,
        "gradient_checkpointing": True,
    },
    {
        "model_name": "microsoft/deberta-v3-base",
        "model_type": "deberta-v3-conservative",
        "dimensions": 768,
        "batch_size": 8,  # Muy pequeño
        "learning_rate": 5e-6,  # Ultra bajo
        "num_epochs": 30,  # Más épocas pero con LR bajo
        "warmup_ratio": 0.3,  # 30% warmup
        "temperature": 0.1,  # Más alta para evitar colapso
        "margin": 0.1,  # Margin pequeño inicialmente
        "max_length": 384,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,  # Para simular batch_size=32
        "pooling_strategy": "cls",  # DeBERTa funciona mejor con CLS
    },
    {
        "model_name": "microsoft/deberta-v3-base",
        "model_type": "deberta-v3-tuned",
        "dimensions": 768,
        "batch_size": 12,
        "learning_rate": 1e-5,
        "num_epochs": 25,
        "warmup_ratio": 0.25,
        "temperature": 0.07,
        "margin": 0.15,
        "max_length": 256,  # Más corto para estabilidad
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 2,
        "pooling_strategy": "cls",  # Importante
        "weight_decay": 0.05,  # Más regularización
        "uniformity_weight": 0.2,  # Más peso anti-colapso
    },
    {
        "model_name": "microsoft/deberta-v3-small",
        "model_type": "deberta-v3-small",
        "dimensions": 768,
        "batch_size": 32,  # Puede manejar batch más grande
        "learning_rate": 2e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.2,
        "temperature": 0.05,
        "margin": 0.2,
        "max_length": 384,
        "gradient_checkpointing": False,  # No necesario para small
        "pooling_strategy": "mean",  # Puede funcionar con mean
    },
    
    {
        "model_name": "projecte-aina/roberta-base-ca-v2",
        "model_type": "roberta-catalan",
        "dimensions": 768,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.1,
        "temperature": 0.05,
        "margin": 0.2,
        "max_length": 384,
        "gradient_checkpointing": True,
    }
]

@dataclass
class TrainingConfig:
    """Configuración de entrenamiento adaptable por modelo"""
    model_name: str
    model_type: str
    dimensions: int
    max_length: int = 384
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    temperature: float = 0.05
    margin: float = 0.3
    loss_type: str = "triplet"
    pooling_strategy: str = "mean"
    output_dir: str = "/content/drive/MyDrive/models_embedding"
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    fp16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42
    
    # Parámetros anti-colapso para corpus cerrado
    uniformity_weight: float = 0.1
    diversity_threshold: float = 0.85  # Si avg_similarity > threshold, aplicar penalización
    hard_negatives_ratio: float = 0.5
    
    def __post_init__(self):
        # Calcular warmup_steps si se proporciona warmup_ratio
        if self.warmup_ratio and not self.warmup_steps:
            # Esto se calculará más tarde cuando sepamos el tamaño del dataset
            pass

class AntiCollapseContrastiveLoss(nn.Module):
    """Loss contrastiva con prevención de colapso para corpus cerrado"""
    
    def __init__(self, temperature: float = 0.05, margin: float = 0.2, 
                 uniformity_weight: float = 0.1, diversity_threshold: float = 0.85):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.uniformity_weight = uniformity_weight
        self.diversity_threshold = diversity_threshold

    def infonce_loss(self, anchor, positive, negative):
        """InfoNCE loss (SimCLR style)"""
        batch_size = anchor.size(0)
        
        # Similitudes
        pos_sim = F.cosine_similarity(anchor, positive) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negative) / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
        
    def triplet_loss(self, anchor, positive, negative):
        """Triplet loss with margin"""
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    
    def uniformity_loss(self, embeddings):
        """Penaliza cuando los embeddings son muy similares entre sí"""
        # Calcular todas las similitudes por pares
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Excluir la diagonal (similitud consigo mismo)
        mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_values = sim_matrix[mask]
        
        # Penalizar si la similitud promedio es muy alta
        avg_sim = sim_values.mean()
        if avg_sim > self.diversity_threshold:
            penalty = (avg_sim - self.diversity_threshold) ** 2
        else:
            penalty = torch.tensor(0.0, device=embeddings.device)
        
        return penalty
    
    def forward(self, anchor, positive, negative):
        """Loss combinada con anti-colapso"""
        # Loss principal
        main_loss = self.triplet_loss(anchor, positive, negative)
        
        # Loss de uniformidad (anti-colapso)
        all_embeddings = torch.cat([anchor, positive, negative], dim=0)
        uniform_loss = self.uniformity_loss(all_embeddings)
        
        # Combinar losses
        total_loss = main_loss + self.uniformity_weight * uniform_loss
        
        return total_loss, {
            'main_loss': main_loss.item(),
            'uniform_loss': uniform_loss.item(),
            'avg_similarity': F.cosine_similarity(anchor, positive).mean().item()
        }

class ContrastiveDataset(Dataset):
    """Dataset para pares contrastivos"""
    
    def __init__(self, filepath: str, tokenizer, max_length: int = 384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = self._load_pairs(filepath)
        logger.info(f"Cargados {len(self.pairs)} pares de entrenamiento")
        
    def _load_pairs(self, filepath: str) -> List[Dict]:
        """Carga pares desde archivo JSONL"""
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Tokenizar los tres textos
        anchor = self._tokenize(pair['anchor'])
        positive = self._tokenize(pair['positive'])
        negative = self._tokenize(pair['negative'])
        
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'difficulty': torch.tensor(pair['difficulty'], dtype=torch.float32)
        }
    
    def _tokenize(self, text: str):
        """Tokeniza un texto"""
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Remover la dimensión extra del batch
        return {k: v.squeeze(0) for k, v in encoded.items()}

class RoBERTaEmbeddings(nn.Module):
    """Modelo RoBERTa para generar embeddings"""
    
    def __init__(self, model_name: str, pooling_strategy: str = "mean"):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.hidden_size = self.roberta.config.hidden_size
        
        # Capa de proyección opcional (reduce dimensionalidad)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.Tanh()
        )
        
    def forward(self, input_ids, attention_mask):
        # Asegurar que los inputs están en el mismo dispositivo que el modelo
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Salida de RoBERTa
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Aplicar pooling
        if self.pooling_strategy == "mean":
            # Mean pooling con attention mask
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling_strategy == "cls":
            # CLS token
            embeddings = outputs.last_hidden_state[:, 0]
        else:  # max
            # Max pooling
            embeddings = self._max_pooling(outputs.last_hidden_state, attention_mask)
        
        # Proyección
        embeddings = self.projection(embeddings)
        
        # Normalizar para cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling considerando attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _max_pooling(self, token_embeddings, attention_mask):
        """Max pooling considerando attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(token_embeddings, 1)[0]
        return max_embeddings

class SeriesTrainer:
    """Entrenador que ejecuta múltiples modelos en serie"""
    
    def __init__(self, base_output_dir: str, train_file: str):
        self.base_output_dir = Path(base_output_dir)
        self.train_file = train_file
        self.results = []
        
    def train_all_models(self, models_config: List[Dict]):
        """Entrena todos los modelos en la lista"""
        logger.info("="*80)
        logger.info(f"🚀 INICIANDO ENTRENAMIENTO EN SERIE DE {len(models_config)} MODELOS")
        logger.info("="*80)
        
        for i, model_config in enumerate(models_config, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 MODELO {i}/{len(models_config)}: {model_config['model_name']}")
            logger.info(f"{'='*80}")
            
            try:
                # Limpiar memoria GPU antes de cada modelo
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Crear configuración específica para este modelo
                config = TrainingConfig(
                    model_name=model_config['model_name'],
                    model_type=model_config['model_type'],
                    dimensions=model_config['dimensions'],
                    batch_size=model_config['batch_size'],
                    learning_rate=model_config['learning_rate'],
                    num_epochs=model_config['num_epochs'],
                    warmup_ratio=model_config.get('warmup_ratio', 0.1),
                    temperature=model_config.get('temperature', 0.05),
                    margin=model_config.get('margin', 0.3),
                    max_length=model_config.get('max_length', 384),
                    gradient_checkpointing=model_config.get('gradient_checkpointing', True),
                    output_dir=str(self.base_output_dir / model_config['model_type'])
                )
                
                # Entrenar modelo
                trainer = EmbeddingTrainer(config)
                best_loss = trainer.train(self.train_file)
                
                # Guardar resultado
                self.results.append({
                    'model': model_config['model_name'],
                    'type': model_config['model_type'],
                    'status': 'success',
                    'best_loss': best_loss,
                    'output_dir': config.output_dir
                })
                
                logger.info(f"✅ Modelo {model_config['model_type']} completado con éxito!")
                
            except Exception as e:
                logger.error(f"❌ Error entrenando {model_config['model_name']}: {str(e)}")
                logger.error(traceback.format_exc())
                
                self.results.append({
                    'model': model_config['model_name'],
                    'type': model_config['model_type'],
                    'status': 'failed',
                    'error': str(e)
                })
                
                # Continuar con el siguiente modelo
                continue
        
        # Guardar resumen de resultados
        self._save_results_summary()
        
    def _save_results_summary(self):
        """Guarda un resumen de todos los entrenamientos"""
        summary_path = self.base_output_dir / "training_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_models': len(self.results),
                'successful': sum(1 for r in self.results if r['status'] == 'success'),
                'failed': sum(1 for r in self.results if r['status'] == 'failed'),
                'results': self.results
            }, f, indent=2)
        
        # Imprimir resumen
        logger.info("\n" + "="*80)
        logger.info("📊 RESUMEN DE ENTRENAMIENTO")
        logger.info("="*80)
        
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        
        logger.info(f"✅ Exitosos: {len(successful)}/{len(self.results)}")
        for r in successful:
            logger.info(f"   - {r['type']}: loss={r.get('best_loss', 'N/A'):.4f}")
        
        if failed:
            logger.info(f"\n❌ Fallidos: {len(failed)}")
            for r in failed:
                logger.info(f"   - {r['type']}: {r['error']}")
        
        logger.info(f"\n📁 Resumen guardado en: {summary_path}")

class EmbeddingTrainer:
    """Clase principal para entrenar embeddings"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Verificar GPU con diagnóstico detallado
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"   CUDA version: {torch.version.cuda}")
            logger.info(f"   PyTorch version: {torch.__version__}")
            
            # Limpiar caché GPU
            torch.cuda.empty_cache()
        else:
            logger.warning("⚠️ GPU no disponible, entrenando en CPU (será lento)")
            logger.info(f"   PyTorch version: {torch.__version__}")
            
            # Diagnóstico rápido
            if '+cpu' in torch.__version__:
                logger.error("❌ PyTorch CPU instalado. Necesitas versión CUDA.")
                logger.info("Ejecuta: fix_cuda_install.bat")
                sys.exit(1)
        
        # Inicializar modelo y tokenizer
        logger.info(f"Cargando modelo: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = RoBERTaEmbeddings(config.model_name, config.pooling_strategy)
        self.model.to(self.device)
        
        # Gradient checkpointing para ahorrar memoria
        if config.gradient_checkpointing and hasattr(self.model.roberta, 'gradient_checkpointing_enable'):
            self.model.roberta.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing activado")
        
        # Loss function
        self.criterion = AntiCollapseContrastiveLoss(
            config.temperature, 
            config.margin,
            config.uniformity_weight,
            config.diversity_threshold
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.fp16 else None
        
        # Directorio de salida
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def train(self, train_file: str):
        """Entrena el modelo"""
        # Crear dataset y dataloader
        dataset = ContrastiveDataset(train_file, self.tokenizer, self.config.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2, 
            drop_last=True
        )

        # Calcular warmup_steps si se usa warmup_ratio
        if self.config.warmup_ratio and not self.config.warmup_steps:
            total_steps = len(dataloader) * self.config.num_epochs
            self.config.warmup_steps = int(total_steps * self.config.warmup_ratio)
                
        # Optimizador
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(dataloader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Variables de entrenamiento
        global_step = 0
        best_loss = float('inf')
        
        logger.info(f"🚀 Iniciando entrenamiento")
        logger.info(f"  - Épocas: {self.config.num_epochs}")
        logger.info(f"  - Batch size: {self.config.batch_size}")
        logger.info(f"  - Learning rate: {self.config.learning_rate}")
        logger.info(f"  - Total pasos: {total_steps}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Época {epoch+1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Mover batch a GPU correctamente
                if isinstance(batch['anchor'], dict):
                    # Mover cada tensor dentro del diccionario
                    anchor_inputs = {k: v.to(self.device) for k, v in batch['anchor'].items()}
                    positive_inputs = {k: v.to(self.device) for k, v in batch['positive'].items()}
                    negative_inputs = {k: v.to(self.device) for k, v in batch['negative'].items()}
                else:
                    # Si el batch ya viene procesado
                    anchor_inputs = batch['anchor']
                    positive_inputs = batch['positive']
                    negative_inputs = batch['negative']
                
                difficulty = batch['difficulty'].to(self.device) if isinstance(batch['difficulty'], torch.Tensor) else batch['difficulty']
                
                # Forward pass con mixed precision
                with autocast(enabled=self.config.fp16):
                    # Obtener embeddings
                    anchor_emb = self._get_embeddings(anchor_inputs)
                    positive_emb = self._get_embeddings(positive_inputs)
                    negative_emb = self._get_embeddings(negative_inputs)
                    
                    # Calcular loss
                    loss, loss_metrics = self.criterion(anchor_emb, positive_emb, negative_emb)
                    
                    # Gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Actualizar métricas
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                avg_loss = epoch_loss / (step + 1)
                
                # Actualizar progress bar
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    self._log_metrics(global_step, avg_loss, scheduler.get_last_lr()[0])
                
                # Guardar checkpoint
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step, avg_loss)
            
            # Fin de época
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"✓ Época {epoch+1} completada. Loss promedio: {avg_epoch_loss:.4f}")
            
            # Guardar modelo con loss en el nombre
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                # Formatear loss para el nombre del archivo (reemplazar punto por guión)
                loss_str = f"{avg_epoch_loss:.4f}".replace(".", "_")
                self._save_model(f"{loss_str}")
                logger.info(f"💾 Nuevo mejor modelo guardado: model-{loss_str} (loss: {best_loss:.4f})")
        
        # Guardar modelo final
        self._save_model("final")
        logger.info("✅ Entrenamiento completado!")
        return best_loss
    
    def _get_embeddings(self, batch_dict):
        """Obtiene embeddings de un batch"""
        if isinstance(batch_dict, dict):
            # Si es un diccionario, extraer input_ids y attention_mask
            input_ids = batch_dict['input_ids']
            attention_mask = batch_dict['attention_mask']
            
            # Asegurar que están en el dispositivo correcto y tienen la forma correcta
            if len(input_ids.shape) == 3:
                input_ids = input_ids.squeeze(1)
                attention_mask = attention_mask.squeeze(1)
        else:
            # Si no es diccionario, asumir que ya son los tensores correctos
            input_ids = batch_dict
            attention_mask = torch.ones_like(input_ids)
        
        # Asegurar que están en el dispositivo correcto
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        return self.model(input_ids, attention_mask)
    
    def _log_metrics(self, step: int, loss: float, lr: float):
        """Registra métricas"""
        memory_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        logger.info(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}, GPU_mem={memory_used:.1f}GB")
    
    def _save_checkpoint(self, step: int, loss: float):
        """Guarda checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Guardar modelo
        self.model.roberta.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Guardar projection layer
        torch.save(self.model.projection.state_dict(), checkpoint_dir / "projection.pt")
        
        # Guardar config
        config_dict = {
            "pooling_strategy": self.config.pooling_strategy,
            "hidden_size": self.model.hidden_size,
            "step": step,
            "loss": loss
        }
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"💾 Checkpoint guardado en {checkpoint_dir}")
    
    def _save_model(self, suffix: str):
        """Guarda el modelo completo"""
        save_dir = Path(self.config.output_dir) / f"model-{suffix}"
        save_dir.mkdir(exist_ok=True)
        
        # Guardar modelo y tokenizer
        self.model.roberta.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Guardar projection layer
        torch.save(self.model.projection.state_dict(), save_dir / "projection.pt")
        
        # Guardar configuración completa
        config_dict = {
            "model_name": self.config.model_name,
            "pooling_strategy": self.config.pooling_strategy,
            "hidden_size": self.model.hidden_size,
            "max_length": self.config.max_length,
            "training_config": self.config.__dict__
        }
        with open(save_dir / "embedding_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Script de inferencia
        self._save_inference_script(save_dir)
        
        logger.info(f"✅ Modelo guardado en {save_dir}")
    
    def _save_inference_script(self, save_dir: Path):
        """Guarda script para usar el modelo entrenado"""
        inference_code = '''#!/usr/bin/env python3
"""Script para usar el modelo de embeddings entrenado"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import json

class EmbeddingModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar config
        with open(f"{model_path}/embedding_config.json") as f:
            self.config = json.load(f)
        
        # Cargar modelo y tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        
        # Cargar projection layer
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(self.config['hidden_size'], 512),
            torch.nn.Tanh()
        ).to(self.device)
        self.projection.load_state_dict(torch.load(f"{model_path}/projection.pt"))
        
        self.model.eval()
        self.projection.eval()
    
    def encode(self, texts, batch_size=32):
        """Codifica textos a embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenizar
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config['max_length'],
                    return_tensors='pt'
                ).to(self.device)
                
                # Obtener embeddings
                outputs = self.model(**encoded)
                
                # Pooling
                if self.config['pooling_strategy'] == 'mean':
                    mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
                else:
                    pooled = outputs.last_hidden_state[:, 0]
                
                # Projection y normalización
                projected = self.projection(pooled)
                normalized = F.normalize(projected, p=2, dim=1)
                
                embeddings.append(normalized.cpu())
        
        return torch.cat(embeddings, dim=0).numpy()

# Ejemplo de uso
if __name__ == "__main__":
    model = EmbeddingModel(".")
    
    # Codificar textos
    texts = [
        "Això és un exemple de text en català",
        "Un altre text per codificar"
    ]
    
    embeddings = model.encode(texts)
    print(f"Shape: {embeddings.shape}")
    
    # Calcular similitud
    similarity = embeddings[0] @ embeddings[1].T
    print(f"Similitud: {similarity:.3f}")
'''
        
        with open(save_dir / "inference.py", "w", encoding="utf-8") as f:
            f.write(inference_code)


def main():
    """Función principal para entrenamiento en serie"""
    
    parser = argparse.ArgumentParser(description="Entrenar múltiples modelos de embeddings en serie")
    parser.add_argument("--train_file", type=str, default="/content/train_pairs.jsonl",
                       help="Archivo con pares de entrenamiento")
    parser.add_argument("--output_dir", type=str, 
                       default="/content/drive/MyDrive/models_embedding",
                       help="Directorio base de salida")
    parser.add_argument("--models", type=str, nargs='+',
                       help="Lista de tipos de modelo a entrenar (deja vacío para todos)")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Saltar modelos que ya tienen carpeta de salida")
    
    args = parser.parse_args()
    
    # Verificar archivo de entrada
    if not Path(args.train_file).exists():
        logger.error(f"❌ No se encontró {args.train_file}")
        sys.exit(1)
    
    # Filtrar modelos si se especifican
    models_to_train = MODELS_TO_TRAIN
    if args.models:
        models_to_train = [m for m in MODELS_TO_TRAIN if m['model_type'] in args.models]
        logger.info(f"Entrenando solo: {[m['model_type'] for m in models_to_train]}")
    
    # Filtrar modelos existentes si se solicita
    if args.skip_existing:
        filtered_models = []
        for model in models_to_train:
            output_path = Path(args.output_dir) / model['model_type']
            if output_path.exists():
                logger.info(f"⏭️  Saltando {model['model_type']} (ya existe)")
            else:
                filtered_models.append(model)
        models_to_train = filtered_models
    
    if not models_to_train:
        logger.info("No hay modelos para entrenar.")
        return
    
    # Crear entrenador en serie
    series_trainer = SeriesTrainer(args.output_dir, args.train_file)
    
    try:
        series_trainer.train_all_models(models_to_train)
        
        print("\n✅ Serie de entrenamiento completada!")
        print(f"📁 Modelos guardados en: {args.output_dir}/")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Serie interrumpida por el usuario")
    except Exception as e:
        logger.error(f"\n❌ Error durante la serie: {e}")
        raise

if __name__ == "__main__":
    main()