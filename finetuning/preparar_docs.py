#!/usr/bin/env python3
"""
Preparación óptima de datos para entrenar embeddings discriminativos
Optimizado para Windows 11 + GPU RTX 4060 (8GB VRAM)
"""

import os
import sys
import json
import hashlib
import gc
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack, csr_matrix
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuración específica Windows
if sys.platform == 'win32':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preparation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Estructura de un chunk con toda la metadata necesaria"""
    chunk_id: str
    doc_id: int
    text: str
    position: float  # 0.0 a 1.0
    length: int
    chunk_index: int
    total_chunks: int
    unique_terms: List[str] = None
    
@dataclass
class TrainingPair:
    """Par contrastivo para entrenamiento"""
    anchor_id: str
    positive_id: str
    negative_id: str
    difficulty: float
    pair_type: str  # 'hard', 'semi-hard', 'easy'

class OptimalDataPreparation:
    def __init__(self, 
                 chunk_size: int = 384,
                 overlap: int = 128,
                 model_name: str = "projecte-aina/roberta-base-ca-v2",
                 max_memory_gb: float = 12.0,
                 use_gpu: bool = False,
                 min_chunk_words: int = 20):
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_memory_gb = max_memory_gb
        self.use_gpu = use_gpu
        self.min_chunk_words = min_chunk_words
        
        # Para Windows - encoding explícito
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Inicializar tokenizer con manejo de errores
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"✓ Tokenizer cargado: {model_name}")
        except Exception as e:
            logger.error(f"Error cargando tokenizer: {e}")
            logger.info("Intenta: pip install transformers torch")
            raise
        
        self.vectorizer = None
        self._monitor_resources()
    
    def _monitor_resources(self):
        """Monitorea uso de recursos"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / 1e9
        
        total_memory = psutil.virtual_memory().total / 1e9
        available_memory = psutil.virtual_memory().available / 1e9
        
        logger.info(f"💾 Memoria: {memory_gb:.1f}GB usados / {available_memory:.1f}GB disponibles de {total_memory:.1f}GB")
        
        # Verificar GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"🎮 GPU detectada: {gpu_name} ({gpu_memory:.1f}GB)")
        except:
            logger.info("⚠️ GPU no disponible o torch no instalado")
    
    def prepare_from_folder(self, 
                          input_folder: str,
                          output_folder: str,
                          sample_size: Optional[int] = None) -> Dict:
        """Pipeline completo de preparación desde carpeta de TXTs"""
        
        logger.info("🚀 Iniciando preparación de datos...")
        
        # 1. Cargar documentos
        documents = self.load_documents(input_folder, sample_size)
        if not documents:
            raise ValueError(f"No se encontraron documentos en {input_folder}")
        logger.info(f"✓ Cargados {len(documents)} documentos")
        
        # 2. Crear chunks
        all_chunks = self.create_all_chunks(documents)
        logger.info(f"✓ Creados {len(all_chunks)} chunks")
        
        # Verificar memoria antes de continuar
        self._check_memory_usage(len(all_chunks))
        
        # 3. Calcular embeddings TF-IDF para análisis inicial
        tfidf_embeddings = self.compute_tfidf_embeddings_safe(all_chunks)
        
        # 4. Identificar términos únicos por chunk
        self.identify_unique_terms(all_chunks, tfidf_embeddings)
        
        # 5. Generar pares contrastivos
        training_pairs = self.generate_optimal_pairs_memory_efficient(all_chunks, tfidf_embeddings)
        logger.info(f"✓ Generados {len(training_pairs)} pares de entrenamiento")
        
        # 6. Validar pares
        training_pairs = self.validate_pairs(training_pairs, all_chunks)
        logger.info(f"✓ Validados {len(training_pairs)} pares")
        
        # 7. Dividir datos
        train, val, test = self.split_data(training_pairs, all_chunks)
        
        # 8. Guardar datasets
        self.save_prepared_data_windows(train, val, test, all_chunks, output_folder)
        
        # 9. Generar reporte
        report = self.generate_report(documents, all_chunks, training_pairs)
        
        # 10. Guardar reporte
        report_path = Path(output_folder) / 'preparation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Reporte guardado en {report_path}")
        
        return report
    
    def load_documents(self, folder: str, sample_size: Optional[int] = None) -> List[Dict]:
        """Carga documentos desde carpeta con manejo robusto de encoding"""
        documents = []
        folder_path = Path(folder)
        
        if not folder_path.exists():
            raise ValueError(f"La carpeta {folder} no existe")
        
        txt_files = sorted(folder_path.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"No se encontraron archivos .txt en {folder}")
            return documents
        
        if sample_size and sample_size < len(txt_files):
            txt_files = random.sample(txt_files, sample_size)
            logger.info(f"Muestreando {sample_size} de {len(txt_files)} archivos")
        
        for doc_id, txt_file in enumerate(tqdm(txt_files, desc="Cargando documentos")):
            try:
                # Intentar diferentes encodings para Windows
                content = None
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        with open(txt_file, 'r', encoding=encoding) as f:
                            content = f.read().strip()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    logger.warning(f"No se pudo decodificar {txt_file}")
                    continue
                
                # Preprocesar texto
                content = self._preprocess_text(content)
                
                if len(content.split()) >= self.min_chunk_words:
                    documents.append({
                        'doc_id': doc_id,
                        'filename': txt_file.name,
                        'content': content,
                        'length': len(content)
                    })
                
            except Exception as e:
                logger.error(f"Error cargando {txt_file}: {e}")
        
        return documents
    
    def _preprocess_text(self, text: str) -> str:
        """Limpieza específica para textos"""
        # Normalizar espacios
        text = ' '.join(text.split())
        
        # Eliminar caracteres problemáticos
        text = text.replace('\x00', '')
        text = text.replace('\ufeff', '')  # BOM
        
        # Normalizar comillas y guiones
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '-', '…': '...'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _check_memory_usage(self, n_chunks: int):
        """Verifica si hay suficiente memoria para procesar"""
        # Estimar uso de memoria para matriz de similitud
        estimated_memory_gb = (n_chunks ** 2 * 4) / 1e9  # float32
        available_memory = psutil.virtual_memory().available / 1e9
        
        if estimated_memory_gb > available_memory * 0.8:
            logger.warning(f"⚠️ Matriz completa requeriría {estimated_memory_gb:.1f}GB")
            logger.info("Se usará procesamiento por bloques para ahorrar memoria")
    
    def create_all_chunks(self, documents: List[Dict]) -> List[Chunk]:
        """Crea chunks de todos los documentos con validación"""
        all_chunks = []
        
        for doc in tqdm(documents, desc="Creando chunks"):
            chunks = self.create_chunks_from_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def create_chunks_from_document(self, document: Dict) -> List[Chunk]:
        """Crea chunks con overlap de un documento"""
        chunks = []
        text = document['content']
        doc_id = document['doc_id']
        
        # Tokenizar para chunking preciso
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        except Exception as e:
            logger.warning(f"Error tokenizando documento {doc_id}: {e}")
            return chunks
        
        if len(tokens) < self.min_chunk_words:
            return chunks
        
        stride = max(1, self.chunk_size - self.overlap)
        total_chunks = max(1, (len(tokens) - self.overlap) // stride)
        
        for i, start_idx in enumerate(range(0, len(tokens) - self.overlap, stride)):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            if len(chunk_tokens) < self.min_chunk_words:
                continue
            
            # Decodificar chunk
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Validar chunk
            if len(chunk_text.split()) >= self.min_chunk_words:
                chunk = Chunk(
                    chunk_id=f"doc{doc_id:04d}_chunk{i:03d}",
                    doc_id=doc_id,
                    text=chunk_text,
                    position=start_idx / max(1, len(tokens)),
                    length=len(chunk_tokens),
                    chunk_index=i,
                    total_chunks=total_chunks
                )
                chunks.append(chunk)
        
        return chunks
    
    def compute_tfidf_embeddings_safe(self, chunks: List[Chunk]):
        """Calcula embeddings TF-IDF con gestión de memoria"""
        n_chunks = len(chunks)
        texts = [chunk.text for chunk in chunks]
        
        # Configurar vectorizer optimizado
        self.vectorizer = TfidfVectorizer(
            max_features=min(5000, n_chunks // 2),
            min_df=2 if n_chunks > 1000 else 1,
            max_df=0.95,
            dtype=np.float32,
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'\b\w+\b'
        )
        
        # Para datasets grandes, mantener sparse
        if n_chunks > 5000:
            logger.info("Usando matrices sparse para eficiencia de memoria")
            return self.vectorizer.fit_transform(texts)
        else:
            return self.vectorizer.fit_transform(texts).toarray()
    
    def identify_unique_terms(self, chunks: List[Chunk], embeddings):
        """Identifica términos únicos de cada chunk"""
        feature_names = self.vectorizer.get_feature_names_out()
        is_sparse = hasattr(embeddings, 'toarray')
        
        for i, chunk in enumerate(chunks):
            if is_sparse:
                chunk_embedding = embeddings[i].toarray().flatten()
            else:
                chunk_embedding = embeddings[i]
            
            # Top términos por TF-IDF score
            top_indices = np.argsort(chunk_embedding)[-10:][::-1]
            top_terms = [
                feature_names[idx] 
                for idx in top_indices 
                if chunk_embedding[idx] > 0
            ]
            
            chunk.unique_terms = top_terms[:5]
    
    def generate_optimal_pairs_memory_efficient(self, chunks: List[Chunk], embeddings):
        """Genera pares con gestión eficiente de memoria"""
        n_chunks = len(chunks)
        pairs = []
        
        logger.info("Generando pares contrastivos...")
        
        # Para datasets muy grandes, usar procesamiento por bloques
        if n_chunks > 10000:
            return self._generate_pairs_blocked(chunks, embeddings)
        
        # Procesamiento estándar con optimizaciones
        is_sparse = hasattr(embeddings, 'toarray')
        batch_size = min(500, n_chunks // 10) if n_chunks > 1000 else n_chunks
        
        for i in tqdm(range(0, n_chunks, batch_size), desc="Procesando batches"):
            end_i = min(i + batch_size, n_chunks)
            
            # Calcular similitudes para este batch
            if is_sparse:
                batch_embeddings = embeddings[i:end_i]
                batch_similarities = cosine_similarity(batch_embeddings, embeddings)
            else:
                batch_similarities = cosine_similarity(
                    embeddings[i:end_i], 
                    embeddings
                )
            
            # Procesar cada chunk en el batch
            for j in range(end_i - i):
                anchor_idx = i + j
                anchor = chunks[anchor_idx]
                
                # Obtener positivos
                positives = self._get_positive_chunks(anchor_idx, chunks, max_distance=3)
                
                # Obtener negativos con similitudes
                similarities = batch_similarities[j]
                negatives = self._get_negative_chunks_from_similarities(
                    anchor_idx, chunks, similarities
                )
                
                # Crear pares limitados
                for pos_idx in positives[:3]:
                    for neg_idx, difficulty, pair_type in negatives[:5]:
                        pairs.append(TrainingPair(
                            anchor_id=anchor.chunk_id,
                            positive_id=chunks[pos_idx].chunk_id,
                            negative_id=chunks[neg_idx].chunk_id,
                            difficulty=float(difficulty),
                            pair_type=pair_type
                        ))
            
            # Limpiar memoria
            if i % 2000 == 0:
                gc.collect()
        
        return self._balance_pairs(pairs)
    
    def _generate_pairs_blocked(self, chunks: List[Chunk], embeddings):
        """Generación de pares para datasets muy grandes"""
        n_chunks = len(chunks)
        pairs = []
        block_size = 5000
        
        logger.info(f"Procesando {n_chunks} chunks en bloques de {block_size}")
        
        for block_start in tqdm(range(0, n_chunks, block_size), desc="Bloques"):
            block_end = min(block_start + block_size, n_chunks)
            block_chunks = chunks[block_start:block_end]
            
            # Embeddings del bloque
            if hasattr(embeddings, 'toarray'):
                block_embeddings = embeddings[block_start:block_end]
            else:
                block_embeddings = embeddings[block_start:block_end]
            
            # Generar pares dentro del bloque
            block_pairs = self._generate_pairs_for_block(
                block_chunks, block_embeddings, block_start
            )
            pairs.extend(block_pairs)
            
            # Limpiar memoria
            gc.collect()
            
            # Limitar total de pares
            if len(pairs) > 500000:
                logger.warning("Limitando a 500k pares para evitar overflow de memoria")
                break
        
        return self._balance_pairs(pairs)
    
    def _generate_pairs_for_block(self, block_chunks, block_embeddings, offset):
        """Genera pares para un bloque específico"""
        pairs = []
        n_block = len(block_chunks)
        
        # Calcular similitudes dentro del bloque
        if hasattr(block_embeddings, 'toarray'):
            similarities = cosine_similarity(block_embeddings)
        else:
            similarities = cosine_similarity(block_embeddings)
        
        for i in range(n_block):
            anchor = block_chunks[i]
            global_idx = offset + i
            
            # Positivos (mismo documento)
            positives = []
            for j in range(n_block):
                if (block_chunks[j].doc_id == anchor.doc_id and 
                    i != j and 
                    abs(block_chunks[j].chunk_index - anchor.chunk_index) <= 3):
                    positives.append(j)
            
            # Negativos del bloque
            negatives = []
            for j in range(n_block):
                if block_chunks[j].doc_id != anchor.doc_id:
                    sim = similarities[i, j]
                    if 0.7 <= sim <= 0.9:
                        negatives.append((j, sim, 'hard'))
                    elif 0.4 <= sim < 0.7:
                        negatives.append((j, sim, 'semi-hard'))
                    elif sim < 0.4:
                        negatives.append((j, sim, 'easy'))
            
            # Crear pares limitados
            for pos_idx in positives[:2]:
                for neg_idx, diff, ptype in sorted(negatives, key=lambda x: x[1], reverse=True)[:3]:
                    pairs.append(TrainingPair(
                        anchor_id=anchor.chunk_id,
                        positive_id=block_chunks[pos_idx].chunk_id,
                        negative_id=block_chunks[neg_idx].chunk_id,
                        difficulty=float(diff),
                        pair_type=ptype
                    ))
        
        return pairs
    
    def _get_positive_chunks(self, anchor_idx: int, chunks: List[Chunk], 
                           max_distance: int = 3) -> List[int]:
        """Obtiene chunks positivos (mismo documento, cercanos)"""
        anchor = chunks[anchor_idx]
        positives = []
        
        # Buscar en ventana limitada para eficiencia
        search_range = min(100, len(chunks) // 10)
        start = max(0, anchor_idx - search_range)
        end = min(len(chunks), anchor_idx + search_range)
        
        for j in range(start, end):
            chunk = chunks[j]
            if (chunk.doc_id == anchor.doc_id and 
                j != anchor_idx and 
                abs(chunk.chunk_index - anchor.chunk_index) <= max_distance):
                positives.append(j)
        
        return positives[:5]
    
    def _get_negative_chunks_from_similarities(self, anchor_idx: int, 
                                             chunks: List[Chunk],
                                             similarities: np.ndarray) -> List[Tuple]:
        """Obtiene negativos basados en similitudes pre-calculadas"""
        anchor = chunks[anchor_idx]
        negatives = []
        
        # Índices de otros documentos
        other_doc_indices = [
            i for i in range(len(similarities))
            if i < len(chunks) and chunks[i].doc_id != anchor.doc_id
        ]
        
        if not other_doc_indices:
            return []
        
        # Categorizar por similitud
        for i in other_doc_indices:
            sim = similarities[i]
            if 0.7 <= sim <= 0.9:
                negatives.append((i, sim, 'hard'))
            elif 0.4 <= sim < 0.7:
                negatives.append((i, sim, 'semi-hard'))
            elif sim < 0.4:
                negatives.append((i, sim, 'easy'))
        
        # Ordenar y diversificar
        negatives.sort(key=lambda x: x[1], reverse=True)
        
        # Seleccionar con diversidad
        selected = []
        selected_terms = set()
        
        for neg_idx, sim, pair_type in negatives:
            neg_terms = set(chunks[neg_idx].unique_terms or [])
            
            # Priorizar chunks con términos diferentes
            overlap = len(neg_terms.intersection(selected_terms))
            if overlap < 2:  # Permitir algo de overlap
                selected.append((neg_idx, sim, pair_type))
                selected_terms.update(neg_terms)
                
                if len(selected) >= 10:
                    break
        
        return selected
    
    def _balance_pairs(self, pairs: List[TrainingPair]) -> List[TrainingPair]:
        """Balancea pares por tipo para entrenamiento óptimo"""
        if not pairs:
            return []
        
        # Agrupar por tipo
        pairs_by_type = defaultdict(list)
        for pair in pairs:
            pairs_by_type[pair.pair_type].append(pair)
        
        # Estadísticas
        for ptype, plist in pairs_by_type.items():
            logger.info(f"  {ptype}: {len(plist)} pares")
        
        # Calcular cantidad objetivo por tipo
        total_desired = min(200000, len(pairs))  # Máximo 200k pares
        type_ratios = {'hard': 0.5, 'semi-hard': 0.3, 'easy': 0.2}
        
        balanced_pairs = []
        for pair_type, ratio in type_ratios.items():
            type_pairs = pairs_by_type[pair_type]
            n_desired = int(total_desired * ratio)
            
            if len(type_pairs) > n_desired:
                # Submuestrear con semilla para reproducibilidad
                random.seed(42)
                selected = random.sample(type_pairs, n_desired)
            else:
                selected = type_pairs
            
            balanced_pairs.extend(selected)
        
        random.shuffle(balanced_pairs)
        return balanced_pairs
    
    def validate_pairs(self, pairs: List[TrainingPair], chunks: List[Chunk]) -> List[TrainingPair]:
        """Validar calidad de pares antes de guardar"""
        if not pairs:
            return []
        
        chunk_dict = {c.chunk_id: c for c in chunks}
        valid_pairs = []
        invalid_count = 0
        
        for pair in pairs:
            try:
                anchor = chunk_dict[pair.anchor_id]
                positive = chunk_dict[pair.positive_id]
                negative = chunk_dict[pair.negative_id]
                
                # Validaciones
                if (len(anchor.text.split()) >= 10 and
                    len(positive.text.split()) >= 10 and
                    len(negative.text.split()) >= 10 and
                    anchor.text != positive.text and
                    anchor.text != negative.text and
                    positive.text != negative.text):
                    valid_pairs.append(pair)
                else:
                    invalid_count += 1
            except KeyError:
                invalid_count += 1
        
        if invalid_count > 0:
            logger.warning(f"Descartados {invalid_count} pares inválidos")
        
        return valid_pairs
    
    def split_data(self, pairs: List[TrainingPair], chunks: List[Chunk],
                  train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple:
        """Divide datos asegurando que documentos completos estén en un solo split"""
        
        if not pairs:
            return [], [], []
        
        # Obtener documentos únicos
        doc_ids = list(set(chunk.doc_id for chunk in chunks))
        random.seed(42)  # Reproducibilidad
        random.shuffle(doc_ids)
        
        # Dividir por documentos
        n_docs = len(doc_ids)
        n_train = int(n_docs * train_ratio)
        n_val = int(n_docs * val_ratio)
        
        train_docs = set(doc_ids[:n_train])
        val_docs = set(doc_ids[n_train:n_train + n_val])
        test_docs = set(doc_ids[n_train + n_val:])
        
        # Crear índice chunk_id -> doc_id
        chunk_to_doc = {chunk.chunk_id: chunk.doc_id for chunk in chunks}
        
        # Dividir pares según documentos
        train_pairs = []
        val_pairs = []
        test_pairs = []
        
        for pair in pairs:
            try:
                anchor_doc = chunk_to_doc[pair.anchor_id]
                
                if anchor_doc in train_docs:
                    train_pairs.append(pair)
                elif anchor_doc in val_docs:
                    val_pairs.append(pair)
                else:
                    test_pairs.append(pair)
            except KeyError:
                continue
        
        logger.info(f"División: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
        
        return train_pairs, val_pairs, test_pairs
    
    def save_prepared_data_windows(self, train_pairs, val_pairs, test_pairs,
                                  chunks: List[Chunk], output_folder: str):
        """Guardado optimizado para Windows sin compresión para uso directo"""
        
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Crear índice chunk_id -> chunk para búsqueda rápida
        chunk_dict = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Guardar chunks en formato JSON directo
        logger.info("Guardando chunks...")
        chunks_data = []
        for chunk in chunks:
            chunk_data = asdict(chunk)
            # Convertir numpy types a Python natives
            chunk_data['position'] = float(chunk_data['position'])
            chunks_data.append(chunk_data)
        
        with open(output_path / 'chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Guardados {len(chunks)} chunks")
        
        # Guardar pares en JSONL sin comprimir
        for split_name, pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
            if not pairs:
                logger.warning(f"No hay pares para {split_name}")
                continue
                
            filepath = output_path / f'{split_name}_pairs.jsonl'
            
            with open(filepath, 'w', encoding='utf-8') as f:
                batch = []
                
                for i, pair in enumerate(tqdm(pairs, desc=f"Guardando {split_name}")):
                    try:
                        anchor = chunk_dict[pair.anchor_id]
                        positive = chunk_dict[pair.positive_id]
                        negative = chunk_dict[pair.negative_id]
                        
                        pair_data = {
                            'anchor': anchor.text,
                            'positive': positive.text,
                            'negative': negative.text,
                            'difficulty': float(pair.difficulty),
                            'type': pair.pair_type,
                            'ids': {
                                'anchor': pair.anchor_id,
                                'positive': pair.positive_id,
                                'negative': pair.negative_id
                            }
                        }
                        batch.append(json.dumps(pair_data, ensure_ascii=False))
                        
                        # Escribir cada 1000 pares
                        if len(batch) >= 1000:
                            f.write('\n'.join(batch) + '\n')
                            batch = []
                    except KeyError as e:
                        logger.warning(f"Chunk no encontrado: {e}")
                        continue
                
                # Escribir resto
                if batch:
                    f.write('\n'.join(batch) + '\n')
            
            logger.info(f"✓ Guardados {len(pairs)} pares en {filepath}")
        
        # Guardar metadata
        metadata = {
            'total_chunks': len(chunks),
            'total_documents': len(set(chunk.doc_id for chunk in chunks)),
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'min_chunk_words': self.min_chunk_words,
            'splits': {
                'train': len(train_pairs),
                'val': len(val_pairs),
                'test': len(test_pairs)
            },
            'pair_types': self._count_pair_types(train_pairs)
        }
        
        with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Metadata guardada")
    
    def _count_pair_types(self, pairs: List[TrainingPair]) -> Dict[str, int]:
        """Cuenta tipos de pares"""
        counts = defaultdict(int)
        for pair in pairs:
            counts[pair.pair_type] += 1
        return dict(counts)
    
    def generate_report(self, documents: List[Dict], chunks: List[Chunk], 
                       pairs: List[TrainingPair]) -> Dict:
        """Genera reporte detallado de preparación de datos"""
        
        # Cálculos básicos
        chunk_lengths = [len(chunk.text.split()) for chunk in chunks]
        pair_difficulties = [p.difficulty for p in pairs] if pairs else [0]
        
        # Contar tipos
        pair_type_counts = defaultdict(int)
        for pair in pairs:
            pair_type_counts[pair.pair_type] += 1
        
        report = {
            'summary': {
                'total_documents': len(documents),
                'total_chunks': len(chunks),
                'total_pairs': len(pairs),
                'avg_chunks_per_doc': len(chunks) / max(1, len(documents)),
                'avg_chunk_length_words': np.mean(chunk_lengths) if chunk_lengths else 0,
                'std_chunk_length_words': np.std(chunk_lengths) if chunk_lengths else 0
            },
            'chunk_stats': {
                'min_length': min(chunk_lengths) if chunk_lengths else 0,
                'max_length': max(chunk_lengths) if chunk_lengths else 0,
                'median_length': np.median(chunk_lengths) if chunk_lengths else 0
            },
            'pair_distribution': dict(pair_type_counts),
            'difficulty_stats': {
                'mean': np.mean(pair_difficulties) if pair_difficulties else 0,
                'std': np.std(pair_difficulties) if pair_difficulties else 0,
                'min': min(pair_difficulties) if pair_difficulties else 0,
                'max': max(pair_difficulties) if pair_difficulties else 0
            },
            'quality_metrics': self._generate_quality_metrics(pairs, chunks)
        }
        
        # Mostrar reporte
        logger.info("\n" + "="*50)
        logger.info("📊 REPORTE DE PREPARACIÓN")
        logger.info("="*50)
        logger.info(f"📄 Documentos procesados: {report['summary']['total_documents']}")
        logger.info(f"📦 Chunks creados: {report['summary']['total_chunks']}")
        logger.info(f"🔗 Pares generados: {report['summary']['total_pairs']}")
        logger.info(f"📏 Longitud promedio chunks: {report['summary']['avg_chunk_length_words']:.1f} palabras")
        
        if report['pair_distribution']:
            logger.info("\n📊 Distribución de pares:")
            for ptype, count in report['pair_distribution'].items():
                percentage = (count / len(pairs)) * 100 if pairs else 0
                logger.info(f"  - {ptype}: {count} ({percentage:.1f}%)")
        
        logger.info("="*50 + "\n")
        
        return report
    
    def _generate_quality_metrics(self, pairs: List[TrainingPair], chunks: List[Chunk]) -> Dict:
        """Genera métricas de calidad adicionales"""
        if not pairs:
            return {}
        
        chunk_dict = {c.chunk_id: c for c in chunks}
        
        # Calcular longitudes promedio
        anchor_lengths = []
        for pair in pairs[:1000]:  # Muestra para eficiencia
            try:
                anchor = chunk_dict[pair.anchor_id]
                anchor_lengths.append(len(anchor.text.split()))
            except KeyError:
                continue
        
        # Documentos únicos en pares
        unique_docs = set()
        for pair in pairs:
            try:
                anchor_doc = chunk_dict[pair.anchor_id].doc_id
                unique_docs.add(anchor_doc)
            except KeyError:
                continue
        
        return {
            'avg_anchor_length_words': np.mean(anchor_lengths) if anchor_lengths else 0,
            'unique_documents_in_pairs': len(unique_docs),
            'document_coverage': len(unique_docs) / max(1, len(set(c.doc_id for c in chunks)))
        }


def main():
    """Función principal con manejo robusto de argumentos"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preparación óptima de datos para embeddings RoBERTa-ca-v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input_folder", type=str, default="PNTs",
                       help="Carpeta con documentos TXT")
    parser.add_argument("--output_folder", type=str, default="prepared_data",
                       help="Carpeta para guardar datos preparados")
    parser.add_argument("--chunk_size", type=int, default=384,
                       help="Tamaño de chunks en tokens")
    parser.add_argument("--overlap", type=int, default=128,
                       help="Overlap entre chunks")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Número de documentos a procesar (None = todos)")
    parser.add_argument("--max_memory_gb", type=float, default=12.0,
                       help="Límite de memoria RAM en GB")
    parser.add_argument("--min_chunk_words", type=int, default=20,
                       help="Mínimo de palabras por chunk")
    
    args = parser.parse_args()
    
    try:
        # Preparar datos
        preparator = OptimalDataPreparation(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            max_memory_gb=args.max_memory_gb,
            min_chunk_words=args.min_chunk_words
        )
        
        report = preparator.prepare_from_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            sample_size=args.sample_size
        )
        
        print("\n✅ Preparación completada exitosamente!")
        print(f"📁 Datos guardados en: {args.output_folder}/")
        print("\n🎯 Próximos pasos:")
        print("1. Revisar metadata.json para estadísticas")
        print("2. Usar train_pairs.jsonl para entrenamiento")
        print("3. Validar con val_pairs.jsonl")
        print("\n💡 Para cargar los datos:")
        print("   import json")
        print("   # Cargar pares línea por línea")
        print("   with open('prepared_data/train_pairs.jsonl', 'r', encoding='utf-8') as f:")
        print("       pairs = [json.loads(line) for line in f]")
        print("\n   # O cargar todos los chunks")
        print("   with open('prepared_data/chunks.json', 'r', encoding='utf-8') as f:")
        print("       chunks = json.load(f)")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        logger.info("\nVerificar:")
        logger.info("1. La carpeta de entrada existe y contiene archivos .txt")
        logger.info("2. Tienes permisos de escritura en la carpeta de salida")
        logger.info("3. Las dependencias están instaladas (ver requirements.txt)")
        raise


if __name__ == "__main__":
    main()