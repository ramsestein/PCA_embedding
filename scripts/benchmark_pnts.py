#!/usr/bin/env python3
"""
Script de benchmark para evaluar diferentes estrategias de re-ranking sobre documentos PNTs.
Implementa tres estrategias:
1. Baseline: Solo embeddings (modelo all-mini)
2. MRF solo: Solo Markov Random Field
3. MRF + Embeddings: Combinación de ambas estrategias
"""

import os
import json
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import regex as re

# Importar módulos del re-ranker
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rerank_markov.types import Chunk, ScoredChunk
from rerank_markov.mrf import mrf_sd_score
from rerank_markov.utils import tokenize
from rerank_markov.index_stats import compute_corpus_statistics, CorpusStats


class PNTsBenchmark:
    """Clase para ejecutar benchmarks sobre documentos PNTs."""
    
    def __init__(self, pnts_dir: str = "PNTs", benchmark_file: str = "benchmark/preguntas_con_docs_es.json"):
        self.pnts_dir = Path(pnts_dir)
        self.benchmark_file = Path(benchmark_file)
        self.model = None
        self.chunks = []
        self.corpus_stats = None
        
    def load_model(self):
        """Carga el modelo de embeddings SAPBERT."""
        print("Cargando modelo SAPBERT-UMLS...")
        
        # Cambiar a usar el mejor modelo SAPBERT
        model_path = "sapbert-umls/model-0_0029"
        
        if not os.path.exists(model_path):
            raise ValueError(f"Modelo no encontrado en: {model_path}")
            
        print(f"Modelo cargado desde: {model_path}")
        
        try:
            self.model = SentenceTransformer(model_path)
            print("Modelo cargado exitosamente!")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            raise
        
    def load_pnts_documents(self) -> List[Chunk]:
        """Carga todos los documentos PNTs y los convierte en chunks."""
        print(f"Cargando documentos desde: {self.pnts_dir}")
        
        chunks = []
        chunk_id = 0
        
        for txt_file in self.pnts_dir.glob("*.txt"):
            if txt_file.stat().st_size == 0:  # Saltar archivos vacíos
                continue
                
            print(f"Procesando: {txt_file.name}")
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Dividir en chunks de aproximadamente 200-300 palabras
            sentences = re.split(r'[.!?]+', content)
            current_chunk = ""
            current_position = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Si el chunk actual + la nueva oración es muy largo, crear nuevo chunk
                if len(current_chunk.split()) + len(sentence.split()) > 300:
                    if current_chunk.strip():
                        # Crear chunk anterior
                        chunk = Chunk(
                            id=f"chunk_{chunk_id:03d}",
                            text=current_chunk.strip(),
                            doc_id=txt_file.stem,
                            position=current_position,
                            embedding=None,
                            meta={"source_file": txt_file.name}
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                        current_position += 1
                    
                    # Iniciar nuevo chunk
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
            
            # Agregar el último chunk si existe
            if current_chunk.strip():
                chunk = Chunk(
                    id=f"chunk_{chunk_id:03d}",
                    text=current_chunk.strip(),
                    doc_id=txt_file.stem,
                    position=current_position,
                    embedding=None,
                    meta={"source_file": txt_file.name}
                )
                chunks.append(chunk)
                chunk_id += 1
        
        print(f"Total de chunks creados: {len(chunks)}")
        self.chunks = chunks
        return chunks
    
    def compute_embeddings(self):
        """Calcula embeddings para todos los chunks."""
        if not self.model:
            raise ValueError("Modelo no cargado")
            
        print("Calculando embeddings para todos los chunks...")
        
        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]
            
        print("Embeddings calculados exitosamente!")
        
    def compute_corpus_statistics(self):
        """Calcula estadísticas del corpus para MRF."""
        print("Calculando estadísticas del corpus...")
        self.corpus_stats = compute_corpus_statistics(self.chunks)
        print("Estadísticas del corpus calculadas!")
        
    def search_embeddings_only(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """Búsqueda usando solo similitud de embeddings (baseline)."""
        if not self.model:
            raise ValueError("Modelo no cargado")
            
        # Calcular embedding de la query
        query_embedding = self.model.encode([query])[0]
        
        # Calcular similitudes coseno
        scored_chunks = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                scored_chunk = ScoredChunk(
                    chunk=chunk,
                    total_score=similarity,
                    embedding_score=similarity,
                    ppr_score=0.0,
                    qlm_score=0.0,
                    mrf_score=0.0,
                    rank=0
                )
                scored_chunks.append(scored_chunk)
        
        # Ordenar por score y asignar ranks
        scored_chunks.sort(key=lambda x: x.total_score, reverse=True)
        for i, scored_chunk in enumerate(scored_chunks):
            scored_chunk.rank = i + 1
            
        return scored_chunks[:top_k]
    
    def search_mrf_only(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """Búsqueda usando solo MRF (Markov Random Field)."""
        if not self.corpus_stats:
            raise ValueError("Estadísticas del corpus no calculadas")
            
        # Calcular scores MRF para todos los chunks
        scored_chunks = []
        for chunk in self.chunks:
            mrf_score = mrf_sd_score(query, chunk, window=8)
            
            scored_chunk = ScoredChunk(
                chunk=chunk,
                total_score=mrf_score,
                embedding_score=0.0,
                ppr_score=0.0,
                qlm_score=0.0,
                mrf_score=mrf_score,
                rank=0
            )
            scored_chunks.append(scored_chunk)
        
        # Ordenar por score y asignar ranks
        scored_chunks.sort(key=lambda x: x.total_score, reverse=True)
        for i, scored_chunk in enumerate(scored_chunks):
            scored_chunk.rank = i + 1
            
        return scored_chunks[:top_k]
    
    def search_mrf_plus_embeddings(self, query: str, top_k: int = 5, 
                                  mrf_weight: float = 0.6, 
                                  embedding_weight: float = 0.4) -> List[ScoredChunk]:
        """Búsqueda combinando MRF y embeddings."""
        if not self.model or not self.corpus_stats:
            raise ValueError("Modelo o estadísticas del corpus no cargados")
            
        # Calcular embedding de la query
        query_embedding = self.model.encode([query])[0]
        
        # Calcular scores combinados
        scored_chunks = []
        for chunk in self.chunks:
            # Score de embeddings
            if chunk.embedding is not None:
                embedding_score = self._cosine_similarity(query_embedding, chunk.embedding)
            else:
                embedding_score = 0.0
                
            # Score MRF
            mrf_score = mrf_sd_score(query, chunk, window=8)
            
            # Combinar scores
            total_score = (mrf_weight * mrf_score + 
                          embedding_weight * embedding_score)
            
            scored_chunk = ScoredChunk(
                chunk=chunk,
                total_score=total_score,
                embedding_score=embedding_score,
                ppr_score=0.0,
                qlm_score=0.0,
                mrf_score=mrf_score,
                rank=0
            )
            scored_chunks.append(scored_chunk)
        
        # Ordenar por score y asignar ranks
        scored_chunks.sort(key=lambda x: x.total_score, reverse=True)
        for i, scored_chunk in enumerate(scored_chunks):
            scored_chunk.rank = i + 1
            
        return scored_chunks[:top_k]
    
    def search_mrf_plus_embeddings_adaptive(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """Búsqueda combinando MRF y embeddings con pesos adaptativos."""
        if not self.model or not self.corpus_stats:
            raise ValueError("Modelo o estadísticas del corpus no cargados")
            
        # Calcular embedding de la query
        query_embedding = self.model.encode([query])[0]
        
        # Calcular pesos adaptativos
        query_length = len(query.split())
        mrf_weight, embedding_weight = self._calculate_adaptive_weights(query_length)
        
        # Calcular scores combinados
        scored_chunks = []
        for chunk in self.chunks:
            # Score de embeddings
            if chunk.embedding is not None:
                embedding_score = self._cosine_similarity(query_embedding, chunk.embedding)
            else:
                embedding_score = 0.0
                
            # Score MRF
            mrf_score = mrf_sd_score(query, chunk, window=8)
            
            # Combinar scores con pesos adaptativos
            total_score = (mrf_weight * mrf_score + 
                          embedding_weight * embedding_score)
            
            scored_chunk = ScoredChunk(
                chunk=chunk,
                total_score=total_score,
                embedding_score=embedding_score,
                ppr_score=0.0,
                qlm_score=0.0,
                mrf_score=mrf_score,
                rank=0
            )
            scored_chunks.append(scored_chunk)
        
        # Ordenar por score y asignar ranks
        scored_chunks.sort(key=lambda x: x.total_score, reverse=True)
        for i, scored_chunk in enumerate(scored_chunks):
            scored_chunk.rank = i + 1
            
        return scored_chunks[:top_k]
    
    def search_mrf_plus_embeddings_optimized_windows(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """Búsqueda combinando MRF y embeddings con ventanas MRF optimizadas."""
        if not self.model or not self.corpus_stats:
            raise ValueError("Modelo o estadísticas del corpus no cargados")
            
        # Calcular embedding de la query
        query_embedding = self.model.encode([query])[0]
        
        # Calcular pesos adaptativos
        query_length = len(query.split())
        mrf_weight, embedding_weight = self._calculate_adaptive_weights(query_length)
        
        # Calcular scores combinados
        scored_chunks = []
        for chunk in self.chunks:
            # Score de embeddings
            if chunk.embedding is not None:
                embedding_score = self._cosine_similarity(query_embedding, chunk.embedding)
            else:
                embedding_score = 0.0
                
            # Score MRF con ventanas optimizadas
            mrf_score = self._optimized_mrf_scoring(query, chunk)
            
            # Combinar scores con pesos adaptativos
            total_score = (mrf_weight * mrf_score + 
                          embedding_weight * embedding_score)
            
            scored_chunk = ScoredChunk(
                chunk=chunk,
                total_score=total_score,
                embedding_score=embedding_score,
                ppr_score=0.0,
                qlm_score=0.0,
                mrf_score=mrf_score,
                rank=0
            )
            scored_chunks.append(scored_chunk)
        
        # Ordenar por score y asignar ranks
        scored_chunks.sort(key=lambda x: x.total_score, reverse=True)
        for i, scored_chunk in enumerate(scored_chunks):
            scored_chunk.rank = i + 1
            
        return scored_chunks[:top_k]
    
    def search_mrf_plus_embeddings_intelligent_normalization(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """Búsqueda combinando MRF y embeddings con normalización inteligente."""
        if not self.model or not self.corpus_stats:
            raise ValueError("Modelo o estadísticas del corpus no cargados")
            
        # Calcular embedding de la query
        query_embedding = self.model.encode([query])[0]
        
        # Calcular pesos adaptativos
        query_length = len(query.split())
        mrf_weight, embedding_weight = self._calculate_adaptive_weights(query_length)
        
        # Calcular scores para todos los chunks
        mrf_scores = []
        embedding_scores = []
        
        for chunk in self.chunks:
            # Score MRF
            mrf_score = mrf_sd_score(query, chunk, window=8)
            mrf_scores.append(mrf_score)
            
            # Score de embeddings
            if chunk.embedding is not None:
                embedding_score = self._cosine_similarity(query_embedding, chunk.embedding)
            else:
                embedding_score = 0.0
            embedding_scores.append(embedding_score)
        
        # Aplicar normalización inteligente
        mrf_normalized, embedding_normalized = self._intelligent_score_normalization(mrf_scores, embedding_scores)
        
        # Calcular scores combinados
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            # Combinar scores normalizados con pesos adaptativos
            total_score = (mrf_weight * mrf_normalized[i] + 
                          embedding_weight * embedding_normalized[i])
            
            scored_chunk = ScoredChunk(
                chunk=chunk,
                total_score=total_score,
                embedding_score=embedding_normalized[i],
                ppr_score=0.0,
                qlm_score=0.0,
                mrf_score=mrf_normalized[i],
                rank=0
            )
            scored_chunks.append(scored_chunk)
        
        # Ordenar por score y asignar ranks
        scored_chunks.sort(key=lambda x: x.total_score, reverse=True)
        for i, scored_chunk in enumerate(scored_chunks):
            scored_chunk.rank = i + 1
            
        return scored_chunks[:top_k]
    
    def search_mrf_plus_embeddings_ensemble(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """Búsqueda combinando MRF y embeddings con ensemble de múltiples modelos MRF."""
        if not self.model or not self.corpus_stats:
            raise ValueError("Modelo o estadísticas del corpus no cargados")
            
        # Calcular embedding de la query
        query_embedding = self.model.encode([query])[0]
        
        # Calcular pesos adaptativos
        query_length = len(query.split())
        mrf_weight, embedding_weight = self._calculate_adaptive_weights(query_length)
        
        # Calcular scores de ensemble MRF
        mrf_ensemble_scores = self._ensemble_mrf_scoring(query)
        
        # Calcular scores combinados
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            # Score de embeddings
            if chunk.embedding is not None:
                embedding_score = self._cosine_similarity(query_embedding, chunk.embedding)
            else:
                embedding_score = 0.0
            
            # Combinar scores con pesos adaptativos
            total_score = (mrf_weight * mrf_ensemble_scores[i] + 
                          embedding_weight * embedding_score)
            
            scored_chunk = ScoredChunk(
                chunk=chunk,
                total_score=total_score,
                embedding_score=embedding_score,
                ppr_score=0.0,
                qlm_score=0.0,
                mrf_score=mrf_ensemble_scores[i],
                rank=0
            )
            scored_chunks.append(scored_chunk)
        
        # Ordenar por score y asignar ranks
        scored_chunks.sort(key=lambda x: x.total_score, reverse=True)
        for i, scored_chunk in enumerate(scored_chunks):
            scored_chunk.rank = i + 1
            
        return scored_chunks[:top_k]
    
    def search_mrf_plus_embeddings_adaptive_learning(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """Búsqueda combinando MRF y embeddings con sistema de aprendizaje adaptativo."""
        if not self.model or not self.corpus_stats:
            raise ValueError("Modelo o estadísticas del corpus no cargados")
            
        # Calcular embedding de la query
        query_embedding = self.model.encode([query])[0]
        
        # Obtener pesos adaptativos con aprendizaje
        mrf_weight, embedding_weight = self._get_adaptive_weights_with_learning(query)
        
        # Calcular scores de ensemble MRF
        mrf_ensemble_scores = self._ensemble_mrf_scoring(query)
        
        # Calcular scores combinados
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            # Score de embeddings
            if chunk.embedding is not None:
                embedding_score = self._cosine_similarity(query_embedding, chunk.embedding)
            else:
                embedding_score = 0.0
            
            # Combinar scores con pesos adaptativos aprendidos
            total_score = (mrf_weight * mrf_ensemble_scores[i] + 
                          embedding_weight * embedding_score)
            
            scored_chunk = ScoredChunk(
                chunk=chunk,
                total_score=total_score,
                embedding_score=embedding_score,
                ppr_score=0.0,
                qlm_score=0.0,
                mrf_score=mrf_ensemble_scores[i],
                rank=0
            )
            scored_chunks.append(scored_chunk)
        
        # Ordenar por score y asignar ranks
        scored_chunks.sort(key=lambda x: x.total_score, reverse=True)
        for i, scored_chunk in enumerate(scored_chunks):
            scored_chunk.rank = i + 1
            
        return scored_chunks[:top_k]
    
    def _calculate_adaptive_weights(self, query_length: int) -> Tuple[float, float]:
        """Calcula pesos adaptativos basados en la longitud de la query."""
        
        if query_length <= 3:
            # Queries cortas: más peso a embeddings (mejor para términos simples)
            mrf_weight = 0.4
            embedding_weight = 0.6
        elif query_length <= 6:
            # Queries medianas: balance equilibrado
            mrf_weight = 0.5
            embedding_weight = 0.5
        elif query_length <= 10:
            # Queries largas: más peso a MRF (mejor para queries complejas)
            mrf_weight = 0.7
            embedding_weight = 0.3
        else:
            # Queries muy largas: dominancia de MRF
            mrf_weight = 0.8
            embedding_weight = 0.2
        
        return mrf_weight, embedding_weight
    
    def _optimized_mrf_scoring(self, query: str, chunk) -> float:
        """MRF con ventanas optimizadas basadas en características de la query."""
        
        query_tokens = tokenize(query)
        query_length = len(query_tokens)
        
        # Ventana adaptativa basada en la longitud de la query
        if query_length <= 3:
            # Queries cortas: ventana pequeña para términos específicos
            windows = [3, 4, 5]
        elif query_length <= 6:
            # Queries medianas: ventana media
            windows = [5, 6, 7]
        else:
            # Queries largas: ventana grande para capturar contexto
            windows = [7, 8, 9]
        
        # Calcular scores con diferentes ventanas y promediar
        scores = []
        for window in windows:
            score = mrf_sd_score(query, chunk, window=window)
            scores.append(score)
        
        # Promedio ponderado (dar más peso a ventanas intermedias)
        if len(scores) == 3:
            weights = [0.3, 0.5, 0.2]  # Peso alto en ventana intermedia
            return np.average(scores, weights=weights)
        else:
            return np.mean(scores) if scores else 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _intelligent_score_normalization(self, mrf_scores: List[float], embedding_scores: List[float]) -> Tuple[List[float], List[float]]:
        """Normalización inteligente que preserva la distribución y evita dominancia."""
        
        # Convertir a numpy arrays
        mrf_array = np.array(mrf_scores)
        embedding_array = np.array(embedding_scores)
        
        # Normalización robusta (resistente a outliers)
        mrf_normalized = self._robust_normalize(mrf_array)
        embedding_normalized = self._robust_normalize(embedding_array)
        
        # Escalar a rangos específicos para evitar dominancia
        mrf_scaled = self._scale_to_range(mrf_normalized, 0.3, 0.7)
        embedding_scaled = self._scale_to_range(embedding_normalized, 0.3, 0.7)
        
        return mrf_scaled.tolist(), embedding_scaled.tolist()
    
    def _robust_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalización robusta usando percentiles en lugar de media/desv estándar."""
        
        if len(scores) == 0:
            return scores
        
        # Usar percentiles para normalización robusta
        p25 = np.percentile(scores, 25)
        p75 = np.percentile(scores, 75)
        iqr = p75 - p25
        
        if iqr == 0:
            # Si no hay variabilidad, normalizar por el rango total
            score_range = scores.max() - scores.min()
            if score_range == 0:
                return np.zeros_like(scores)
            return (scores - scores.min()) / score_range
        
        # Normalización robusta usando IQR
        normalized = (scores - p25) / iqr
        
        # Clipping para evitar valores extremos
        normalized = np.clip(normalized, -2, 2)
        
        return normalized
    
    def _scale_to_range(self, scores: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Escala los scores a un rango específico."""
        
        if len(scores) == 0:
            return scores
        
        score_min = scores.min()
        score_max = scores.max()
        
        if score_max == score_min:
            # Si no hay variabilidad, asignar valor medio
            return np.full_like(scores, (min_val + max_val) / 2)
        
        # Escalar al rango deseado
        scaled = min_val + (scores - score_min) * (max_val - min_val) / (score_max - score_min)
        
        return scaled
    
    def _ensemble_mrf_scoring(self, query: str) -> List[float]:
        """Ensemble de múltiples modelos MRF con diferentes configuraciones."""
        
        # Configuraciones del ensemble MRF
        mrf_configs = [
            # Configuración 1: Unigram pesado
            {'w_unigram': 0.8, 'w_ordered': 0.1, 'w_unordered': 0.1, 'window': 8},
            # Configuración 2: Bigramas ordenados pesados
            {'w_unigram': 0.3, 'w_ordered': 0.6, 'w_unordered': 0.1, 'window': 6},
            # Configuración 3: Ventanas no ordenadas pesadas
            {'w_unigram': 0.2, 'w_ordered': 0.1, 'w_unordered': 0.7, 'window': 10},
            # Configuración 4: Balanceado con ventana pequeña
            {'w_unigram': 0.5, 'w_ordered': 0.3, 'w_unordered': 0.2, 'window': 4},
            # Configuración 5: Balanceado con ventana grande
            {'w_unigram': 0.4, 'w_ordered': 0.3, 'w_unordered': 0.3, 'window': 12}
        ]
        
        # Calcular scores para cada configuración
        ensemble_scores = []
        for chunk in self.chunks:
            chunk_scores = []
            
            for config in mrf_configs:
                score = mrf_sd_score(
                    query, 
                    chunk, 
                    window=config['window'],
                    w_unigram=config['w_unigram'],
                    w_ordered=config['w_ordered'],
                    w_unordered=config['w_unordered']
                )
                chunk_scores.append(score)
            
            # Combinar scores del ensemble (promedio ponderado)
            ensemble_score = self._weighted_ensemble_combination(chunk_scores, query)
            ensemble_scores.append(ensemble_score)
        
        return ensemble_scores
    
    def _weighted_ensemble_combination(self, scores: List[float], query: str) -> float:
        """Combina scores del ensemble con pesos adaptativos basados en la query."""
        
        if not scores:
            return 0.0
        
        # Ponderar configuraciones basándose en características de la query
        query_tokens = query.split()
        query_length = len(query_tokens)
        
        # Pesos adaptativos para el ensemble
        if query_length <= 3:
            # Queries cortas: favorecer unigramas y bigramas ordenados
            weights = [0.4, 0.3, 0.1, 0.15, 0.05]
        elif query_length <= 6:
            # Queries medianas: balance equilibrado
            weights = [0.25, 0.25, 0.2, 0.15, 0.15]
        else:
            # Queries largas: favorecer ventanas no ordenadas y balance
            weights = [0.15, 0.2, 0.3, 0.2, 0.15]
        
        # Normalizar pesos
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Combinar scores ponderados
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        
        return weighted_sum
    
    def _get_adaptive_weights_with_learning(self, query: str) -> Tuple[float, float]:
        """Obtiene pesos adaptativos con aprendizaje de queries anteriores."""
        
        # Inicializar sistema de aprendizaje si no existe
        if not hasattr(self, '_learning_history'):
            self._learning_history = {
                'query_patterns': {},
                'success_rates': {'mrf': 0.5, 'embedding': 0.5},
                'query_count': 0
            }
        
        # Analizar patrones de la query actual
        query_features = self._extract_query_features(query)
        
        # Buscar queries similares en el historial
        similar_queries = self._find_similar_queries(query_features)
        
        # Calcular pesos basados en el historial de éxito
        if similar_queries:
            mrf_weight, embedding_weight = self._calculate_weights_from_history(similar_queries)
        else:
            # Usar pesos por defecto para queries nuevas
            mrf_weight, embedding_weight = self._calculate_adaptive_weights(len(query.split()))
        
        # Actualizar historial
        self._update_learning_history(query_features, mrf_weight, embedding_weight)
        
        return mrf_weight, embedding_weight
    
    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extrae características de la query para el aprendizaje."""
        
        tokens = query.split()
        
        return {
            'length': len(tokens),
            'avg_token_length': np.mean([len(token) for token in tokens]) if tokens else 0,
            'has_medical_terms': any(term.lower() in ['paciente', 'medicamento', 'administración', 'dosis'] for term in tokens),
            'has_question_words': any(term.lower() in ['qué', 'cómo', 'cuándo', 'dónde', 'quién'] for term in tokens),
            'has_technical_terms': any(term.lower() in ['hipoxemia', 'nebulización', 'intravenosa', 'subcutánea'] for term in tokens)
        }
    
    def _find_similar_queries(self, current_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Encuentra queries similares en el historial de aprendizaje."""
        
        similar_queries = []
        
        for pattern_key, history in self._learning_history['query_patterns'].items():
            # Recrear las características del patrón desde la clave
            pattern_features = self._recreate_features_from_pattern(pattern_key)
            similarity_score = self._calculate_pattern_similarity(current_features, pattern_features)
            if similarity_score > 0.7:  # Umbral de similitud
                similar_queries.extend(history)
        
        return similar_queries
    
    def _recreate_features_from_pattern(self, pattern_key: str) -> Dict[str, Any]:
        """Recrea las características desde la clave del patrón."""
        
        parts = pattern_key.split('_')
        
        # Mapear grupos de longitud
        length_mapping = {'short': 3, 'medium': 6, 'long': 10}
        length = length_mapping.get(parts[0], 5)
        
        # Mapear características booleanas
        has_medical = parts[1] == 'med'
        has_question = parts[2] == 'q'
        has_technical = parts[3] == 'tech'
        
        return {
            'length': length,
            'avg_token_length': 5.0,  # Valor por defecto
            'has_medical_terms': has_medical,
            'has_question_words': has_question,
            'has_technical_terms': has_technical
        }
    
    def _calculate_pattern_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calcula similitud entre dos patrones de query."""
        
        # Similitud en longitud
        length_sim = 1.0 - abs(features1['length'] - features2['length']) / max(features1['length'], features2['length'], 1)
        
        # Similitud en características booleanas
        boolean_sim = sum(1 for key in ['has_medical_terms', 'has_question_words', 'has_technical_terms'] 
                         if features1.get(key, False) == features2.get(key, False)) / 3.0
        
        # Similitud en longitud promedio de tokens
        token_sim = 1.0 - abs(features1['avg_token_length'] - features2['avg_token_length']) / max(features1['avg_token_length'], features2['avg_token_length'], 1)
        
        # Promedio ponderado
        return 0.4 * length_sim + 0.4 * boolean_sim + 0.2 * token_sim
    
    def _calculate_weights_from_history(self, similar_queries: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calcula pesos basándose en el historial de queries similares."""
        
        if not similar_queries:
            return 0.5, 0.5
        
        # Calcular tasas de éxito para cada estrategia
        mrf_successes = sum(1 for q in similar_queries if q.get('mrf_success', False))
        embedding_successes = sum(1 for q in similar_queries if q.get('embedding_success', False))
        
        total_queries = len(similar_queries)
        
        if total_queries == 0:
            return 0.5, 0.5
        
        # Calcular pesos basados en éxito relativo
        mrf_success_rate = mrf_successes / total_queries
        embedding_success_rate = embedding_successes / total_queries
        
        # Normalizar y ajustar pesos
        total_success = mrf_success_rate + embedding_success_rate
        if total_success > 0:
            mrf_weight = mrf_success_rate / total_success
            embedding_weight = embedding_success_rate / total_success
        else:
            mrf_weight = embedding_weight = 0.5
        
        # Aplicar suavizado para evitar pesos extremos
        mrf_weight = max(0.2, min(0.8, mrf_weight))
        embedding_weight = max(0.2, min(0.8, embedding_weight))
        
        # Renormalizar
        total = mrf_weight + embedding_weight
        mrf_weight /= total
        embedding_weight /= total
        
        return mrf_weight, embedding_weight
    
    def _update_learning_history(self, query_features: Dict[str, Any], mrf_weight: float, embedding_weight: float):
        """Actualiza el historial de aprendizaje con la query actual."""
        
        # Crear patrón de query
        pattern_key = self._create_pattern_key(query_features)
        
        if pattern_key not in self._learning_history['query_patterns']:
            self._learning_history['query_patterns'][pattern_key] = []
        
        # Agregar entrada al historial
        history_entry = {
            'features': query_features,
            'mrf_weight': mrf_weight,
            'embedding_weight': embedding_weight,
            'timestamp': time.time(),
            'mrf_success': False,  # Se actualizará después de la evaluación
            'embedding_success': False  # Se actualizará después de la evaluación
        }
        
        self._learning_history['query_patterns'][pattern_key].append(history_entry)
        
        # Mantener solo las últimas 10 entradas por patrón
        if len(self._learning_history['query_patterns'][pattern_key]) > 10:
            self._learning_history['query_patterns'][pattern_key] = self._learning_history['query_patterns'][pattern_key][-10:]
        
        self._learning_history['query_count'] += 1
    
    def _create_pattern_key(self, features: Dict[str, Any]) -> str:
        """Crea una clave de patrón para agrupar queries similares."""
        
        # Agrupar por rangos de longitud
        if features['length'] <= 3:
            length_group = 'short'
        elif features['length'] <= 6:
            length_group = 'medium'
        else:
            length_group = 'long'
        
        # Crear clave compuesta
        key_parts = [
            length_group,
            'med' if features['has_medical_terms'] else 'no_med',
            'q' if features['has_question_words'] else 'no_q',
            'tech' if features['has_technical_terms'] else 'no_tech'
        ]
        
        return '_'.join(key_parts)
    
    def update_success_metrics(self, query: str, strategy: str, success: bool):
        """Actualiza métricas de éxito para el aprendizaje."""
        
        if not hasattr(self, '_learning_history'):
            return
        
        # Encontrar la entrada más reciente para esta query
        query_features = self._extract_query_features(query)
        pattern_key = self._create_pattern_key(query_features)
        
        if pattern_key in self._learning_history['query_patterns']:
            # Actualizar la entrada más reciente
            recent_entries = self._learning_history['query_patterns'][pattern_key]
            if recent_entries:
                latest_entry = recent_entries[-1]
                if strategy == 'mrf':
                    latest_entry['mrf_success'] = success
                elif strategy == 'embedding':
                    latest_entry['embedding_success'] = success
    
    def load_benchmark_queries(self) -> List[Dict[str, str]]:
        """Carga las queries del archivo de benchmark."""
        with open(self.benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def evaluate_strategy(self, strategy_name: str, search_function, 
                         queries: List[Dict[str, str]], top_k: int = 5) -> Dict[str, Any]:
        """Evalúa una estrategia específica."""
        print(f"\n=== Evaluando estrategia: {strategy_name} ===")
        
        results = {
            'strategy': strategy_name,
            'total_queries': len(queries),
            'top1_correct': 0,
            'top5_correct': 0,
            'query_results': []
        }
        
        for i, query_data in enumerate(queries):
            query = query_data['query']
            expected_doc = query_data['document_expected']
            
            print(f"Query {i+1}/{len(queries)}: {query[:60]}...")
            
            # Ejecutar búsqueda
            start_time = time.time()
            top_results = search_function(query, top_k)
            search_time = time.time() - start_time
            
            # Evaluar resultados
            top1_correct = False
            top5_correct = False
            
            # Normalizar nombres de documentos para comparación
            expected_doc_normalized = expected_doc.replace('_limpio.txt', '_limpio')
            
            for result in top_results:
                if result.chunk.doc_id == expected_doc_normalized:
                    if result.rank == 1:
                        top1_correct = True
                    if result.rank <= 5:
                        top5_correct = True
                    break
            
            if top1_correct:
                results['top1_correct'] += 1
            if top5_correct:
                results['top5_correct'] += 1
                
            # Guardar resultado detallado
            query_result = {
                'query': query,
                'expected_doc': expected_doc,
                'top1_correct': top1_correct,
                'top5_correct': top5_correct,
                'search_time': search_time,
                'top_results': [
                    {
                        'rank': r.rank,
                        'doc_id': r.chunk.doc_id,
                        'score': float(r.total_score),
                        'text_preview': r.chunk.text[:100] + "..."
                    }
                    for r in top_results
                ]
            }
            results['query_results'].append(query_result)
        
        # Calcular métricas
        results['top1_accuracy'] = results['top1_correct'] / results['total_queries']
        results['top5_accuracy'] = results['top5_correct'] / results['total_queries']
        
        print(f"Top1 Accuracy: {results['top1_accuracy']:.3f} ({results['top1_correct']}/{results['total_queries']})")
        print(f"Top5 Accuracy: {results['top5_accuracy']:.3f} ({results['top5_correct']}/{results['total_queries']})")
        
        return results
    
    def run_full_benchmark(self, top_k: int = 5):
        """Ejecuta el benchmark completo con las tres estrategias."""
        print("=== INICIANDO BENCHMARK COMPLETO DE PNTs ===")
        
        # 1. Cargar modelo
        self.load_model()
        
        # 2. Cargar documentos PNTs
        self.load_pnts_documents()
        
        # 3. Calcular embeddings
        self.compute_embeddings()
        
        # 4. Calcular estadísticas del corpus
        self.compute_corpus_statistics()
        
        # 5. Cargar queries de benchmark
        queries = self.load_benchmark_queries()
        print(f"Queries cargadas: {len(queries)}")
        
        # 6. Evaluar las tres estrategias
        results = {}
        
        # Estrategia 1: Solo embeddings (baseline)
        results['embeddings_only'] = self.evaluate_strategy(
            "Solo Embeddings (Baseline)",
            self.search_embeddings_only,
            queries,
            top_k
        )
        
        # Estrategia 2: Solo MRF
        results['mrf_only'] = self.evaluate_strategy(
            "Solo MRF",
            self.search_mrf_only,
            queries,
            top_k
        )
        
        # Estrategia 3: MRF + Embeddings
        results['mrf_plus_embeddings'] = self.evaluate_strategy(
            "MRF + Embeddings",
            self.search_mrf_plus_embeddings,
            queries,
            top_k
        )
        
        # Estrategia 4: MRF + Embeddings con Pesos Adaptativos (Mejora 1)
        results['mrf_plus_embeddings_adaptive'] = self.evaluate_strategy(
            "MRF + Embeddings (Pesos Adaptativos)",
            self.search_mrf_plus_embeddings_adaptive,
            queries,
            top_k
        )
        
        # Estrategia 5: MRF + Embeddings con Ventanas Optimizadas (Mejora 2)
        results['mrf_plus_embeddings_optimized_windows'] = self.evaluate_strategy(
            "MRF + Embeddings (Ventanas Optimizadas)",
            self.search_mrf_plus_embeddings_optimized_windows,
            queries,
            top_k
        )
        
        # Estrategia 6: MRF + Embeddings con Normalización Inteligente (Mejora 3)
        results['mrf_plus_embeddings_intelligent_normalization'] = self.evaluate_strategy(
            "MRF + Embeddings (Normalización Inteligente)",
            self.search_mrf_plus_embeddings_intelligent_normalization,
            queries,
            top_k
        )
        
        # Estrategia 7: MRF + Embeddings con Ensemble (Mejora 4)
        results['mrf_plus_embeddings_ensemble'] = self.evaluate_strategy(
            "MRF + Embeddings (Ensemble)",
            self.search_mrf_plus_embeddings_ensemble,
            queries,
            top_k
        )
        
        # Estrategia 8: MRF + Embeddings con Sistema de Aprendizaje Adaptativo (Mejora 5)
        results['mrf_plus_embeddings_adaptive_learning'] = self.evaluate_strategy(
            "MRF + Embeddings (Aprendizaje Adaptativo)",
            self.search_mrf_plus_embeddings_adaptive_learning,
            queries,
            top_k
        )
        
        # 7. Mostrar resumen comparativo
        self._print_comparative_summary(results)
        
        # 8. Guardar resultados
        self._save_results(results)
        
        return results
    
    def _print_comparative_summary(self, results: Dict[str, Any]):
        """Imprime un resumen comparativo de todas las estrategias."""
        print("\n" + "="*80)
        print("RESUMEN COMPARATIVO DE ESTRATEGIAS")
        print("="*80)
        
        strategies = ['embeddings_only', 'mrf_only', 'mrf_plus_embeddings', 'mrf_plus_embeddings_adaptive', 'mrf_plus_embeddings_optimized_windows', 'mrf_plus_embeddings_intelligent_normalization', 'mrf_plus_embeddings_ensemble', 'mrf_plus_embeddings_adaptive_learning']
        strategy_names = ['Solo Embeddings', 'Solo MRF', 'MRF + Embeddings', 'MRF + Embeddings (Pesos Adaptativos)', 'MRF + Embeddings (Ventanas Optimizadas)', 'MRF + Embeddings (Normalización Inteligente)', 'MRF + Embeddings (Ensemble)', 'MRF + Embeddings (Aprendizaje Adaptativo)']
        
        print(f"{'Estrategia':<20} {'Top1 Acc':<10} {'Top5 Acc':<10} {'Top1':<8} {'Top5':<8}")
        print("-" * 80)
        
        for strategy, name in zip(strategies, strategy_names):
            r = results[strategy]
            print(f"{name:<20} {r['top1_accuracy']:<10.3f} {r['top5_accuracy']:<10.3f} "
                  f"{r['top1_correct']:<8} {r['top5_correct']:<8}")
        
        print("\n" + "="*80)
        
        # Encontrar la mejor estrategia
        best_top1 = max(results.values(), key=lambda x: x['top1_accuracy'])
        best_top5 = max(results.values(), key=lambda x: x['top5_accuracy'])
        
        print(f"MEJOR TOP1: {best_top1['strategy']} ({best_top1['top1_accuracy']:.3f})")
        print(f"MEJOR TOP5: {best_top5['strategy']} ({best_top5['top5_accuracy']:.3f})")
        
        # Comparar con baseline
        baseline = results['embeddings_only']
        for strategy in ['mrf_only', 'mrf_plus_embeddings', 'mrf_plus_embeddings_adaptive', 'mrf_plus_embeddings_optimized_windows', 'mrf_plus_embeddings_intelligent_normalization', 'mrf_plus_embeddings_ensemble', 'mrf_plus_embeddings_adaptive_learning']:
            r = results[strategy]
            
            # Evitar división por cero
            if baseline['top1_accuracy'] > 0:
                top1_improvement = ((r['top1_accuracy'] - baseline['top1_accuracy']) / 
                                   baseline['top1_accuracy']) * 100
            else:
                top1_improvement = float('inf') if r['top1_accuracy'] > 0 else 0.0
                
            if baseline['top5_accuracy'] > 0:
                top5_improvement = ((r['top5_accuracy'] - baseline['top5_accuracy']) / 
                                   baseline['top5_accuracy']) * 100
            else:
                top5_improvement = float('inf') if r['top5_accuracy'] > 0 else 0.0
            
            print(f"\n{strategy.upper()}:")
            if top1_improvement == float('inf'):
                print(f"  Top1: +∞% vs baseline (de 0% a {r['top1_accuracy']:.1%})")
            else:
                print(f"  Top1: {top1_improvement:+.1f}% vs baseline")
                
            if top5_improvement == float('inf'):
                print(f"  Top5: +∞% vs baseline (de 0% a {r['top5_accuracy']:.1%})")
            else:
                print(f"  Top5: {top5_improvement:+.1f}% vs baseline")
    
    def _save_results(self, results: Dict[str, Any]):
        """Guarda los resultados en un archivo JSON."""
        output_file = f"benchmark_results_{int(time.time())}.json"
        
        # Convertir numpy arrays a listas para serialización JSON
        serializable_results = {}
        for strategy, result in results.items():
            serializable_results[strategy] = {
                'strategy': result['strategy'],
                'total_queries': result['total_queries'],
                'top1_correct': result['top1_correct'],
                'top5_correct': result['top5_correct'],
                'top1_accuracy': float(result['top1_accuracy']),
                'top5_accuracy': float(result['top5_accuracy']),
                'query_results': result['query_results']
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResultados guardados en: {output_file}")


def main():
    """Función principal del script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark de estrategias de re-ranking para PNTs')
    parser.add_argument('--pnts-dir', default='PNTs', help='Directorio con documentos PNTs')
    parser.add_argument('--benchmark-file', default='benchmark/preguntas_con_docs_es.json', 
                       help='Archivo de benchmark con queries')
    parser.add_argument('--top-k', type=int, default=5, help='Número de resultados a evaluar')
    
    args = parser.parse_args()
    
    # Crear y ejecutar benchmark
    benchmark = PNTsBenchmark(args.pnts_dir, args.benchmark_file)
    
    try:
        results = benchmark.run_full_benchmark(args.top_k)
        print("\n¡Benchmark completado exitosamente!")
        
    except Exception as e:
        print(f"Error durante el benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
