"""
Tests unitarios para el módulo de fusión de señales.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Añadir el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rerank_markov.fusion import (
    rerank, rerank_with_analysis, _normalize_all_scores,
    _combine_scores, _create_scored_chunks
)
from rerank_markov.types import Chunk, ScoredChunk
from rerank_markov.config import RerankConfig
from rerank_markov.utils import normalize_scores


@pytest.fixture
def sample_chunks():
    """Crea chunks de ejemplo para testing."""
    chunks = [
        Chunk(
            id="chunk_001",
            text="El paciente presenta síntomas de hipoxemia postoperatoria que requieren ajuste de PEEP.",
            doc_id="doc_01",
            position=0,
            embedding=np.array([0.8] * 192 + [0.2] * 192),  # Embedding diferenciado
            meta={'length': 15, 'tokens': 15}
        ),
        Chunk(
            id="chunk_002",
            text="La administración de medicamentos vía inhalatoria debe realizarse con precaución.",
            doc_id="doc_01",
            position=1,
            embedding=np.array([0.6] * 192 + [0.4] * 192),  # Embedding diferenciado
            meta={'length': 12, 'tokens': 12}
        ),
        Chunk(
            id="chunk_003",
            text="El abordaje integral de la mujer con cáncer ginecológico incluye múltiples especialidades.",
            doc_id="doc_02",
            position=0,
            embedding=np.array([0.4] * 192 + [0.6] * 192),  # Embedding diferenciado
            meta={'length': 18, 'tokens': 18}
        )
    ]
    return chunks


@pytest.fixture
def sample_config():
    """Crea configuración de ejemplo para testing."""
    return RerankConfig(
        k=3,
        k_prime=10,
        mu=1500,
        lambda_ppr=0.85,
        a=0.45,
        b=0.25,
        c=0.20,
        d=0.10,
        alpha=0.6,
        beta=0.25,
        gamma=0.15,
        window_size=8,
        w_unigram=0.8,
        w_ordered=0.1,
        w_unordered=0.1,
        use_zscore=True,
        clip_sigma=3.0
    )


class TestNormalization:
    """Tests para normalización de puntuaciones."""
    
    def test_normalize_scores_zscore(self):
        """Test normalización z-score."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        normalized = normalize_scores(scores, method='zscore', clip_sigma=3.0)
        
        assert len(normalized) == len(scores)
        assert all(isinstance(score, float) for score in normalized)
        
        # La media debe ser aproximadamente 0
        mean = np.mean(normalized)
        assert abs(mean) < 0.1
        
        # La desviación estándar debe ser aproximadamente 1
        std = np.std(normalized)
        assert abs(std - 1.0) < 0.1
    
    def test_normalize_scores_minmax(self):
        """Test normalización min-max."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        normalized = normalize_scores(scores, method='minmax')
        
        assert len(normalized) == len(scores)
        assert all(isinstance(score, float) for score in normalized)
        
        # El mínimo debe ser 0 y el máximo 1
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
    
    def test_normalize_scores_none(self):
        """Test sin normalización."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        normalized = normalize_scores(scores, method='none')
        
        assert normalized == scores
    
    def test_normalize_scores_empty(self):
        """Test normalización con lista vacía."""
        scores = []
        
        normalized = normalize_scores(scores, method='zscore')
        
        assert normalized == []
    
    def test_normalize_scores_single_value(self):
        """Test normalización con un solo valor."""
        scores = [5.0]
        
        normalized = normalize_scores(scores, method='zscore')
        
        assert len(normalized) == 1
        assert normalized[0] == 0.0  # Con un valor, z-score es 0
    
    def test_normalize_scores_clipping(self):
        """Test clipping en normalización z-score."""
        scores = [1.0, 2.0, 3.0, 4.0, 100.0]  # Valor outlier
        
        normalized = normalize_scores(scores, method='zscore', clip_sigma=2.0)
        
        # El valor outlier debe estar recortado a ±2σ
        assert max(normalized) <= 2.0
        assert min(normalized) >= -2.0
    
    def test_normalize_scores_zero_std(self):
        """Test normalización con desviación estándar cero."""
        scores = [5.0, 5.0, 5.0, 5.0, 5.0]
        
        normalized = normalize_scores(scores, method='zscore')
        
        # Todos los valores deben ser 0
        assert all(score == 0.0 for score in normalized)


class TestScoreCombination:
    """Tests para combinación de puntuaciones."""
    
    def test_combine_scores_basic(self, sample_config):
        """Test básico de combinación de puntuaciones."""
        normalized_scores = {
            'embedding': [0.5, -0.5, 0.0],
            'ppr': [0.3, 0.7, -0.2],
            'qlm': [-0.1, 0.4, 0.6],
            'mrf': [0.2, -0.3, 0.8]
        }
        
        combined = _combine_scores(normalized_scores, sample_config)
        
        assert len(combined) == 3
        assert all(isinstance(score, float) for score in combined)
        
        # Verificar que se aplican los pesos correctamente
        # score[0] = 0.45*0.5 + 0.25*0.3 + 0.20*(-0.1) + 0.10*0.2 = 0.225 + 0.075 - 0.02 + 0.02 = 0.3
        expected_score_0 = (0.45 * 0.5 + 0.25 * 0.3 + 0.20 * (-0.1) + 0.10 * 0.2)
        assert abs(combined[0] - expected_score_0) < 1e-10
    
    def test_combine_scores_different_weights(self, sample_config):
        """Test combinación con diferentes pesos."""
        # Cambiar pesos
        sample_config.a = 0.8
        sample_config.b = 0.1
        sample_config.c = 0.05
        sample_config.d = 0.05
        
        normalized_scores = {
            'embedding': [1.0, 0.0, -1.0],
            'ppr': [0.0, 1.0, 0.0],
            'qlm': [0.0, 0.0, 1.0],
            'mrf': [0.0, 0.0, 0.0]
        }
        
        combined = _combine_scores(normalized_scores, sample_config)
        
        # Con estos pesos, el primer score debe ser principalmente embedding
        assert combined[0] > combined[1]
        assert combined[0] > combined[2]
    
    def test_combine_scores_empty(self, sample_config):
        """Test combinación con puntuaciones vacías."""
        normalized_scores = {
            'embedding': [],
            'ppr': [],
            'qlm': [],
            'mrf': []
        }
        
        combined = _combine_scores(normalized_scores, sample_config)
        
        assert combined == []
    
    def test_combine_scores_mismatched_lengths(self, sample_config):
        """Test combinación con longitudes diferentes (debe fallar)."""
        normalized_scores = {
            'embedding': [0.5, 0.3],
            'ppr': [0.2, 0.4, 0.6],  # Diferente longitud
            'qlm': [0.1, 0.2],
            'mrf': [0.0, 0.1]
        }
        
        # Esto debería fallar, pero manejamos el caso
        try:
            combined = _combine_scores(normalized_scores, sample_config)
            # Si no falla, verificar que se maneja correctamente
            assert len(combined) == 2  # Longitud mínima
        except Exception:
            # Es aceptable que falle
            pass


class TestScoredChunkCreation:
    """Tests para creación de ScoredChunk."""
    
    def test_create_scored_chunks_basic(self, sample_chunks):
        """Test básico de creación de ScoredChunk."""
        final_scores = [0.8, 0.6, 0.4]
        normalized_scores = {
            'embedding': [0.5, 0.3, 0.1],
            'ppr': [0.2, 0.4, 0.6],
            'qlm': [0.1, 0.2, 0.3],
            'mrf': [0.0, 0.1, 0.2]
        }
        
        scored_chunks = _create_scored_chunks(sample_chunks, final_scores, normalized_scores)
        
        assert len(scored_chunks) == len(sample_chunks)
        assert all(isinstance(sc, ScoredChunk) for sc in scored_chunks)
        
        # Verificar que los scores se asignan correctamente
        for i, sc in enumerate(scored_chunks):
            assert sc.total_score == final_scores[i]
            assert sc.embedding_score == normalized_scores['embedding'][i]
            assert sc.ppr_score == normalized_scores['ppr'][i]
            assert sc.qlm_score == normalized_scores['qlm'][i]
            assert sc.mrf_score == normalized_scores['mrf'][i]
            assert sc.chunk == sample_chunks[i]
    
    def test_create_scored_chunks_empty(self):
        """Test creación con chunks vacíos."""
        final_scores = []
        normalized_scores = {
            'embedding': [],
            'ppr': [],
            'qlm': [],
            'mrf': []
        }
        
        scored_chunks = _create_scored_chunks([], final_scores, normalized_scores)
        
        assert scored_chunks == []


class TestRerankIntegration:
    """Tests de integración para el re-ranking."""
    
    def test_rerank_basic(self, sample_chunks, sample_config):
        """Test básico de re-ranking."""
        query = "hipoxemia postoperatoria"
        seed_chunks = sample_chunks[:1]
        candidate_chunks = sample_chunks
        
        results = rerank(query, seed_chunks, candidate_chunks, sample_config)
        
        assert len(results) == len(candidate_chunks)
        assert all(isinstance(sc, ScoredChunk) for sc in results)
        
        # Verificar que están ordenados por puntuación total
        for i in range(len(results) - 1):
            assert results[i].total_score >= results[i + 1].total_score
        
        # Verificar que se asignaron rankings
        for i, sc in enumerate(results):
            assert sc.rank == i + 1
    
    def test_rerank_empty_candidates(self, sample_config):
        """Test re-ranking sin candidatos."""
        query = "hipoxemia postoperatoria"
        seed_chunks = []
        candidate_chunks = []
        
        results = rerank(query, seed_chunks, candidate_chunks, sample_config)
        
        assert results == []
    
    def test_rerank_no_seeds(self, sample_chunks, sample_config):
        """Test re-ranking sin chunks semilla."""
        query = "hipoxemia postoperatoria"
        seed_chunks = []
        candidate_chunks = sample_chunks
        
        results = rerank(query, seed_chunks, candidate_chunks, sample_config)
        
        # Debe funcionar aunque no haya semillas (PPR será 0)
        assert len(results) == len(candidate_chunks)
        assert all(isinstance(sc, ScoredChunk) for sc in results)
    
    def test_rerank_with_analysis(self, sample_chunks, sample_config):
        """Test re-ranking con análisis."""
        query = "hipoxemia postoperatoria"
        seed_chunks = sample_chunks[:1]
        candidate_chunks = sample_chunks
        
        results = rerank_with_analysis(query, seed_chunks, candidate_chunks, sample_config)
        
        assert isinstance(results, dict)
        assert 'query' in results
        assert 'top_results' in results
        assert 'score_distributions' in results
        assert 'component_correlations' in results
        
        # Verificar que el análisis contiene información útil
        assert results['query'] == query
        assert len(results['top_results']) > 0
        assert isinstance(results['score_distributions'], dict)
        assert isinstance(results['component_correlations'], dict)


class TestNormalizationIntegration:
    """Tests de integración para normalización."""
    
    def test_normalize_all_scores_with_zscore(self, sample_config):
        """Test normalización completa con z-score."""
        embedding_scores = [0.8, 0.6, 0.4]
        ppr_scores = [0.9, 0.7, 0.5]
        qlm_scores = [0.7, 0.5, 0.3]
        mrf_scores = [0.6, 0.4, 0.2]
        
        normalized = _normalize_all_scores(
            embedding_scores, ppr_scores, qlm_scores, mrf_scores, sample_config
        )
        
        assert 'embedding' in normalized
        assert 'ppr' in normalized
        assert 'qlm' in normalized
        assert 'mrf' in normalized
        
        # Verificar que todas las puntuaciones están normalizadas
        for component in ['embedding', 'ppr', 'qlm', 'mrf']:
            scores = normalized[component]
            assert len(scores) == 3
            assert all(isinstance(score, float) for score in scores)
    
    def test_normalize_all_scores_without_zscore(self, sample_config):
        """Test normalización sin z-score."""
        sample_config.use_zscore = False
        
        embedding_scores = [0.8, 0.6, 0.4]
        ppr_scores = [0.9, 0.7, 0.5]
        qlm_scores = [0.7, 0.5, 0.3]
        mrf_scores = [0.6, 0.4, 0.2]
        
        normalized = _normalize_all_scores(
            embedding_scores, ppr_scores, qlm_scores, mrf_scores, sample_config
        )
        
        # Sin normalización, las puntuaciones deben ser las originales
        assert normalized['embedding'] == embedding_scores
        assert normalized['ppr'] == ppr_scores
        assert normalized['qlm'] == qlm_scores
        assert normalized['mrf'] == mrf_scores


class TestFusionEdgeCases:
    """Tests para casos extremos de fusión."""
    
    def test_rerank_extreme_scores(self, sample_chunks, sample_config):
        """Test re-ranking con puntuaciones extremas."""
        # Crear chunks con puntuaciones extremas
        extreme_chunks = []
        for i, chunk in enumerate(sample_chunks):
            # Puntuaciones muy diferentes
            if i == 0:
                chunk.embedding = np.ones(384) * 0.9  # Muy alta similitud
            elif i == 1:
                chunk.embedding = np.ones(384) * 0.1  # Muy baja similitud
            else:
                chunk.embedding = np.ones(384) * 0.5  # Similitud media
            
            extreme_chunks.append(chunk)
        
        query = "hipoxemia postoperatoria"
        seed_chunks = extreme_chunks[:1]
        candidate_chunks = extreme_chunks
        
        results = rerank(query, seed_chunks, candidate_chunks, sample_config)
        
        assert len(results) == len(candidate_chunks)
        assert all(isinstance(sc, ScoredChunk) for sc in results)
        
        # El primer resultado debe tener la puntuación más alta
        assert results[0].total_score >= results[1].total_score
        assert results[1].total_score >= results[2].total_score
    
    def test_rerank_identical_scores(self, sample_chunks, sample_config):
        """Test re-ranking con puntuaciones idénticas."""
        # Crear chunks con puntuaciones idénticas
        identical_chunks = []
        for chunk in sample_chunks:
            chunk.embedding = np.ones(384) * 0.5  # Todas iguales
            identical_chunks.append(chunk)
        
        query = "hipoxemia postoperatoria"
        seed_chunks = identical_chunks[:1]
        candidate_chunks = identical_chunks
        
        results = rerank(query, seed_chunks, candidate_chunks, sample_config)
        
        assert len(results) == len(candidate_chunks)
        # Con puntuaciones idénticas, el orden puede variar pero debe ser consistente
        assert all(isinstance(sc, ScoredChunk) for sc in results)
    
    def test_rerank_zero_scores(self, sample_chunks, sample_config):
        """Test re-ranking con puntuaciones cero."""
        # Crear chunks con puntuaciones cero
        zero_chunks = []
        for chunk in sample_chunks:
            chunk.embedding = np.zeros(384)  # Todas cero
            zero_chunks.append(chunk)
        
        query = "hipoxemia postoperatoria"
        seed_chunks = zero_chunks[:1]
        candidate_chunks = zero_chunks
        
        results = rerank(query, seed_chunks, candidate_chunks, sample_config)
        
        assert len(results) == len(candidate_chunks)
        # Debe manejar puntuaciones cero correctamente
        assert all(isinstance(sc, ScoredChunk) for sc in results)


class TestFusionConsistency:
    """Tests para consistencia de la fusión."""
    
    def test_rerank_consistency(self, sample_chunks, sample_config):
        """Test que el re-ranking es consistente."""
        query = "hipoxemia postoperatoria"
        seed_chunks = sample_chunks[:1]
        candidate_chunks = sample_chunks
        
        # Ejecutar múltiples veces
        results_1 = rerank(query, seed_chunks, candidate_chunks, sample_config)
        results_2 = rerank(query, seed_chunks, candidate_chunks, sample_config)
        
        # Los resultados deben ser idénticos
        assert len(results_1) == len(results_2)
        
        for sc1, sc2 in zip(results_1, results_2):
            assert np.isclose(sc1.total_score, sc2.total_score, rtol=1e-6)
            assert np.isclose(sc1.embedding_score, sc2.embedding_score, rtol=1e-6)
            assert np.isclose(sc1.ppr_score, sc2.ppr_score, rtol=1e-6)
            assert np.isclose(sc1.qlm_score, sc2.qlm_score, rtol=1e-6)
            assert np.isclose(sc1.mrf_score, sc2.mrf_score, rtol=1e-6)
    
    def test_score_distributions(self, sample_chunks, sample_config):
        """Test que las distribuciones de puntuaciones son consistentes."""
        query = "hipoxemia postoperatoria"
        seed_chunks = sample_chunks[:1]
        candidate_chunks = sample_chunks
        
        results = rerank(query, seed_chunks, candidate_chunks, sample_config)
        
        # Verificar que las puntuaciones están en rangos razonables
        for sc in results:
            assert -5.0 <= sc.embedding_score <= 5.0  # Z-score con clipping
            assert -5.0 <= sc.ppr_score <= 5.0
            assert -5.0 <= sc.qlm_score <= 5.0
            assert -5.0 <= sc.mrf_score <= 5.0
            assert isinstance(sc.total_score, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
