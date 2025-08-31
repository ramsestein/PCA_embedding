"""
Tests unitarios para Markov Random Field (MRF) con dependencias secuenciales.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Añadir el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rerank_markov.mrf import (
    mrf_sd_score, _compute_unigram_score, _compute_ordered_bigram_score,
    _compute_unordered_bigram_score, mrf_enhanced_score
)
from rerank_markov.types import Chunk


@pytest.fixture
def sample_chunks():
    """Crea chunks de ejemplo para testing."""
    chunks = [
        Chunk(
            id="chunk_001",
            text="El paciente presenta síntomas de hipoxemia postoperatoria que requieren ajuste de PEEP.",
            doc_id="doc_01",
            position=0,
            embedding=np.random.rand(384),
            meta={'length': 15, 'tokens': 15}
        ),
        Chunk(
            id="chunk_002",
            text="La administración de medicamentos vía inhalatoria debe realizarse con precaución.",
            doc_id="doc_01",
            position=1,
            embedding=np.random.rand(384),
            meta={'length': 12, 'tokens': 12}
        ),
        Chunk(
            id="chunk_003",
            text="El abordaje integral de la mujer con cáncer ginecológico incluye múltiples especialidades.",
            doc_id="doc_02",
            position=0,
            embedding=np.random.rand(384),
            meta={'length': 18, 'tokens': 18}
        )
    ]
    return chunks


class TestMRFUnigram:
    """Tests para características unigram."""
    
    def test_unigram_basic(self, sample_chunks):
        """Test básico de puntuación unigram."""
        query_tokens = ["hipoxemia", "postoperatoria"]
        chunk_tokens = ["el", "paciente", "presenta", "síntomas", "de", "hipoxemia", "postoperatoria"]
        
        score = _compute_unigram_score(query_tokens, chunk_tokens)
        
        assert isinstance(score, float)
        assert score > 0  # Debe ser positivo ya que ambos términos están presentes
        assert score <= 1.0  # Debe estar normalizado
    
    def test_unigram_no_matches(self, sample_chunks):
        """Test unigram cuando no hay coincidencias."""
        query_tokens = ["diabetes", "mellitus"]
        chunk_tokens = ["el", "paciente", "presenta", "síntomas"]
        
        score = _compute_unigram_score(query_tokens, chunk_tokens)
        
        assert score == 0.0  # Debe ser 0 sin coincidencias
    
    def test_unigram_partial_matches(self, sample_chunks):
        """Test unigram con coincidencias parciales."""
        query_tokens = ["hipoxemia", "diabetes", "postoperatoria"]
        chunk_tokens = ["el", "paciente", "presenta", "hipoxemia", "postoperatoria"]
        
        score = _compute_unigram_score(query_tokens, chunk_tokens)
        
        assert 0 < score < 1.0  # Debe ser parcial (2 de 3 términos)
        # Debe ser aproximadamente 2/3 = 0.67
        assert abs(score - 0.67) < 0.1
    
    def test_unigram_empty_query(self, sample_chunks):
        """Test unigram con query vacía."""
        query_tokens = []
        chunk_tokens = ["el", "paciente", "presenta"]
        
        score = _compute_unigram_score(query_tokens, chunk_tokens)
        
        assert score == 0.0
    
    def test_unigram_term_frequency_bonus(self, sample_chunks):
        """Test que la frecuencia de términos afecta la puntuación."""
        query_tokens = ["el", "paciente"]
        chunk_tokens = ["el", "el", "el", "paciente", "presenta", "síntomas"]
        
        score = _compute_unigram_score(query_tokens, chunk_tokens)
        
        assert score > 0.5  # Debe ser alto por alta frecuencia de "el"
    
    def test_unigram_single_token(self, sample_chunks):
        """Test unigram con un solo token."""
        query_tokens = ["hipoxemia"]
        chunk_tokens = ["el", "paciente", "presenta", "hipoxemia"]
        
        score = _compute_unigram_score(query_tokens, chunk_tokens)
        
        assert score == 1.0  # Debe ser 1.0 (100% de cobertura)


class TestMRFOrderedBigram:
    """Tests para bigramas ordenados."""
    
    def test_ordered_bigram_basic(self, sample_chunks):
        """Test básico de bigramas ordenados."""
        query_tokens = ["hipoxemia", "postoperatoria"]
        chunk_tokens = ["el", "paciente", "presenta", "hipoxemia", "postoperatoria", "que"]
        
        score = _compute_ordered_bigram_score(query_tokens, chunk_tokens)
        
        assert isinstance(score, float)
        assert score == 1.0  # Debe ser 1.0 ya que el bigrama está presente en orden
    
    def test_ordered_bigram_no_matches(self, sample_chunks):
        """Test bigramas ordenados sin coincidencias."""
        query_tokens = ["hipoxemia", "postoperatoria"]
        chunk_tokens = ["el", "paciente", "presenta", "síntomas"]
        
        score = _compute_ordered_bigram_score(query_tokens, chunk_tokens)
        
        assert score == 0.0  # Debe ser 0 sin bigramas coincidentes
    
    def test_ordered_bigram_partial_matches(self, sample_chunks):
        """Test bigramas ordenados con coincidencias parciales."""
        query_tokens = ["hipoxemia", "postoperatoria", "que", "requieren"]
        chunk_tokens = ["el", "paciente", "presenta", "hipoxemia", "postoperatoria", "que"]
        
        score = _compute_ordered_bigram_score(query_tokens, chunk_tokens)
        
        # Debe encontrar 2 de 3 bigramas: (hipoxemia, postoperatoria) y (postoperatoria, que)
        assert score == 2/3
    
    def test_ordered_bigram_single_token(self, sample_chunks):
        """Test bigramas ordenados con query de un solo token."""
        query_tokens = ["hipoxemia"]
        chunk_tokens = ["el", "paciente", "presenta", "hipoxemia"]
        
        score = _compute_ordered_bigram_score(query_tokens, chunk_tokens)
        
        assert score == 0.0  # No hay bigramas con un solo token
    
    def test_ordered_bigram_reversed_order(self, sample_chunks):
        """Test que el orden importa para bigramas."""
        query_tokens = ["hipoxemia", "postoperatoria"]
        chunk_tokens = ["el", "paciente", "postoperatoria", "hipoxemia"]  # Orden invertido
        
        score = _compute_ordered_bigram_score(query_tokens, chunk_tokens)
        
        assert score == 0.0  # Debe ser 0 porque el orden está invertido


class TestMRFUnorderedBigram:
    """Tests para bigramas no ordenados."""
    
    def test_unordered_bigram_basic(self, sample_chunks):
        """Test básico de bigramas no ordenados."""
        query_tokens = ["hipoxemia", "postoperatoria"]
        chunk_tokens = ["el", "paciente", "presenta", "hipoxemia", "postoperatoria", "que"]
        
        score = _compute_unordered_bigram_score(query_tokens, chunk_tokens, window=8)
        
        assert isinstance(score, float)
        assert score == 1.0  # Debe ser 1.0 ya que ambos términos están en la ventana
    
    def test_unordered_bigram_within_window(self, sample_chunks):
        """Test bigramas no ordenados dentro de la ventana."""
        query_tokens = ["hipoxemia", "postoperatoria"]
        chunk_tokens = ["hipoxemia", "el", "paciente", "presenta", "síntomas", "de", "postoperatoria"]
        
        score = _compute_unordered_bigram_score(query_tokens, chunk_tokens, window=8)
        
        assert score == 1.0  # Debe ser 1.0 ya que están dentro de la ventana
    
    def test_unordered_bigram_outside_window(self, sample_chunks):
        """Test bigramas no ordenados fuera de la ventana."""
        query_tokens = ["hipoxemia", "postoperatoria"]
        chunk_tokens = ["hipoxemia", "a", "b", "c", "d", "e", "f", "g", "h", "postoperatoria"]
        
        score = _compute_unordered_bigram_score(query_tokens, chunk_tokens, window=8)
        
        assert score == 0.0  # Debe ser 0 ya que están fuera de la ventana
    
    def test_unordered_bigram_different_windows(self, sample_chunks):
        """Test bigramas no ordenados con diferentes tamaños de ventana."""
        query_tokens = ["hipoxemia", "postoperatoria"]
        chunk_tokens = ["hipoxemia", "a", "b", "c", "d", "e", "f", "g", "h", "postoperatoria"]
        
        score_window_4 = _compute_unordered_bigram_score(query_tokens, chunk_tokens, window=4)
        score_window_8 = _compute_unordered_bigram_score(query_tokens, chunk_tokens, window=8)
        
        assert score_window_4 == 0.0  # Ventana pequeña
        assert score_window_8 == 0.0  # Ventana grande pero términos muy separados
    
    def test_unordered_bigram_single_token(self, sample_chunks):
        """Test bigramas no ordenados con query de un solo token."""
        query_tokens = ["hipoxemia"]
        chunk_tokens = ["el", "paciente", "presenta", "hipoxemia"]
        
        score = _compute_unordered_bigram_score(query_tokens, chunk_tokens, window=8)
        
        assert score == 0.0  # No hay bigramas con un solo token


class TestMRFCombined:
    """Tests para la puntuación MRF combinada."""
    
    def test_mrf_sd_basic(self, sample_chunks):
        """Test básico de MRF SD."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]  # Contiene ambos términos
        
        score = mrf_sd_score(query, chunk)
        
        assert isinstance(score, float)
        assert score > 0  # Debe ser positivo
    
    def test_mrf_sd_weight_impact(self, sample_chunks):
        """Test que los pesos afectan la puntuación final."""
        query = "hipoxemia postoperatoria"
        
        # Crear un chunk donde solo algunos componentes MRF den scores altos
        # Este chunk tiene "hipoxemia" pero no "postoperatoria" en orden consecutivo
        test_chunk = Chunk(
            id="test_weight_impact",
            text="El paciente presenta hipoxemia severa. La complicación postoperatoria es rara.",
            doc_id="doc_test",
            position=0,
            embedding=np.ones(384) * 0.5
        )
        
        # Peso alto en unigram (debe dar score alto porque ambos términos están presentes)
        score_unigram_heavy = mrf_sd_score(
            query, test_chunk, w_unigram=0.9, w_ordered=0.05, w_unordered=0.05
        )
        
        # Peso alto en bigramas ordenados (debe dar score bajo porque no están consecutivos)
        score_ordered_heavy = mrf_sd_score(
            query, test_chunk, w_unigram=0.1, w_ordered=0.8, w_unordered=0.1
        )
        
        # Peso alto en bigramas no ordenados (debe dar score medio)
        score_unordered_heavy = mrf_sd_score(
            query, test_chunk, w_unigram=0.1, w_ordered=0.1, w_unordered=0.8
        )
        
        # Las puntuaciones deben ser diferentes
        assert not np.isclose(score_unigram_heavy, score_ordered_heavy, rtol=1e-6)
        assert not np.isclose(score_ordered_heavy, score_unordered_heavy, rtol=1e-6)
    
    def test_mrf_sd_empty_query(self, sample_chunks):
        """Test MRF SD con query vacía."""
        query = ""
        chunk = sample_chunks[0]
        
        score = mrf_sd_score(query, chunk)
        
        assert score == 0.0
    
    def test_mrf_sd_no_matches(self, sample_chunks):
        """Test MRF SD sin coincidencias."""
        query = "diabetes mellitus"
        chunk = sample_chunks[0]
        
        score = mrf_sd_score(query, chunk)
        
        assert score == 0.0
    
    def test_mrf_sd_window_impact(self, sample_chunks):
        """Test que el tamaño de ventana afecta la puntuación."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]
        
        score_window_4 = mrf_sd_score(query, chunk, window=4)
        score_window_8 = mrf_sd_score(query, chunk, window=8)
        score_window_16 = mrf_sd_score(query, chunk, window=16)
        
        # Con ventana más grande, más probabilidad de encontrar bigramas no ordenados
        assert score_window_4 <= score_window_8 <= score_window_16


class TestMRFEnhanced:
    """Tests para MRF mejorado."""
    
    def test_mrf_enhanced_basic(self, sample_chunks):
        """Test básico de MRF mejorado."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]
        
        score = mrf_enhanced_score(query, chunk)
        
        assert isinstance(score, float)
        assert score > 0
    
    def test_mrf_enhanced_without_adaptive_weights(self, sample_chunks):
        """Test MRF mejorado sin pesos adaptativos."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]
        
        score = mrf_enhanced_score(query, chunk, use_adaptive_weights=False)
        
        assert isinstance(score, float)
        assert score > 0
    
    def test_mrf_enhanced_consistency(self, sample_chunks):
        """Test que MRF mejorado es consistente."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]
        
        score1 = mrf_enhanced_score(query, chunk)
        score2 = mrf_enhanced_score(query, chunk)
        
        assert abs(score1 - score2) < 1e-10  # Debe ser idéntico


class TestMRFEdgeCases:
    """Tests para casos extremos de MRF."""
    
    def test_mrf_empty_chunk(self):
        """Test MRF con chunk vacío."""
        empty_chunk = Chunk(
            id="empty",
            text="",
            doc_id="doc_empty",
            position=0,
            embedding=np.random.rand(384)
        )
        
        query = "hipoxemia postoperatoria"
        score = mrf_sd_score(query, empty_chunk)
        
        assert score == 0.0
    
    def test_mrf_single_token_query(self, sample_chunks):
        """Test MRF con query de un solo token."""
        query = "hipoxemia"
        chunk = sample_chunks[0]
        
        score = mrf_sd_score(query, chunk)
        
        # Solo debe considerar características unigram
        assert isinstance(score, float)
        assert score > 0
    
    def test_mrf_repeated_tokens(self, sample_chunks):
        """Test MRF con tokens repetidos en la query."""
        query = "hipoxemia hipoxemia hipoxemia"
        chunk = sample_chunks[0]
        
        score = mrf_sd_score(query, chunk)
        
        assert isinstance(score, float)
        assert score > 0
    
    def test_mrf_long_query(self, sample_chunks):
        """Test MRF con query larga."""
        query = "el paciente presenta síntomas de hipoxemia postoperatoria que requieren ajuste"
        chunk = sample_chunks[0]
        
        score = mrf_sd_score(query, chunk)
        
        assert isinstance(score, float)
        assert score > 0


class TestMRFIntegration:
    """Tests de integración para MRF."""
    
    def test_mrf_with_realistic_medical_text(self):
        """Test MRF con texto médico realista."""
        medical_text = """
        La hipoxemia postoperatoria es una complicación frecuente que requiere
        monitorización continua y ajuste de parámetros ventilatorios como PEEP.
        El manejo incluye oxigenoterapia, posicionamiento del paciente y
        evaluación de la función pulmonar.
        """
        
        chunk = Chunk(
            id="medical_chunk",
            text=medical_text,
            doc_id="medical_doc",
            position=0,
            embedding=np.random.rand(384)
        )
        
        query = "hipoxemia postoperatoria PEEP"
        score = mrf_sd_score(query, chunk)
        
        assert isinstance(score, float)
        assert score > 0  # Debe encontrar coincidencias
    
    def test_mrf_component_contribution(self, sample_chunks):
        """Test que cada componente contribuye a la puntuación final."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]
        
        # Calcular puntuaciones individuales
        unigram_score = _compute_unigram_score(
            ["hipoxemia", "postoperatoria"], 
            chunk.text.lower().split()
        )
        ordered_score = _compute_ordered_bigram_score(
            ["hipoxemia", "postoperatoria"], 
            chunk.text.lower().split()
        )
        unordered_score = _compute_unordered_bigram_score(
            ["hipoxemia", "postoperatoria"], 
            chunk.text.lower().split(), 
            window=8
        )
        
        # Puntuación combinada
        combined_score = mrf_sd_score(query, chunk)
        
        # Debe ser una combinación ponderada de las puntuaciones individuales
        assert isinstance(combined_score, float)
        assert combined_score > 0
    
    def test_mrf_ranking_consistency(self, sample_chunks):
        """Test que MRF produce rankings consistentes."""
        query = "hipoxemia postoperatoria"
        
        # Calcular puntuaciones para todos los chunks
        scores = []
        for chunk in sample_chunks:
            score = mrf_sd_score(query, chunk)
            scores.append((chunk.id, score))
        
        # Ordenar por puntuación
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Verificar que el ranking es consistente
        assert len(scores) == len(sample_chunks)
        assert all(isinstance(score, float) for _, score in scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
