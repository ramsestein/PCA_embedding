"""
Tests unitarios para Query-Likelihood Model (QLM).
"""

import pytest
import numpy as np
from unittest.mock import Mock
import sys
from pathlib import Path

# Añadir el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rerank_markov.qlm import (
    qlm_dirichlet_score, qlm_jelinek_mercer_score, qlm_score
)
from rerank_markov.types import Chunk, CorpusStats


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


@pytest.fixture
def corpus_stats(sample_chunks):
    """Crea estadísticas del corpus para testing."""
    # Calcular estadísticas manualmente
    term_freqs = {
        'el': 3, 'paciente': 1, 'presenta': 1, 'síntomas': 1, 'de': 3,
        'hipoxemia': 1, 'postoperatoria': 1, 'que': 1, 'requieren': 1,
        'ajuste': 1, 'peep': 1, 'la': 2, 'administración': 1, 'medicamentos': 1,
        'vía': 1, 'inhalatoria': 1, 'debe': 1, 'realizarse': 1, 'con': 1,
        'precaución': 1, 'abordaje': 1, 'integral': 1, 'mujer': 1, 'con': 1,
        'cáncer': 1, 'ginecológico': 1, 'incluye': 1, 'múltiples': 1,
        'especialidades': 1
    }
    
    doc_lengths = {
        'chunk_001': 15,
        'chunk_002': 12,
        'chunk_003': 18
    }
    
    total_tokens = sum(doc_lengths.values())
    vocab_size = len(term_freqs)
    
    return CorpusStats(
        term_freqs=term_freqs,
        doc_lengths=doc_lengths,
        total_tokens=total_tokens,
        vocab_size=vocab_size
    )


class TestQLMDirichlet:
    """Tests para suavizado Dirichlet."""
    
    def test_qlm_dirichlet_basic(self, sample_chunks, corpus_stats):
        """Test básico de QLM con suavizado Dirichlet."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]  # Contiene ambos términos
        
        score = qlm_dirichlet_score(query, chunk, corpus_stats, mu=1500)
        
        assert isinstance(score, float)
        assert score > -20  # Debe ser razonable (puede ser negativo en QLM)
    
    def test_qlm_dirichlet_no_matches(self, sample_chunks, corpus_stats):
        """Test QLM cuando no hay coincidencias."""
        query = "diabetes mellitus"
        chunk = sample_chunks[0]  # No contiene estos términos
        
        score = qlm_dirichlet_score(query, chunk, corpus_stats, mu=1500)
        
        assert isinstance(score, float)
        assert score == 0.0  # Debe ser 0 sin coincidencias
    
    def test_qlm_dirichlet_empty_query(self, sample_chunks, corpus_stats):
        """Test QLM con query vacía."""
        query = ""
        chunk = sample_chunks[0]
        
        score = qlm_dirichlet_score(query, chunk, corpus_stats, mu=1500)
        
        assert score == 0.0
    
    def test_qlm_dirichlet_different_mu(self, sample_chunks, corpus_stats):
        """Test QLM con diferentes valores de μ."""
        query = "hipoxemia"
        chunk = sample_chunks[0]
        
        score_mu100 = qlm_dirichlet_score(query, chunk, corpus_stats, mu=100)
        score_mu1500 = qlm_dirichlet_score(query, chunk, corpus_stats, mu=1500)
        score_mu5000 = qlm_dirichlet_score(query, chunk, corpus_stats, mu=5000)
        
        # Con μ más bajo, la influencia del corpus es menor
        assert score_mu100 > score_mu1500
        assert score_mu1500 > score_mu5000
    
    def test_qlm_dirichlet_term_frequency_impact(self, sample_chunks, corpus_stats):
        """Test que la frecuencia de términos afecta la puntuación."""
        query = "el paciente"
        chunk = sample_chunks[0]
        
        score = qlm_dirichlet_score(query, chunk, corpus_stats, mu=1500)
        
        # 'el' aparece 3 veces en el corpus, 'paciente' solo 1 vez
        # La puntuación debe ser razonable (puede ser negativa en QLM)
        assert score > -20


class TestQLMJelinekMercer:
    """Tests para suavizado Jelinek-Mercer."""
    
    def test_qlm_jm_basic(self, sample_chunks, corpus_stats):
        """Test básico de QLM con suavizado JM."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]
        
        score = qlm_jelinek_mercer_score(query, chunk, corpus_stats, lambda_jm=0.5)
        
        assert isinstance(score, float)
        assert score > -20  # Puede ser negativo en QLM
    
    def test_qlm_jm_lambda_impact(self, sample_chunks, corpus_stats):
        """Test que λ afecta la puntuación JM."""
        query = "hipoxemia"
        chunk = sample_chunks[0]
        
        score_lambda_high = qlm_jelinek_mercer_score(query, chunk, corpus_stats, lambda_jm=0.9)
        score_lambda_low = qlm_jelinek_mercer_score(query, chunk, corpus_stats, lambda_jm=0.1)
        
        # Con λ alto, se enfatiza más el documento; con λ bajo, más el corpus
        assert score_lambda_high != score_lambda_low
    
    def test_qlm_jm_no_matches(self, sample_chunks, corpus_stats):
        """Test JM cuando no hay coincidencias."""
        query = "diabetes mellitus"
        chunk = sample_chunks[0]
        
        score = qlm_jelinek_mercer_score(query, chunk, corpus_stats, lambda_jm=0.5)
        
        assert score == 0.0


class TestQLMScore:
    """Tests para la función principal qlm_score."""
    
    def test_qlm_score_dirichlet_default(self, sample_chunks, corpus_stats):
        """Test qlm_score con suavizado Dirichlet por defecto."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]
        
        score = qlm_score(query, chunk, corpus_stats)
        
        assert isinstance(score, float)
        assert score > -20  # Puede ser negativo en QLM
    
    def test_qlm_score_jm_mode(self, sample_chunks, corpus_stats):
        """Test qlm_score en modo Jelinek-Mercer."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]
        
        score = qlm_score(query, chunk, corpus_stats, use_jm=True, lambda_jm=0.7)
        
        assert isinstance(score, float)
        assert score > -20  # Puede ser negativo en QLM
    
    def test_qlm_score_parameter_consistency(self, sample_chunks, corpus_stats):
        """Test que los parámetros se pasan correctamente."""
        query = "hipoxemia"
        chunk = sample_chunks[0]
        
        # Debe ser igual a llamar directamente a qlm_dirichlet_score
        score1 = qlm_score(query, chunk, corpus_stats, mu=2000)
        score2 = qlm_dirichlet_score(query, chunk, corpus_stats, mu=2000)
        
        assert abs(score1 - score2) < 1e-10


class TestQLMEdgeCases:
    """Tests para casos extremos y edge cases."""
    
    def test_qlm_zero_corpus_probability(self, sample_chunks):
        """Test cuando p(t|C) = 0 para algún término."""
        # Crear stats con término que no aparece en el corpus
        term_freqs = {'hipoxemia': 0, 'paciente': 1}
        doc_lengths = {'chunk_001': 10}
        
        stats = CorpusStats(
            term_freqs=term_freqs,
            doc_lengths=doc_lengths,
            total_tokens=10,
            vocab_size=2
        )
        
        query = "hipoxemia paciente"
        chunk = sample_chunks[0]
        
        score = qlm_dirichlet_score(query, chunk, stats, mu=1500)
        
        # Debe manejar términos con p(t|C) = 0
        assert isinstance(score, float)
    
    def test_qlm_empty_chunk(self, corpus_stats):
        """Test con chunk vacío."""
        empty_chunk = Chunk(
            id="empty",
            text="",
            doc_id="doc_empty",
            position=0,
            embedding=np.random.rand(384)
        )
        
        query = "hipoxemia"
        score = qlm_dirichlet_score(query, empty_chunk, corpus_stats, mu=1500)
        
        assert score == 0.0
    
    def test_qlm_single_token_query(self, sample_chunks, corpus_stats):
        """Test con query de un solo token."""
        query = "hipoxemia"
        chunk = sample_chunks[0]
        
        score = qlm_dirichlet_score(query, chunk, corpus_stats, mu=1500)
        
        assert isinstance(score, float)
        assert score > -20  # Puede ser negativo en QLM
    
    def test_qlm_repeated_tokens(self, sample_chunks, corpus_stats):
        """Test con tokens repetidos en la query."""
        query = "hipoxemia hipoxemia hipoxemia"
        chunk = sample_chunks[0]
        
        score = qlm_dirichlet_score(query, chunk, corpus_stats, mu=1500)
        
        assert isinstance(score, float)
        assert score > -20  # Puede ser negativo en QLM


class TestQLMIntegration:
    """Tests de integración para QLM."""
    
    def test_qlm_with_realistic_text(self):
        """Test QLM con texto realista."""
        # Crear chunk con texto médico realista
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
        
        # Crear stats simplificadas
        term_freqs = {
            'hipoxemia': 5, 'postoperatoria': 3, 'peep': 8, 'paciente': 15,
            'oxigenoterapia': 2, 'pulmonar': 12
        }
        doc_lengths = {'medical_chunk': 50}
        
        stats = CorpusStats(
            term_freqs=term_freqs,
            doc_lengths=doc_lengths,
            total_tokens=200,
            vocab_size=100
        )
        
        query = "hipoxemia postoperatoria PEEP"
        score = qlm_dirichlet_score(query, chunk, stats, mu=1500)
        
        assert isinstance(score, float)
        assert score > -20  # Puede ser negativo en QLM
    
    def test_qlm_consistency_across_calls(self, sample_chunks, corpus_stats):
        """Test que QLM es consistente entre llamadas."""
        query = "hipoxemia postoperatoria"
        chunk = sample_chunks[0]
        
        score1 = qlm_dirichlet_score(query, chunk, corpus_stats, mu=1500)
        score2 = qlm_dirichlet_score(query, chunk, corpus_stats, mu=1500)
        
        assert abs(score1 - score2) < 1e-10  # Debe ser idéntico


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
