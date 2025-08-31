"""
Tests unitarios para Personalized PageRank (PPR) y construcción del grafo.
"""

import pytest
import numpy as np
import networkx as nx
import sys
from pathlib import Path

# Añadir el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rerank_markov.graph import (
    build_chunk_graph, personalized_pagerank, build_enhanced_chunk_graph,
    compute_graph_metrics, personalized_pagerank_with_teleport
)
from rerank_markov.types import Chunk


@pytest.fixture
def sample_chunks():
    """Crea chunks de ejemplo para testing."""
    chunks = [
        Chunk(
            id="chunk_001",
            text="El paciente presenta síntomas de hipoxemia postoperatoria.",
            doc_id="doc_01",
            position=0,
            embedding=np.random.rand(384),
            meta={'length': 10, 'tokens': 10}
        ),
        Chunk(
            id="chunk_002",
            text="La administración de medicamentos vía inhalatoria.",
            doc_id="doc_01",
            position=1,
            embedding=np.random.rand(384),
            meta={'length': 8, 'tokens': 8}
        ),
        Chunk(
            id="chunk_003",
            text="El abordaje integral de la mujer con cáncer ginecológico.",
            doc_id="doc_02",
            position=0,
            embedding=np.random.rand(384),
            meta={'length': 12, 'tokens': 12}
        ),
        Chunk(
            id="chunk_004",
            text="La crisis epiléptica generalizada requiere intervención.",
            doc_id="doc_02",
            position=1,
            embedding=np.random.rand(384),
            meta={'length': 9, 'tokens': 9}
        )
    ]
    return chunks


class TestGraphConstruction:
    """Tests para construcción del grafo."""
    
    def test_build_chunk_graph_basic(self, sample_chunks):
        """Test básico de construcción del grafo."""
        G = build_chunk_graph(sample_chunks)
        
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == len(sample_chunks)
        assert G.number_of_edges() > 0  # Debe haber aristas
    
    def test_build_chunk_graph_empty(self):
        """Test construcción del grafo con lista vacía."""
        G = build_chunk_graph([])
        
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
    
    def test_build_chunk_graph_weights(self, sample_chunks):
        """Test que los pesos afectan la construcción del grafo."""
        # Grafo con pesos altos en embeddings
        G_emb_heavy = build_chunk_graph(sample_chunks, alpha=0.8, beta=0.1, gamma=0.1)
        
        # Grafo con pesos altos en contigüidad
        G_contig_heavy = build_chunk_graph(sample_chunks, alpha=0.1, beta=0.8, gamma=0.1)
        
        # Los grafos deben tener el mismo número de nodos y aristas
        assert G_emb_heavy.number_of_nodes() == G_contig_heavy.number_of_nodes()
        assert G_emb_heavy.number_of_edges() == G_contig_heavy.number_of_edges()
        
        # Pero los pesos de las aristas deben ser diferentes
        edge_weights_emb = [data['weight'] for _, _, data in G_emb_heavy.edges(data=True)]
        edge_weights_contig = [data['weight'] for _, _, data in G_contig_heavy.edges(data=True)]
        
        # Al menos algunas aristas deben tener pesos diferentes
        assert len(set(edge_weights_emb)) > 1 or len(set(edge_weights_contig)) > 1
    
    def test_build_chunk_graph_contiguity(self, sample_chunks):
        """Test que la contigüidad se refleja en el grafo."""
        G = build_chunk_graph(sample_chunks, alpha=0.0, beta=1.0, gamma=0.0)
        
        # Solo debe considerar contigüidad
        # chunk_001 y chunk_002 están en el mismo documento y son consecutivos
        assert G.has_edge("chunk_001", "chunk_002")
        assert G.has_edge("chunk_003", "chunk_004")
        
        # No debe haber aristas entre documentos diferentes
        assert not G.has_edge("chunk_001", "chunk_003")
    
    def test_build_chunk_graph_embedding_similarity(self, sample_chunks):
        """Test que la similitud de embeddings se refleja en el grafo."""
        G = build_chunk_graph(sample_chunks, alpha=1.0, beta=0.0, gamma=0.0)
        
        # Debe haber aristas basadas en similitud de embeddings
        assert G.number_of_edges() > 0
        
        # Verificar que las aristas tienen pesos
        for _, _, data in G.edges(data=True):
            assert 'weight' in data
            assert data['weight'] > 0
    
    def test_build_chunk_graph_links(self, sample_chunks):
        """Test que los enlaces se reflejan en el grafo."""
        # Añadir enlaces a los chunks
        sample_chunks[0].meta['links'] = ['chunk_002']
        sample_chunks[1].meta['links'] = ['chunk_003']
        
        G = build_chunk_graph(sample_chunks, alpha=0.0, beta=0.0, gamma=1.0)
        
        # Debe haber aristas por enlaces
        assert G.has_edge("chunk_001", "chunk_002")
        assert G.has_edge("chunk_002", "chunk_003")
    
    def test_build_enhanced_chunk_graph(self, sample_chunks):
        """Test construcción del grafo mejorado."""
        G = build_enhanced_chunk_graph(sample_chunks)
        
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == len(sample_chunks)
        assert G.number_of_edges() > 0
    
    def test_graph_metrics(self, sample_chunks):
        """Test cálculo de métricas del grafo."""
        G = build_chunk_graph(sample_chunks)
        metrics = compute_graph_metrics(G)
        
        assert 'num_nodes' in metrics
        assert 'num_edges' in metrics
        assert 'density' in metrics
        assert metrics['num_nodes'] == len(sample_chunks)
        assert metrics['num_edges'] >= 0
        assert 0 <= metrics['density'] <= 1


class TestPersonalizedPageRank:
    """Tests para Personalized PageRank."""
    
    def test_personalized_pagerank_basic(self, sample_chunks):
        """Test básico de PPR."""
        G = build_chunk_graph(sample_chunks)
        seed_ids = ["chunk_001", "chunk_002"]
        
        ppr_scores = personalized_pagerank(G, seed_ids)
        
        assert isinstance(ppr_scores, dict)
        assert len(ppr_scores) == G.number_of_nodes()
        
        # Los nodos semilla deben tener puntuaciones positivas
        for seed_id in seed_ids:
            assert ppr_scores[seed_id] > 0
        
        # Las puntuaciones deben sumar aproximadamente 1
        total_score = sum(ppr_scores.values())
        assert abs(total_score - 1.0) < 0.1
    
    def test_personalized_pagerank_empty_graph(self):
        """Test PPR con grafo vacío."""
        G = nx.DiGraph()
        seed_ids = ["chunk_001"]
        
        ppr_scores = personalized_pagerank(G, seed_ids)
        
        assert ppr_scores == {}
    
    def test_personalized_pagerank_no_seeds(self, sample_chunks):
        """Test PPR sin nodos semilla."""
        G = build_chunk_graph(sample_chunks)
        seed_ids = []
        
        ppr_scores = personalized_pagerank(G, seed_ids)
        
        assert ppr_scores == {}
    
    def test_personalized_pagerank_lambda_impact(self, sample_chunks):
        """Test que λ afecta las puntuaciones PPR."""
        G = build_chunk_graph(sample_chunks)
        seed_ids = ["chunk_001"]
        
        # Con λ alto, más influencia del grafo
        ppr_high_lambda = personalized_pagerank(G, seed_ids, lambda_=0.95)
        
        # Con λ bajo, más influencia del reinicio
        ppr_low_lambda = personalized_pagerank(G, seed_ids, lambda_=0.5)
        
        # Las puntuaciones deben ser diferentes
        assert ppr_high_lambda != ppr_low_lambda
    
    def test_personalized_pagerank_convergence(self, sample_chunks):
        """Test que PPR converge."""
        G = build_chunk_graph(sample_chunks)
        seed_ids = ["chunk_001"]
        
        # Con tolerancia alta, debe converger rápido
        ppr_high_tol = personalized_pagerank(G, seed_ids, tol=1e-3)
        
        # Con tolerancia baja, debe converger más lento
        ppr_low_tol = personalized_pagerank(G, seed_ids, tol=1e-8)
        
        # Ambas deben producir resultados válidos
        assert len(ppr_high_tol) == len(ppr_low_tol)
        assert all(isinstance(score, float) for score in ppr_high_tol.values())
        assert all(isinstance(score, float) for score in ppr_low_tol.values())
    
    def test_personalized_pagerank_seed_distribution(self, sample_chunks):
        """Test que la distribución de semillas afecta PPR."""
        G = build_chunk_graph(sample_chunks)
        
        # Una semilla
        ppr_single = personalized_pagerank(G, ["chunk_001"])
        
        # Múltiples semillas
        ppr_multiple = personalized_pagerank(G, ["chunk_001", "chunk_002", "chunk_003"])
        
        # Las puntuaciones deben ser diferentes
        assert ppr_single != ppr_multiple
        
        # Con múltiples semillas, la puntuación total debe distribuirse
        total_single = sum(ppr_single.values())
        total_multiple = sum(ppr_multiple.values())
        
        assert abs(total_single - 1.0) < 0.1
        assert abs(total_multiple - 1.0) < 0.1
    
    def test_personalized_pagerank_with_teleport(self, sample_chunks):
        """Test PPR con teleportación."""
        G = build_chunk_graph(sample_chunks)
        seed_ids = ["chunk_001"]
        
        ppr_teleport = personalized_pagerank_with_teleport(
            G, seed_ids, lambda_=0.85, teleport_prob=0.1
        )
        
        assert isinstance(ppr_teleport, dict)
        assert len(ppr_teleport) == G.number_of_nodes()
        
        # Las puntuaciones deben sumar aproximadamente 1
        total_score = sum(ppr_teleport.values())
        assert abs(total_score - 1.0) < 0.1


class TestGraphStructure:
    """Tests para la estructura del grafo."""
    
    def test_graph_node_attributes(self, sample_chunks):
        """Test que los nodos tienen atributos correctos."""
        G = build_chunk_graph(sample_chunks)
        
        for chunk in sample_chunks:
            assert chunk.id in G.nodes()
            assert 'chunk' in G.nodes[chunk.id]
            assert G.nodes[chunk.id]['chunk'] == chunk
    
    def test_graph_edge_weights(self, sample_chunks):
        """Test que las aristas tienen pesos válidos."""
        G = build_chunk_graph(sample_chunks)
        
        for _, _, data in G.edges(data=True):
            assert 'weight' in data
            assert isinstance(data['weight'], (int, float))
            assert data['weight'] > 0
    
    def test_graph_connectivity(self, sample_chunks):
        """Test conectividad del grafo."""
        G = build_chunk_graph(sample_chunks)
        
        # El grafo debe tener al menos un nodo
        if G.number_of_nodes() > 0:
            # Debe haber al menos algunas aristas
            assert G.number_of_edges() > 0
            
            # Verificar que no hay nodos aislados (con self-loops)
            for node in G.nodes():
                assert G.in_degree(node) > 0 or G.out_degree(node) > 0
    
    def test_graph_directionality(self, sample_chunks):
        """Test que el grafo es dirigido."""
        G = build_chunk_graph(sample_chunks)
        
        assert isinstance(G, nx.DiGraph)
        
        # Verificar que las aristas son dirigidas
        if G.number_of_edges() > 0:
            for u, v in G.edges():
                # Si hay arista u->v, no necesariamente debe haber v->u
                # Pero ambas pueden existir
                pass


class TestPPREdgeCases:
    """Tests para casos extremos de PPR."""
    
    def test_ppr_disconnected_graph(self):
        """Test PPR con grafo desconectado."""
        G = nx.DiGraph()
        G.add_nodes_from(['A', 'B', 'C'])
        # Sin aristas
        
        seed_ids = ['A']
        ppr_scores = personalized_pagerank(G, seed_ids)
        
        # Solo el nodo semilla debe tener puntuación positiva
        assert ppr_scores['A'] > 0
        assert ppr_scores['B'] == 0.0
        assert ppr_scores['C'] == 0.0
    
    def test_ppr_single_node_graph(self):
        """Test PPR con grafo de un solo nodo."""
        G = nx.DiGraph()
        G.add_node('A')
        
        seed_ids = ['A']
        ppr_scores = personalized_pagerank(G, seed_ids)
        
        assert ppr_scores['A'] == 1.0
    
    def test_ppr_self_loops(self):
        """Test PPR con self-loops."""
        G = nx.DiGraph()
        G.add_edge('A', 'A', weight=1.0)
        G.add_edge('B', 'B', weight=1.0)
        
        seed_ids = ['A']
        ppr_scores = personalized_pagerank(G, seed_ids)
        
        assert ppr_scores['A'] > ppr_scores['B']  # A debe tener puntuación más alta
    
    def test_ppr_large_lambda(self, sample_chunks):
        """Test PPR con λ muy alto."""
        G = build_chunk_graph(sample_chunks)
        seed_ids = ["chunk_001"]
        
        ppr_scores = personalized_pagerank(G, seed_ids, lambda_=0.99)
        
        assert isinstance(ppr_scores, dict)
        assert len(ppr_scores) == G.number_of_nodes()
    
    def test_ppr_small_lambda(self, sample_chunks):
        """Test PPR con λ muy bajo."""
        G = build_chunk_graph(sample_chunks)
        seed_ids = ["chunk_001"]
        
        ppr_scores = personalized_pagerank(G, seed_ids, lambda_=0.01)
        
        assert isinstance(ppr_scores, dict)
        assert len(ppr_scores) == G.number_of_nodes()


class TestPPRIntegration:
    """Tests de integración para PPR."""
    
    def test_ppr_ranking_consistency(self, sample_chunks):
        """Test que PPR produce rankings consistentes."""
        G = build_chunk_graph(sample_chunks)
        seed_ids = ["chunk_001"]
        
        # Ejecutar PPR múltiples veces
        ppr_1 = personalized_pagerank(G, seed_ids)
        ppr_2 = personalized_pagerank(G, seed_ids)
        
        # Los resultados deben ser idénticos
        assert ppr_1 == ppr_2
        
        # Verificar que el ranking es consistente
        sorted_1 = sorted(ppr_1.items(), key=lambda x: x[1], reverse=True)
        sorted_2 = sorted(ppr_2.items(), key=lambda x: x[1], reverse=True)
        
        assert sorted_1 == sorted_2
    
    def test_ppr_with_realistic_graph(self):
        """Test PPR con grafo realista."""
        # Crear grafo con estructura realista
        G = nx.DiGraph()
        
        # Añadir nodos
        for i in range(10):
            G.add_node(f"chunk_{i:03d}")
        
        # Añadir aristas con pesos realistas
        for i in range(9):
            weight = 0.5 + 0.3 * np.random.random()
            G.add_edge(f"chunk_{i:03d}", f"chunk_{i+1:03d}", weight=weight)
        
        # Añadir algunas aristas adicionales
        G.add_edge("chunk_000", "chunk_005", weight=0.2)
        G.add_edge("chunk_003", "chunk_008", weight=0.3)
        
        seed_ids = ["chunk_000", "chunk_005"]
        ppr_scores = personalized_pagerank(G, seed_ids)
        
        assert isinstance(ppr_scores, dict)
        assert len(ppr_scores) == 10
        
        # Los nodos semilla deben tener puntuaciones altas
        for seed_id in seed_ids:
            assert ppr_scores[seed_id] > 0
        
        # Las puntuaciones deben sumar aproximadamente 1
        total_score = sum(ppr_scores.values())
        assert abs(total_score - 1.0) < 0.1
    
    def test_ppr_component_interaction(self, sample_chunks):
        """Test que los componentes del grafo interactúan correctamente."""
        # Grafo con diferentes configuraciones
        configs = [
            (0.8, 0.1, 0.1),  # Enfasis en embeddings
            (0.1, 0.8, 0.1),  # Enfasis en contigüidad
            (0.1, 0.1, 0.8),  # Enfasis en enlaces
            (0.33, 0.33, 0.34)  # Balanceado
        ]
        
        seed_ids = ["chunk_001"]
        results = []
        
        for alpha, beta, gamma in configs:
            G = build_chunk_graph(sample_chunks, alpha, beta, gamma)
            ppr_scores = personalized_pagerank(G, seed_ids)
            results.append(ppr_scores)
        
        # Los resultados deben ser diferentes para diferentes configuraciones
        assert len(set(tuple(sorted(r.items())) for r in results)) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
