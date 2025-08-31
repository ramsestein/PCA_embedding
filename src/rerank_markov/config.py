"""
Configuración del re-ranker híbrido con validaciones Pydantic.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator
import yaml
import json


class RerankConfig(BaseModel):
    """Configuración principal del re-ranker híbrido."""
    
    # Parámetros de PPR
    lambda_ppr: float = Field(0.85, ge=0.0, le=1.0, description="Factor de damping para PPR")
    k: int = Field(30, gt=0, description="Número de chunks a retornar")
    k_prime: int = Field(100, gt=0, description="Número de candidatos para PPR")
    max_iter: int = Field(100, gt=0, description="Máximo de iteraciones para PPR")
    tol: float = Field(1e-6, gt=0.0, description="Tolerancia de convergencia para PPR")
    
    # Pesos para construcción del grafo
    alpha: float = Field(0.6, ge=0.0, le=1.0, description="Peso para similitud de embeddings")
    beta: float = Field(0.25, ge=0.0, le=1.0, description="Peso para contigüidad")
    gamma: float = Field(0.15, ge=0.0, le=1.0, description="Peso para enlaces")
    
    # Parámetros de QLM
    mu: int = Field(1500, gt=0, description="Parámetro de suavizado Dirichlet")
    use_jm: bool = Field(False, description="Usar suavizado Jelinek-Mercer en lugar de Dirichlet")
    lambda_jm: float = Field(0.5, ge=0.0, le=1.0, description="Parámetro lambda para JM")
    
    # Parámetros de MRF
    window_size: int = Field(8, gt=0, description="Tamaño de ventana para bigramas no ordenados")
    w_unigram: float = Field(0.8, ge=0.0, le=1.0, description="Peso para características unigram")
    w_ordered: float = Field(0.1, ge=0.0, le=1.0, description="Peso para bigramas ordenados")
    w_unordered: float = Field(0.1, ge=0.0, le=1.0, description="Peso para bigramas no ordenados")
    
    # Pesos de fusión final
    a: float = Field(0.45, ge=0.0, le=1.0, description="Peso para embeddings")
    b: float = Field(0.25, ge=0.0, le=1.0, description="Peso para PPR")
    c: float = Field(0.20, ge=0.0, le=1.0, description="Peso para QLM")
    d: float = Field(0.10, ge=0.0, le=1.0, description="Peso para MRF")
    
    # Configuración de normalización
    use_zscore: bool = Field(True, description="Usar normalización z-score")
    clip_sigma: float = Field(3.0, gt=0.0, description="Clipping para normalización z-score")
    
    @validator('k_prime')
    def validate_k_prime(cls, v, values):
        """K' debe ser mayor o igual que K."""
        if 'k' in values and v < values['k']:
            raise ValueError("k_prime debe ser mayor o igual que k")
        return v
    
    @validator('alpha', 'beta', 'gamma')
    def validate_graph_weights(cls, v, values):
        """Los pesos del grafo deben sumar aproximadamente 1."""
        if 'alpha' in values and 'beta' in values and 'gamma' in values:
            total = values['alpha'] + values['beta'] + values['gamma']
            if abs(total - 1.0) > 0.1:
                raise ValueError("alpha + beta + gamma debe ser aproximadamente 1.0")
        return v
    
    @validator('a', 'b', 'c', 'd')
    def validate_fusion_weights(cls, v, values):
        """Los pesos de fusión deben sumar aproximadamente 1."""
        if 'a' in values and 'b' in values and 'c' in values and 'd' in values:
            total = values['a'] + values['b'] + values['c'] + values['d']
            if abs(total - 1.0) > 0.1:
                raise ValueError("a + b + c + d debe ser aproximadamente 1.0")
        return v
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'RerankConfig':
        """Carga configuración desde archivo YAML."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, file_path: str) -> 'RerankConfig':
        """Carga configuración desde archivo JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, file_path: str) -> None:
        """Guarda configuración en archivo YAML."""
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    def to_json(self, file_path: str) -> None:
        """Guarda configuración en archivo JSON."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.dict(), f, indent=2, ensure_ascii=False)
