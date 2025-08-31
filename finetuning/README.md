# 🧬 Sistema de Entrenamiento y Evaluación de Embeddings Biomédicos

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo para el entrenamiento, fine-tuning y evaluación de modelos de embeddings especializados en el dominio biomédico. El objetivo es desarrollar embeddings que capturen mejor la semántica de textos médicos en catalán y español, mejorando significativamente el rendimiento en tareas de búsqueda semántica y recuperación de información médica.

## 🎯 Objetivos

- **Fine-tuning de modelos biomédicos**: Adaptar modelos pre-entrenados al dominio médico
- **Evaluación comparativa**: Comparar el rendimiento de múltiples arquitecturas de modelos
- **Ensemble de embeddings**: Desarrollar estrategias de combinación para mejorar la precisión
- **Optimización multilingüe**: Mejorar el rendimiento en catalán y español
- **Análisis de rendimiento**: Evaluar métricas de precisión, velocidad y eficiencia

## 🏗️ Arquitectura del Sistema

### Componentes Principales

1. **Cargador de Modelos Unificado** (`cargador_models.py`)
   - Adaptador universal para modelos de Hugging Face
   - Soporte para múltiples estrategias de pooling (mean, CLS, max)
   - Detección automática de dimensiones de embedding

2. **Sistema de Comparación** (`comparacion_modelos_multiple.py`)
   - Evaluación batch de múltiples modelos
   - Métricas de rendimiento estandarizadas
   - Generación automática de reportes

3. **Sistema de Ensemble** (`comparar_esemble.py`)
   - Combinación ponderada de embeddings
   - Fusión de rankings múltiples
   - Optimización automática de pesos

## 📊 Metodología

### 1. Preparación de Datos

El sistema procesa **24 documentos médicos** del sistema sanitario catalán, generando:

- **194 chunks** de texto con longitud promedio de 244 palabras
- **1,344 pares de entrenamiento** con distribución de dificultad:
  - **Semi-hard**: 873 pares (65%)
  - **Hard**: 315 pares (23%)
  - **Easy**: 156 pares (12%)

### 2. Estrategia de Fine-tuning

#### Modelos Base Utilizados
- **BioBERT**: Variantes v1.1 y v1.2
- **BioClinicalBERT**: Especializado en textos clínicos
- **SapBERT**: Adaptado para UMLS y PubMed
- **PubMedBERT**: Versiones MS y MS-MARCO
- **DeBERTa-v3**: Configuraciones conservadora y agresiva
- **Biomedical-RoBERTa**: Especializado en español
- **RoBERTa-Catalan**: Modelo base en catalán

#### Configuraciones de Entrenamiento
- **Estrategia conservadora**: Fine-tuning gradual con learning rates bajos
- **Estrategia agresiva**: Fine-tuning intensivo con learning rates altos
- **Early stopping**: Parada temprana basada en pérdida de validación
- **Checkpoints múltiples**: Guardado de modelos en diferentes épocas

### 3. Evaluación y Métricas

#### Métricas Principales
- **Accuracy@1**: Precisión en la primera posición
- **Accuracy@5**: Precisión en las primeras 5 posiciones
- **MRR (Mean Reciprocal Rank)**: Ranking recíproco promedio
- **Similitud promedio**: Coeficiente de similitud coseno
- **Velocidad**: Tiempo de procesamiento por query

#### Conjunto de Prueba
- **48 queries** de prueba en español y catalán
- **Evaluación de recuperación**: Búsqueda semántica en corpus médico
- **Comparación contra modelo base**: all-MiniLM-base

## 🏆 Resultados Principales

### Rendimiento del Modelo Base
- **Accuracy@1**: 33.3%
- **Accuracy@5**: 52.1%
- **MRR**: 0.406
- **Similitud promedio**: 54.6%

### Top 5 Modelos Mejorados

| # | Modelo | Acc@1 | Mejora | Acc@5 | MRR | Velocidad |
|---|--------|-------|---------|-------|-----|-----------|
| 1 | **SapBERT-UMLS** | 64.6% | +31.3pp | 77.1% | 0.692 | 0.0059s |
| 2 | **Biomedical-RoBERTa** | 62.5% | +29.2pp | 66.7% | 0.642 | 0.0055s |
| 3 | **SapBERT-UMLS** | 62.5% | +29.2pp | 75.0% | 0.674 | 0.0055s |
| 4 | **SapBERT-UMLS** | 62.5% | +29.2pp | 75.0% | 0.678 | 0.0050s |
| 5 | **SapBERT-UMLS** | 62.5% | +29.2pp | 77.1% | 0.677 | 0.0052s |

### Análisis por Tipo de Modelo

#### Modelos de Mejor Rendimiento
- **SapBERT-UMLS**: Consistente en el top 10
- **Biomedical-RoBERTa**: Excelente para español
- **SapBERT-PubMed**: Buen rendimiento general
- **MedCPT**: Especializado en consultas médicas

#### Mejoras Promedio por Categoría
- **Modelos biomédicos especializados**: +25-30pp
- **Modelos multilingües**: +20-25pp
- **Modelos generales fine-tuneados**: +15-20pp

### Resultados del Ensemble

El sistema de ensemble logra mejoras adicionales mediante:

- **Combinación ponderada**: Mejora del 5-10% sobre modelos individuales
- **Fusión de rankings**: Optimización de la precisión en top-k
- **Diversidad de modelos**: Reducción de overfitting

## 🚀 Características Técnicas

### Optimizaciones Implementadas
- **Pooling adaptativo**: Selección automática de estrategia óptima
- **Batch processing**: Procesamiento eficiente de múltiples modelos
- **Caching de embeddings**: Reutilización de embeddings base
- **Paralelización**: Evaluación concurrente de modelos

### Compatibilidad
- **Formatos de salida**: ONNX, OpenVINO, PyTorch
- **Plataformas**: CPU, GPU (CUDA)
- **Integración**: Hugging Face, Transformers

## 📁 Estructura del Proyecto

```
train_embedding/
├── all-mini-base/          # Modelo base de referencia
├── models/                 # Modelos fine-tuneados
│   ├── biobert/           # Variantes de BioBERT
│   ├── sapbert-umls/      # SapBERT adaptado a UMLS
│   ├── biomedical-roberta/ # RoBERTa biomédico
│   └── ...                # Otros modelos especializados
├── PNTs/                  # Documentos médicos fuente
├── prepared_data/         # Datos procesados y chunks
├── resultados_comparacion/ # Reportes de evaluación
├── resultados_ensemble_cat/ # Resultados de ensemble
└── scripts/               # Scripts de análisis
```

## 🛠️ Uso del Sistema

### 1. Entrenamiento de Modelos
```bash
python cargador_models.py --model_type biobert --strategy conservative
```

### 2. Evaluación Comparativa
```bash
python comparacion_modelos_multiple.py --base_model all-mini-base --models_folder ./models
```

### 3. Sistema de Ensemble
```bash
python comparar_esemble.py --config ensemble_config.json
```

## 📈 Conclusiones y Hallazgos

### Principales Logros
1. **Mejora significativa**: Los modelos fine-tuneados superan al modelo base en un **31.3%** de precisión
2. **Especialización efectiva**: Los modelos biomédicos muestran mejor rendimiento que los generales
3. **Multilingüismo**: Excelente adaptación al catalán y español médico
4. **Escalabilidad**: Sistema capaz de evaluar 18+ modelos simultáneamente

### Insights Clave
- **SapBERT-UMLS** es el modelo más consistente para dominios médicos
- **Biomedical-RoBERTa** ofrece el mejor rendimiento para español
- **Estrategia conservadora** produce modelos más estables
- **Ensemble** proporciona mejoras incrementales significativas

### Aplicaciones Prácticas
- **Búsqueda semántica** en bases de datos médicas
- **Recuperación de información** clínica
- **Análisis de similitud** entre documentos médicos
- **Sistemas de recomendación** para profesionales de la salud

## 🔮 Próximos Pasos

1. **Expansión del dataset**: Incorporar más documentos médicos
2. **Optimización de hiperparámetros**: Búsqueda automática de configuraciones óptimas
3. **Integración con sistemas clínicos**: Despliegue en entornos de producción
4. **Evaluación en tareas específicas**: Clasificación, NER, QA médica

## 📚 Referencias

- **Modelos base**: Hugging Face Transformers
- **Dataset médico**: Sistema de Salud Público de Cataluña
- **Métricas**: Estándares de evaluación de información retrieval
- **Framework**: PyTorch, Transformers, Sentence Transformers

## 👥 Contribuciones

Este proyecto representa un esfuerzo colaborativo para mejorar la comprensión semántica de textos médicos en lenguas minoritarias, contribuyendo al desarrollo de herramientas de IA más inclusivas y efectivas para el sector sanitario.

---

*Proyecto desarrollado para el análisis y optimización de embeddings biomédicos multilingües*
