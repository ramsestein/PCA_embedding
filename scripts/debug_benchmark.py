#!/usr/bin/env python3
"""
Script de debug para investigar el benchmark de PNTs.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

def debug_benchmark():
    """Función de debug para investigar el benchmark."""
    
    # 1. Cargar queries del benchmark
    benchmark_file = Path("benchmark/preguntas_con_docs_es.json")
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    print(f"Total de queries: {len(queries)}")
    
    # 2. Mostrar algunas queries y documentos esperados
    print("\n=== PRIMERAS 5 QUERIES ===")
    for i, query_data in enumerate(queries[:5]):
        print(f"Query {i+1}: {query_data['query']}")
        print(f"Documento esperado: {query_data['document_expected']}")
        print()
    
    # 3. Cargar modelo
    print("Cargando modelo...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Cargar un documento PNTs como ejemplo
    pnts_dir = Path("PNTs")
    txt_files = list(pnts_dir.glob("*.txt"))
    
    print(f"\n=== DOCUMENTOS PNTs ENCONTRADOS ===")
    for txt_file in txt_files[:5]:
        print(f"Archivo: {txt_file.name}")
        print(f"Tamaño: {txt_file.stat().st_size} bytes")
        
        # Leer contenido
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Primeras 100 chars: {content[:100]}...")
        print()
    
    # 5. Probar una query específica
    print("=== PRUEBA DE QUERY ESPECÍFICA ===")
    test_query = queries[0]['query']
    expected_doc = queries[0]['document_expected']
    
    print(f"Query de prueba: {test_query}")
    print(f"Documento esperado: {expected_doc}")
    
    # Normalizar nombre del documento
    expected_doc_normalized = expected_doc.replace('_limpio.txt', '_limpio')
    print(f"Documento normalizado: {expected_doc_normalized}")
    
    # Buscar el documento correspondiente
    matching_file = None
    for txt_file in txt_files:
        if txt_file.stem == expected_doc_normalized:
            matching_file = txt_file
            break
    
    if matching_file:
        print(f"Archivo encontrado: {matching_file.name}")
        
        # Leer contenido
        with open(matching_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Contenido del archivo:")
        print(f"Longitud: {len(content)} caracteres")
        print(f"Primeras 200 chars: {content[:200]}...")
        
        # Calcular embedding de la query
        query_embedding = model.encode([test_query])[0]
        print(f"Embedding de la query calculado: {query_embedding.shape}")
        
        # Dividir en chunks y calcular embeddings
        sentences = content.split('.')
        chunks = []
        for i, sentence in enumerate(sentences[:3]):  # Solo primeros 3 chunks
            if sentence.strip():
                chunk_text = sentence.strip()
                chunks.append(chunk_text)
        
        print(f"Chunks creados: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk[:100]}...")
        
        # Calcular embeddings de chunks
        chunk_embeddings = model.encode(chunks)
        
        # Calcular similitudes
        print(f"\nSimilitudes coseno:")
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
            print(f"Chunk {i+1}: {similarity:.4f}")
    
    else:
        print(f"Archivo NO encontrado para: {expected_doc_normalized}")
        print("Archivos disponibles:")
        for txt_file in txt_files[:10]:
            print(f"  - {txt_file.stem}")
    
    # 6. Verificar estructura de directorios
    print(f"\n=== ESTRUCTURA DE DIRECTORIOS ===")
    print(f"Directorio actual: {Path.cwd()}")
    print(f"PNTs existe: {pnts_dir.exists()}")
    print(f"Benchmark existe: {benchmark_file.exists()}")
    
    # 7. Mostrar algunos nombres de archivos PNTs
    print(f"\n=== NOMBRES DE ARCHIVOS PNTs ===")
    for txt_file in txt_files[:10]:
        print(f"  {txt_file.stem}")
    
    # 8. Verificar si hay problemas de codificación
    print(f"\n=== PRUEBA DE CODIFICACIÓN ===")
    try:
        with open(txt_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Archivo leído correctamente con UTF-8")
        print(f"Primeros caracteres: {repr(content[:50])}")
    except Exception as e:
        print(f"Error leyendo archivo: {e}")

if __name__ == "__main__":
    debug_benchmark()
