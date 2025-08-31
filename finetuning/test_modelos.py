#!/usr/bin/env python3
"""
test_rapido.py - Verifica que los modelos cargan correctamente
"""

from pathlib import Path
import numpy as np
from cargador_models import UnifiedEmbeddingAdapter

def test_modelos():
    """Test rápido de carga y funcionamiento"""
    
    print("🧪 TEST RÁPIDO DE MODELOS")
    print("="*50)
    
    # Configuración
    models = [
        ('all-MiniLM-L6-v2', './all-MiniLM-L6-v2'),
        ('RoBERTa-PNT', './roberta-pnt')
    ]
    
    test_text = "Protocolo de administración de medicamentos"
    
    for model_name, model_path in models:
        print(f"\n📦 Probando {model_name}...")
        print(f"   Path: {model_path}")
        
        if not Path(model_path).exists():
            print(f"   ❌ No se encuentra la carpeta")
            continue
            
        try:
            # Cargar modelo
            adapter = UnifiedEmbeddingAdapter(
                model_path=model_path,
                model_name=model_name,
                pooling_strategy='mean'
            )
            
            # Generar embedding
            embedding = adapter.embed(test_text)
            
            print(f"   ✅ Modelo cargado correctamente")
            print(f"   📏 Dimensión: {adapter.get_dimension()}")
            print(f"   📊 Shape embedding: {embedding.shape}")
            print(f"   🔢 Norma L2: {np.linalg.norm(embedding):.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Verificar otros archivos necesarios
    print("\n📁 Verificando archivos necesarios:")
    
    files_to_check = [
        ('Documentos PNT', './PNTs', 'dir'),
        ('Preguntas evaluación', './preguntas_con_docs.json', 'file')
    ]
    
    all_ok = True
    for name, path, tipo in files_to_check:
        path_obj = Path(path)
        exists = path_obj.exists()
        
        if tipo == 'dir' and exists:
            num_files = len(list(path_obj.glob('*.txt')))
            print(f"   {'✅' if exists else '❌'} {name}: {path} ({num_files} archivos .txt)")
        else:
            print(f"   {'✅' if exists else '❌'} {name}: {path}")
        
        if not exists:
            all_ok = False
    
    print("\n" + "="*50)
    if all_ok:
        print("✅ Todo listo para ejecutar la comparación!")
        print("\nEjecuta: python comparacion_modelos.py")
    else:
        print("⚠️  Faltan algunos archivos. Verifica los errores arriba.")


if __name__ == "__main__":
    test_modelos()