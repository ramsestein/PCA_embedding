#!/usr/bin/env python3
"""
Script para combinar múltiples archivos .txt en un único archivo grande
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

def combinar_archivos_txt(directorio=".", patron="*.txt", archivo_salida="resultados_exp1.txt", 
                         agregar_separador=True, encoding="utf-8", verbose=True):
    """
    Combina múltiples archivos de texto en uno solo.
    
    Args:
        directorio: Directorio donde buscar los archivos
        patron: Patrón de búsqueda (por defecto *.txt)
        archivo_salida: Nombre del archivo combinado
        agregar_separador: Si agregar separadores entre archivos
        encoding: Codificación de los archivos
        verbose: Mostrar información del proceso
    """
    
    # Convertir a Path
    dir_path = Path(directorio)
    
    # Verificar que el directorio existe
    if not dir_path.exists():
        print(f"Error: El directorio '{directorio}' no existe.")
        return False
    
    # Buscar archivos que coincidan con el patrón
    archivos = list(dir_path.glob(patron))
    
    # Filtrar el archivo de salida si existe en la lista
    archivos = [f for f in archivos if f.name != archivo_salida]
    
    if not archivos:
        print(f"No se encontraron archivos con el patrón '{patron}' en '{directorio}'")
        return False
    
    # Ordenar archivos por nombre
    archivos.sort()
    
    if verbose:
        print(f"Se encontraron {len(archivos)} archivos para combinar:")
        for archivo in archivos:
            print(f"  - {archivo.name}")
        print()
    
    # Crear el archivo combinado
    archivo_salida_path = dir_path / archivo_salida
    total_lineas = 0
    archivos_procesados = 0
    
    try:
        with open(archivo_salida_path, 'w', encoding=encoding) as salida:
            for i, archivo in enumerate(archivos):
                try:
                    if verbose:
                        print(f"Procesando: {archivo.name}...", end="")
                    
                    # Agregar separador si está habilitado
                    if agregar_separador and i > 0:
                        salida.write("\n" + "="*80 + "\n")
                        salida.write(f"=== ARCHIVO: {archivo.name} ===\n")
                        salida.write("="*80 + "\n\n")
                    elif agregar_separador and i == 0:
                        salida.write("="*80 + "\n")
                        salida.write(f"=== ARCHIVO: {archivo.name} ===\n")
                        salida.write("="*80 + "\n\n")
                    
                    # Leer y escribir el contenido
                    with open(archivo, 'r', encoding=encoding) as entrada:
                        contenido = entrada.read()
                        salida.write(contenido)
                        
                        # Asegurar que hay un salto de línea al final
                        if contenido and not contenido.endswith('\n'):
                            salida.write('\n')
                        
                        lineas = contenido.count('\n')
                        total_lineas += lineas
                        
                    archivos_procesados += 1
                    
                    if verbose:
                        print(" ✓")
                        
                except Exception as e:
                    if verbose:
                        print(f" ✗ Error: {str(e)}")
                    continue
            
            # Agregar información al final si está habilitado el separador
            if agregar_separador:
                salida.write("\n" + "="*80 + "\n")
                salida.write(f"=== COMBINACIÓN COMPLETADA ===\n")
                salida.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                salida.write(f"Total de archivos: {archivos_procesados}\n")
                salida.write(f"Total de líneas: {total_lineas}\n")
                salida.write("="*80 + "\n")
        
        if verbose:
            print(f"\n✓ Combinación completada exitosamente!")
            print(f"  - Archivos procesados: {archivos_procesados}/{len(archivos)}")
            print(f"  - Total de líneas: {total_lineas}")
            print(f"  - Archivo de salida: {archivo_salida_path}")
            print(f"  - Tamaño: {archivo_salida_path.stat().st_size:,} bytes")
        
        return True
        
    except Exception as e:
        print(f"\nError al crear el archivo de salida: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Combina múltiples archivos de texto en uno solo.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python combinar_txt.py                     # Combina todos los .txt del directorio actual
  python combinar_txt.py -d ./documentos     # Combina archivos de un directorio específico
  python combinar_txt.py -p "*.log"          # Combina archivos .log en vez de .txt
  python combinar_txt.py -o resultado.txt    # Especifica el nombre del archivo de salida
  python combinar_txt.py --no-separador      # Sin separadores entre archivos
  python combinar_txt.py -r                  # Busca recursivamente en subdirectorios
        """
    )
    
    parser.add_argument('-d', '--directorio', 
                       default='.',
                       help='Directorio donde buscar los archivos (por defecto: directorio actual)')
    
    parser.add_argument('-p', '--patron', 
                       default='*.txt',
                       help='Patrón de búsqueda de archivos (por defecto: *.txt)')
    
    parser.add_argument('-o', '--salida', 
                       default='resultados_exp1.txt',
                       help='Nombre del archivo de salida (por defecto: combinado.txt)')
    
    parser.add_argument('--no-separador', 
                       action='store_true',
                       help='No agregar separadores entre archivos')
    
    parser.add_argument('-e', '--encoding', 
                       default='utf-8',
                       help='Codificación de los archivos (por defecto: utf-8)')
    
    parser.add_argument('-q', '--quiet', 
                       action='store_true',
                       help='Modo silencioso, sin mensajes informativos')
    
    parser.add_argument('-r', '--recursivo', 
                       action='store_true',
                       help='Buscar archivos recursivamente en subdirectorios')
    
    args = parser.parse_args()
    
    # Ajustar el patrón para búsqueda recursiva
    patron = args.patron
    if args.recursivo:
        patron = '**/' + patron
    
    # Ejecutar la combinación
    exito = combinar_archivos_txt(
        directorio=args.directorio,
        patron=patron,
        archivo_salida=args.salida,
        agregar_separador=not args.no_separador,
        encoding=args.encoding,
        verbose=not args.quiet
    )
    
    # Retornar código de salida apropiado
    sys.exit(0 if exito else 1)

if __name__ == "__main__":
    main()