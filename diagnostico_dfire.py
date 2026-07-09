"""
diagnostico_dfire.py

Diagnóstico rápido: inspecciona una muestra de imágenes y sus archivos de
etiqueta para verificar por qué el pipeline SAM no está encontrando cajas
de la clase 'fire'.
"""

from pathlib import Path


def diagnosticar(directorio_dfire: str, n_muestras: int = 5) -> None:
    """
    Imprime el contenido crudo de los primeros N archivos de etiqueta
    encontrados en la carpeta train/labels, junto con la ruta de imagen
    correspondiente, para verificar manualmente el formato y el mapeo
    de rutas.

    Args:
        directorio_dfire: Ruta raíz del dataset D-Fire (ej. la ruta de
            kagglehub, que contiene 'data/train/images' y 'data/train/labels').
        n_muestras: Número de archivos de etiqueta a inspeccionar.
    """
    raiz = Path(directorio_dfire)
    ruta_labels_train = raiz / "data" / "train" / "labels"
    ruta_imagenes_train = raiz / "data" / "train" / "images"

    print(f"¿Existe carpeta de labels?: {ruta_labels_train.is_dir()} -> {ruta_labels_train}")
    print(f"¿Existe carpeta de imágenes?: {ruta_imagenes_train.is_dir()} -> {ruta_imagenes_train}")

    archivos_label = sorted(ruta_labels_train.glob("*.txt"))[:n_muestras]
    print(f"\nTotal de archivos .txt en labels/: {len(list(ruta_labels_train.glob('*.txt')))}")
    print(f"Mostrando los primeros {len(archivos_label)}:\n")

    for archivo in archivos_label:
        contenido = archivo.read_text().strip()
        nombre_base = archivo.stem

        # Busca la imagen correspondiente con cualquier extensión común.
        posibles_imagenes = list(ruta_imagenes_train.glob(f"{nombre_base}.*"))

        print(f"--- {archivo.name} ---")
        print(f"Contenido crudo:\n{contenido if contenido else '(archivo vacío)'}")
        print(f"Imagen correspondiente encontrada: {posibles_imagenes}")
        print()


if __name__ == "__main__":
    diagnosticar("/kaggle/input/smoke-fire-detection-yolo", n_muestras=5)
