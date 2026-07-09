"""
download_dfire.py

Descarga el dataset D-Fire (imágenes + anotaciones de detección en formato
YOLO) desde su versión espejo en Kaggle, para su posterior conversión a
segmentación mediante SAM (Segment Anything Model).

NOTA: el repositorio oficial de GitHub (gaia-solutions-on-demand/DFireDataset)
solo contiene código y documentación; las imágenes reales se alojan en
OneDrive o en este espejo de Kaggle, mucho más simple de descargar en Colab.

Convención de clases en D-Fire (confirmada por la documentación del dataset):
    class 0 = smoke
    class 1 = fire
"""

from pathlib import Path


CLASE_FUEGO_DFIRE = 1  # class id de 'fire' en las anotaciones originales de D-Fire


def descargar_dfire(destino: str = None) -> str:
    """
    Descarga la versión espejo de D-Fire alojada en Kaggle
    ('sayedgamal99/smoke-fire-detection-yolo'), lista para usar sin
    necesidad de OneDrive.

    Args:
        destino: Ignorado si se usa kagglehub (kagglehub gestiona su
            propia caché local). Se mantiene por compatibilidad de firma.

    Returns:
        Ruta local donde kagglehub almacenó el dataset descargado.

    Raises:
        RuntimeError: Si la descarga falla (sin conexión, dataset movido,
            o falta configurar credenciales de Kaggle).
    """
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "kagglehub no está instalado. Ejecuta: !pip install -q kagglehub"
        )

    try:
        ruta = kagglehub.dataset_download("sayedgamal99/smoke-fire-detection-yolo")
    except Exception as exc:
        raise RuntimeError(f"Error al descargar D-Fire desde Kaggle: {exc}") from exc

    print(f"D-Fire (espejo Kaggle) descargado en: {ruta}")
    return ruta


def contar_imagenes_con_fuego(directorio_dataset: str) -> int:
    """
    Cuenta cuántas imágenes tienen al menos una anotación de clase 'fire'
    (class_id == 1), que son las únicas relevantes para nuestro pipeline
    de segmentación de llamas.

    Args:
        directorio_dataset: Ruta raíz del dataset D-Fire descargado.

    Returns:
        Número de imágenes con al menos una instancia de fuego anotada.
    """
    raiz = Path(directorio_dataset)
    archivos_label = list(raiz.rglob("*.txt"))

    conteo = 0
    for archivo in archivos_label:
        try:
            contenido = archivo.read_text().strip().splitlines()
        except Exception:
            continue
        if any(linea.strip().startswith(f"{CLASE_FUEGO_DFIRE} ") for linea in contenido):
            conteo += 1

    print(f"Imágenes con al menos una instancia de 'fire': {conteo}")
    return conteo


if __name__ == "__main__":
    ruta = descargar_dfire()
    contar_imagenes_con_fuego(ruta)
