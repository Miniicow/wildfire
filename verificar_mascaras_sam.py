"""
verificar_mascaras_sam.py

Genera una cuadrícula de comparación (imagen original + caja original +
máscara SAM superpuesta) para un conjunto de muestras, permitiendo
verificar visualmente la calidad de las pseudo-máscaras antes de usarlas
en el entrenamiento del modelo de segmentación.

Autor: FireQuant Project
"""

from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualizar_muestras(
    directorio_dataset: str,
    n_muestras: int = 9,
    columnas: int = 3,
) -> None:
    """
    Muestra una cuadrícula con imágenes del dataset generado y su máscara
    de segmentación superpuesta en rojo semitransparente, para inspección
    visual rápida de la calidad de las pseudo-máscaras de SAM.

    Args:
        directorio_dataset: Ruta raíz del dataset YOLO-seg generado
            (debe contener subcarpetas 'images' y 'labels').
        n_muestras: Número de imágenes a mostrar.
        columnas: Número de columnas en la cuadrícula de visualización.

    Raises:
        FileNotFoundError: Si no se encuentran imágenes en el dataset.
    """
    ruta_imagenes = Path(directorio_dataset) / "images"
    ruta_labels = Path(directorio_dataset) / "labels"

    archivos = sorted(ruta_imagenes.glob("*.jpg"))[:n_muestras]
    if not archivos:
        raise FileNotFoundError(f"No se encontraron imágenes en: {ruta_imagenes}")

    filas = (len(archivos) + columnas - 1) // columnas
    _, ejes = plt.subplots(filas, columnas, figsize=(columnas * 5, filas * 5))
    ejes = np.array(ejes).reshape(-1)

    for idx, ruta_img in enumerate(archivos):
        imagen = cv2.cvtColor(cv2.imread(str(ruta_img)), cv2.COLOR_BGR2RGB)
        alto, ancho = imagen.shape[:2]

        ruta_label = ruta_labels / f"{ruta_img.stem}.txt"
        overlay = imagen.copy()

        if ruta_label.is_file():
            for linea in ruta_label.read_text().strip().splitlines():
                partes = list(map(float, linea.strip().split()[1:]))
                puntos = np.array(
                    [[int(partes[i] * ancho), int(partes[i + 1] * alto)]
                     for i in range(0, len(partes), 2)],
                    dtype=np.int32,
                )
                cv2.fillPoly(overlay, [puntos], color=(255, 0, 0))

        combinada = cv2.addWeighted(imagen, 0.6, overlay, 0.4, 0)
        ejes[idx].imshow(combinada)
        ejes[idx].set_title(ruta_img.stem, fontsize=9)
        ejes[idx].axis("off")

    for j in range(len(archivos), len(ejes)):
        ejes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualizar_muestras("/content/dfire_yolo_seg", n_muestras=9)
