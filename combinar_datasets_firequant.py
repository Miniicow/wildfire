"""
combinar_datasets_firequant.py

Combina el dataset original de Roboflow (210 imágenes, máscaras anotadas
manualmente) con el dataset D-Fire procesado vía SAM (5822 imágenes,
pseudo-máscaras), generando una estructura final train/valid/test lista
para entrenar YOLOv26n-seg.

Estrategia de partición:
    - Se mantiene la partición original de Roboflow tal cual (70/20/10),
      ya que sus máscaras son de mayor confianza (anotación manual).
    - D-Fire se particiona con las mismas proporciones (70/20/10) de forma
      aleatoria, ya que sus pseudo-máscaras son de confianza media (SAM).

"""

import random
import shutil
from pathlib import Path
from typing import List, Tuple


def _listar_pares_yolo_seg(directorio: str) -> List[Tuple[Path, Path]]:
    """
    Lista todos los pares (imagen, label) válidos dentro de una carpeta
    con subcarpetas 'images' y 'labels'.

    Args:
        directorio: Ruta raíz que contiene 'images/' y 'labels/'.

    Returns:
        Lista de tuplas (ruta_imagen, ruta_label) para pares donde ambos
        archivos existen.

    Raises:
        FileNotFoundError: Si no existen las subcarpetas esperadas.
    """
    ruta_imagenes = Path(directorio) / "images"
    ruta_labels = Path(directorio) / "labels"

    if not ruta_imagenes.is_dir() or not ruta_labels.is_dir():
        raise FileNotFoundError(
            f"Se esperaban las carpetas 'images/' y 'labels/' dentro de: {directorio}"
        )

    pares = []
    for img in sorted(ruta_imagenes.glob("*.*")):
        label = ruta_labels / f"{img.stem}.txt"
        if label.is_file():
            pares.append((img, label))

    return pares


def _copiar_pares(
    pares: List[Tuple[Path, Path]],
    destino_images: Path,
    destino_labels: Path,
    prefijo: str = "",
) -> None:
    """
    Copia una lista de pares (imagen, label) a las carpetas destino,
    opcionalmente agregando un prefijo al nombre de archivo para evitar
    colisiones entre fuentes distintas (ej. 'dfire_' vs 'roboflow_').

    Args:
        pares: Lista de tuplas (ruta_imagen, ruta_label).
        destino_images: Carpeta destino para las imágenes.
        destino_labels: Carpeta destino para las etiquetas.
        prefijo: Prefijo a anteponer al nombre de archivo.
    """
    destino_images.mkdir(parents=True, exist_ok=True)
    destino_labels.mkdir(parents=True, exist_ok=True)

    for img, label in pares:
        nombre_nuevo = f"{prefijo}{img.stem}"
        shutil.copy(img, destino_images / f"{nombre_nuevo}{img.suffix}")
        shutil.copy(label, destino_labels / f"{nombre_nuevo}.txt")


def combinar_datasets(
    directorio_roboflow: str,
    directorio_dfire_sam: str,
    directorio_salida: str,
    split_ratios_dfire: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
) -> None:
    """
    Combina el dataset de Roboflow (ya particionado en train/valid/test)
    con el dataset D-Fire+SAM (sin particionar), generando la estructura
    final combinada lista para Ultralytics.

    Args:
        directorio_roboflow: Ruta raíz del dataset de Roboflow, con
            subcarpetas train/, valid/ (o val/), test/, cada una con
            images/ y labels/.
        directorio_dfire_sam: Ruta raíz del dataset D-Fire procesado
            (subcarpetas 'images/' y 'labels/' sin partición).
        directorio_salida: Ruta donde se construirá el dataset combinado
            final (train/valid/test).
        split_ratios_dfire: Proporciones (train, val, test) para partir
            D-Fire aleatoriamente. Deben sumar 1.0.
        seed: Semilla para la partición aleatoria reproducible.

    Raises:
        ValueError: Si split_ratios_dfire no suma 1.0.
        FileNotFoundError: Si alguna de las rutas de entrada no existe.
    """
    if abs(sum(split_ratios_dfire) - 1.0) > 1e-6:
        raise ValueError("split_ratios_dfire debe sumar 1.0")

    # --- 1. Copiar Roboflow tal cual, respetando su partición original ---
    nombres_split_roboflow = {"train": "train", "valid": "valid", "test": "test"}
    # Roboflow a veces usa 'val' en vez de 'valid'; se detecta automáticamente.
    raiz_roboflow = Path(directorio_roboflow)
    if (raiz_roboflow / "val").is_dir() and not (raiz_roboflow / "valid").is_dir():
        nombres_split_roboflow["valid"] = "val"

    total_roboflow = 0
    for split_salida, split_origen in nombres_split_roboflow.items():
        origen = raiz_roboflow / split_origen
        if not origen.is_dir():
            print(f"[AVISO] No se encontró la carpeta '{split_origen}' en Roboflow, se omite.")
            continue
        pares = _listar_pares_yolo_seg(str(origen))
        _copiar_pares(
            pares,
            Path(directorio_salida) / split_salida / "images",
            Path(directorio_salida) / split_salida / "labels",
            prefijo="rf_",
        )
        total_roboflow += len(pares)
        print(f"Roboflow -> {split_salida}: {len(pares)} pares copiados")

    # --- 2. Particionar D-Fire+SAM aleatoriamente y copiar ---
    pares_dfire = _listar_pares_yolo_seg(directorio_dfire_sam)
    random.seed(seed)
    random.shuffle(pares_dfire)

    n_total = len(pares_dfire)
    n_train = int(n_total * split_ratios_dfire[0])
    n_val = int(n_total * split_ratios_dfire[1])

    splits_dfire = {
        "train": pares_dfire[:n_train],
        "valid": pares_dfire[n_train:n_train + n_val],
        "test": pares_dfire[n_train + n_val:],
    }

    total_dfire = 0
    for split_salida, pares in splits_dfire.items():
        _copiar_pares(
            pares,
            Path(directorio_salida) / split_salida / "images",
            Path(directorio_salida) / split_salida / "labels",
            prefijo="dfire_",
        )
        total_dfire += len(pares)
        print(f"D-Fire+SAM -> {split_salida}: {len(pares)} pares copiados")

    # --- 3. Generar data.yaml final ---
    yaml_contenido = (
        f"train: {directorio_salida}/train/images\n"
        f"val: {directorio_salida}/valid/images\n"
        f"test: {directorio_salida}/test/images\n"
        f"nc: 1\n"
        f"names: ['fire']\n"
    )
    (Path(directorio_salida) / "data.yaml").write_text(yaml_contenido)

    print(f"\nDataset combinado generado en: {directorio_salida}")
    print(f"Total Roboflow: {total_roboflow} | Total D-Fire+SAM: {total_dfire} | "
          f"Total combinado: {total_roboflow + total_dfire}")
    print(f"data.yaml generado en: {directorio_salida}/data.yaml")


if __name__ == "__main__":
    combinar_datasets(
        directorio_roboflow="/content/Fire-and-Smoke-Segmentation-1",  # ajusta a tu ruta real
        directorio_dfire_sam="/content/drive/MyDrive/dfire_yolo_seg",
        directorio_salida="/content/firequant_dataset_v2",
    )
