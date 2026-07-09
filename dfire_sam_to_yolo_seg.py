"""
dfire_sam_to_yolo_seg.py

Convierte las anotaciones de bounding box del dataset D-Fire (clase 'fire')
en máscaras de segmentación pixel-level utilizando el Segment Anything
Model (SAM), y exporta el resultado en formato YOLO-segmentation.

Metodología: para cada caja delimitadora de 'fire', se usa la caja como
prompt geométrico para SAM, que retorna la máscara del objeto contenido
en esa región. Este enfoque es el mismo empleado en el dataset "Boreal
Forest Fire" (Pesonen et al., 2025) para generar máscaras de humo a
partir de cajas humanas.

Autor: FireQuant Project
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


CLASE_FUEGO_DFIRE = 1  # class_id de 'fire' en las etiquetas originales de D-Fire
CLASE_FUEGO_SALIDA = 0  # class_id que usaremos en el dataset YOLO-seg final
AREA_MINIMA_CONTORNO = 20


def cargar_modelo_sam(
    checkpoint_path: str = "/content/sam_vit_b_01ec64.pth",
    tipo_modelo: str = "vit_b",
) -> SamPredictor:
    """
    Carga el modelo SAM preentrenado y retorna un SamPredictor listo
    para generar máscaras a partir de prompts (cajas o puntos).

    Args:
        checkpoint_path: Ruta al checkpoint .pth de SAM. Si no existe,
            debe descargarse previamente (ver instrucciones de uso al
            final del archivo).
        tipo_modelo: Variante de SAM ('vit_b', 'vit_l', 'vit_h').
            'vit_b' es la más liviana, recomendada para Colab gratuito.

    Returns:
        Instancia de SamPredictor con el modelo cargado en GPU si está
        disponible.

    Raises:
        FileNotFoundError: Si el checkpoint no existe en la ruta indicada.
    """
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"No se encontró el checkpoint de SAM en: {checkpoint_path}\n"
            "Descárgalo con:\n"
            "!wget https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_b_01ec64.pth -P /content/"
        )

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[tipo_modelo](checkpoint=checkpoint_path)
    sam.to(device=dispositivo)

    print(f"SAM ({tipo_modelo}) cargado en: {dispositivo}")
    return SamPredictor(sam)


def leer_cajas_fuego_yolo(
    ruta_label: str,
    ancho_img: int,
    alto_img: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Lee un archivo de anotación YOLO (detección) y retorna las cajas de
    la clase 'fire' convertidas a coordenadas de píxel (x1, y1, x2, y2).

    Args:
        ruta_label: Ruta al archivo .txt de anotación YOLO.
        ancho_img: Ancho de la imagen en píxeles.
        alto_img: Alto de la imagen en píxeles.

    Returns:
        Lista de cajas (x1, y1, x2, y2) en píxeles, solo de la clase fuego.

    Raises:
        FileNotFoundError: Si el archivo de etiquetas no existe.
    """
    if not Path(ruta_label).is_file():
        raise FileNotFoundError(f"No se encontró el archivo de anotación: {ruta_label}")

    cajas = []
    for linea in Path(ruta_label).read_text().strip().splitlines():
        partes = linea.strip().split()
        if not partes:
            continue
        class_id = int(partes[0])
        if class_id != CLASE_FUEGO_DFIRE:
            continue

        x_centro, y_centro, ancho_rel, alto_rel = map(float, partes[1:5])
        x1 = int((x_centro - ancho_rel / 2) * ancho_img)
        y1 = int((y_centro - alto_rel / 2) * alto_img)
        x2 = int((x_centro + ancho_rel / 2) * ancho_img)
        y2 = int((y_centro + alto_rel / 2) * alto_img)
        cajas.append((max(x1, 0), max(y1, 0), min(x2, ancho_img), min(y2, alto_img)))

    return cajas


def generar_mascara_con_sam(
    predictor: SamPredictor,
    imagen_rgb: np.ndarray,
    cajas: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Genera una máscara binaria combinada usando SAM, guiado por una o
    varias cajas delimitadoras de fuego.

    Args:
        predictor: Instancia de SamPredictor con la imagen ya lista para
            procesar (el método internamente llama a set_image).
        imagen_rgb: Imagen en formato RGB (no BGR) como array de NumPy.
        cajas: Lista de cajas (x1, y1, x2, y2) en píxeles.

    Returns:
        Máscara binaria (uint8, 0/255) del mismo tamaño que la imagen,
        combinando todas las instancias detectadas.

    Raises:
        RuntimeError: Si SAM falla al generar alguna máscara.
    """
    alto, ancho = imagen_rgb.shape[:2]
    mascara_combinada = np.zeros((alto, ancho), dtype=np.uint8)

    predictor.set_image(imagen_rgb)

    for caja in cajas:
        caja_array = np.array(caja)
        try:
            mascaras, puntajes, _ = predictor.predict(
                box=caja_array[None, :],
                multimask_output=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Error al generar máscara SAM para la caja {caja}: {exc}") from exc

        mascara_instancia = mascaras[0].astype(np.uint8) * 255
        mascara_combinada = np.maximum(mascara_combinada, mascara_instancia)

    return mascara_combinada


def mascara_a_yolo_seg(mascara: np.ndarray, class_id: int = CLASE_FUEGO_SALIDA) -> List[str]:
    """
    Convierte una máscara binaria en líneas de anotación YOLO-segmentation.

    Args:
        mascara: Máscara binaria (0/255).
        class_id: ID de clase a asignar en la salida.

    Returns:
        Lista de líneas en formato "class_id x1 y1 x2 y2 ... xn yn"
        (coordenadas normalizadas).
    """
    alto, ancho = mascara.shape[:2]
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lineas = []
    for contorno in contornos:
        if cv2.contourArea(contorno) < AREA_MINIMA_CONTORNO:
            continue
        epsilon = 0.003 * cv2.arcLength(contorno, True)
        contorno_simplificado = cv2.approxPolyDP(contorno, epsilon, True)
        if len(contorno_simplificado) < 3:
            continue

        puntos_norm = []
        for punto in contorno_simplificado.reshape(-1, 2):
            puntos_norm.extend([f"{punto[0] / ancho:.6f}", f"{punto[1] / alto:.6f}"])
        lineas.append(f"{class_id} " + " ".join(puntos_norm))

    return lineas


def _resolver_ruta_label(ruta_img: Path) -> Path:
    """
    Determina la ruta del archivo de etiqueta (.txt) correspondiente a una
    imagen, soportando dos convenciones comunes de estructura de dataset:

    1. Label en la misma carpeta que la imagen (mismo nombre, .txt).
    2. Label en una carpeta hermana 'labels/' (convención YOLO estándar,
       donde las imágenes están en 'images/' y las etiquetas en 'labels/'
       con la misma subestructura de carpetas).

    Args:
        ruta_img: Ruta a la imagen.

    Returns:
        Ruta candidata al archivo .txt (puede no existir; el llamador
        debe verificar con `.is_file()`).
    """
    candidata_misma_carpeta = ruta_img.with_suffix(".txt")
    if candidata_misma_carpeta.is_file():
        return candidata_misma_carpeta

    # Busca reemplazando cualquier segmento 'images' por 'labels' en la ruta.
    partes = list(ruta_img.parts)
    partes_label = [
        "labels" if parte.lower() == "images" else parte
        for parte in partes
    ]
    candidata_carpeta_labels = Path(*partes_label).with_suffix(".txt")
    if candidata_carpeta_labels.is_file():
        return candidata_carpeta_labels

    return None


def procesar_dataset_dfire(
    directorio_dfire: str,
    directorio_salida: str,
    predictor: SamPredictor,
    max_imagenes: int = None,
) -> None:
    """
    Procesa todas las imágenes de D-Fire con al menos una instancia de
    fuego, generando sus pseudo-máscaras vía SAM y exportando el par
    imagen+anotación en formato YOLO-seg.

    Args:
        directorio_dfire: Ruta raíz del dataset D-Fire descargado.
        directorio_salida: Ruta donde se construirá el dataset YOLO-seg.
        predictor: Instancia de SamPredictor ya cargada.
        max_imagenes: Límite opcional de imágenes a procesar (útil para
            pruebas rápidas antes de correr sobre el dataset completo).

    Raises:
        FileNotFoundError: Si no se encuentran imágenes o etiquetas.
    """
    raiz = Path(directorio_dfire)
    archivos_imagen = sorted(raiz.rglob("*.jpg")) + sorted(raiz.rglob("*.png"))

    if not archivos_imagen:
        raise FileNotFoundError(f"No se encontraron imágenes en: {directorio_dfire}")

    # Filtra primero las imágenes que sí tienen al menos una caja de fuego,
    # ANTES de aplicar el límite de muestra. Sin este filtro, un límite bajo
    # (ej. 20) puede caer por completo en un tramo del dataset sin instancias
    # de fuego, dando 0 procesadas aunque el dataset global sí las tenga.
    archivos_con_fuego = []
    for ruta_img in archivos_imagen:
        ruta_label = _resolver_ruta_label(ruta_img)
        if ruta_label is None or not ruta_label.is_file():
            continue
        contenido = ruta_label.read_text().strip().splitlines()
        if any(l.strip().startswith(f"{CLASE_FUEGO_DFIRE} ") for l in contenido):
            archivos_con_fuego.append(ruta_img)
        if max_imagenes and len(archivos_con_fuego) >= max_imagenes:
            break  # ya tenemos suficientes candidatas, no hace falta seguir escaneando

    archivos_imagen = archivos_con_fuego

    procesadas, omitidas, ya_existian = 0, 0, 0

    for ruta_img in archivos_imagen:
        nombre_base = ruta_img.stem
        salida_img = Path(directorio_salida) / "images" / f"{nombre_base}.jpg"
        salida_label = Path(directorio_salida) / "labels" / f"{nombre_base}.txt"

        # Si ya se procesó en una corrida anterior (ej. tras una desconexión
        # de Colab), se salta para no repetir trabajo ni gastar tiempo de GPU.
        if salida_img.is_file() and salida_label.is_file():
            ya_existian += 1
            continue

        ruta_label = _resolver_ruta_label(ruta_img)

        imagen_bgr = cv2.imread(str(ruta_img))
        if imagen_bgr is None:
            omitidas += 1
            continue

        alto, ancho = imagen_bgr.shape[:2]
        cajas_fuego = leer_cajas_fuego_yolo(str(ruta_label), ancho, alto)

        if not cajas_fuego:
            omitidas += 1
            continue  # por seguridad; no debería ocurrir tras el pre-filtrado

        imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
        mascara = generar_mascara_con_sam(predictor, imagen_rgb, cajas_fuego)
        lineas_yolo = mascara_a_yolo_seg(mascara)

        if not lineas_yolo:
            omitidas += 1
            continue

        salida_img.parent.mkdir(parents=True, exist_ok=True)
        salida_label.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(salida_img), imagen_bgr)
        salida_label.write_text("\n".join(lineas_yolo))

        procesadas += 1
        if procesadas % 50 == 0:
            print(f"Procesadas: {procesadas} | Omitidas: {omitidas} | Ya existían: {ya_existian}")

    print(f"\nProceso finalizado. Total procesadas: {procesadas} | Omitidas: {omitidas} | "
          f"Ya existían (saltadas): {ya_existian}")
    print(f"Dataset generado en: {directorio_salida}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Instrucciones previas (ejecutar en Colab antes de este script):
    #
    # !pip install -q git+https://github.com/facebookresearch/segment-anything.git
    # !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P /content/
    # ------------------------------------------------------------------

    predictor_sam = cargar_modelo_sam(checkpoint_path="/content/sam_vit_b_01ec64.pth")

    # Prueba rápida con pocas imágenes antes de correr el dataset completo.
    procesar_dataset_dfire(
        directorio_dfire="/content/DFireDataset",
        directorio_salida="/content/dfire_yolo_seg",
        predictor=predictor_sam,
        max_imagenes=20,  # quitar este límite (o subirlo) tras validar resultados
    )
