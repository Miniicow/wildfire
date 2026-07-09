"""
diagnostico_dfire_v2.py

Replica exactamente la lógica de selección de archivos usada en
`procesar_dataset_dfire` para identificar por qué las primeras N imágenes
no están generando ninguna caja de fuego válida.
"""

from pathlib import Path


def diagnosticar_seleccion(directorio_dfire: str, n_muestras: int = 20) -> None:
    """
    Replica el listado de imágenes de `procesar_dataset_dfire` (mismo
    orden, mismo límite) e imprime, para cada una, si se encontró su
    label y cuántas cajas de clase 'fire' (class_id == 1) contiene.

    Args:
        directorio_dfire: Ruta raíz del dataset D-Fire.
        n_muestras: Número de imágenes a inspeccionar (mismo valor que
            se usó en `procesar_dataset_dfire`).
    """
    raiz = Path(directorio_dfire)
    archivos_imagen = sorted(raiz.rglob("*.jpg")) + sorted(raiz.rglob("*.png"))
    archivos_imagen = archivos_imagen[:n_muestras]

    print(f"Total de imágenes encontradas en todo el dataset: "
          f"{len(list(raiz.rglob('*.jpg'))) + len(list(raiz.rglob('*.png')))}")
    print(f"Mostrando las primeras {len(archivos_imagen)} (mismo orden que el pipeline):\n")

    for ruta_img in archivos_imagen:
        nombre_base = ruta_img.stem
        partes_label = [
            "labels" if p.lower() == "images" else p for p in ruta_img.parts
        ]
        ruta_label = Path(*partes_label).with_suffix(".txt")

        existe_label = ruta_label.is_file()
        n_cajas_fuego = 0
        if existe_label:
            contenido = ruta_label.read_text().strip().splitlines()
            n_cajas_fuego = sum(1 for l in contenido if l.strip().startswith("1 "))

        print(f"{ruta_img.relative_to(raiz)} | label existe: {existe_label} | cajas fuego: {n_cajas_fuego}")


if __name__ == "__main__":
    diagnosticar_seleccion("/kaggle/input/smoke-fire-detection-yolo", n_muestras=20)
