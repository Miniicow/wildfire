"""
procesar_lote_prueba.py

Procesa un lote completo de imágenes de prueba con el modelo FireQuant
entrenado, generando una tabla resumen (FP, nivel, alerta) y guardando
las imágenes anotadas — listo para usar en la presentación/entrega.

Autor: FireQuant Project
"""

from pathlib import Path

from inference_firequant import FireQuantEstimator


def procesar_lote(
    weights_path: str,
    carpeta_imagenes: str,
    carpeta_salida_anotadas: str,
    conf_threshold: float = 0.35,
) -> list:
    """
    Corre el pipeline completo (segmentación + FP + severidad + alerta)
    sobre todas las imágenes de una carpeta, guardando las versiones
    anotadas y retornando un resumen tabular.

    Args:
        weights_path: Ruta al checkpoint entrenado (best.pt).
        carpeta_imagenes: Carpeta con las imágenes de prueba.
        carpeta_salida_anotadas: Carpeta donde se guardan las imágenes
            con la máscara y el texto de FP/nivel superpuestos.
        conf_threshold: Umbral de confianza para la inferencia.

    Returns:
        Lista de diccionarios con los resultados de cada imagen.

    Raises:
        FileNotFoundError: Si la carpeta de imágenes no existe o está vacía.
    """
    carpeta = Path(carpeta_imagenes)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta: {carpeta_imagenes}")

    imagenes = sorted([
        f for f in carpeta.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    if not imagenes:
        raise FileNotFoundError(f"No se encontraron imágenes en: {carpeta_imagenes}")

    Path(carpeta_salida_anotadas).mkdir(parents=True, exist_ok=True)

    estimador = FireQuantEstimator(weights_path=weights_path, conf_threshold=conf_threshold)

    resumen = []
    print(f"{'Imagen':<25}{'FP (%)':<10}{'Nivel':<12}{'Instancias':<12}")
    print("-" * 60)

    for ruta_img in imagenes:
        resultado = estimador.estimar(str(ruta_img))

        salida_anotada = str(Path(carpeta_salida_anotadas) / f"anotada_{ruta_img.name}")
        estimador.visualizar(str(ruta_img), resultado, salida_path=salida_anotada)

        print(f"{ruta_img.name:<25}{resultado.fire_percentage:<10}{resultado.severity_level:<12}{resultado.num_instances:<12}")

        resumen.append({
            "imagen": ruta_img.name,
            "fire_percentage": resultado.fire_percentage,
            "severidad": resultado.severity_level,
            "instancias": resultado.num_instances,
            "confianza_promedio": resultado.mean_confidence,
            "mensaje_alerta": estimador.generar_mensaje_alerta(resultado),
        })

    print(f"\nImágenes anotadas guardadas en: {carpeta_salida_anotadas}")
    return resumen


if __name__ == "__main__":
    resumen = procesar_lote(
        weights_path="/content/drive/MyDrive/FireQuant_runs/yolo26n_seg_v2_combinado/weights/best.pt",
        carpeta_imagenes="/content/imagenes_prueba",
        carpeta_salida_anotadas="/content/imagenes_prueba_anotadas",
    )

    print("\n--- Mensajes de alerta completos ---")
    for r in resumen:
        print(f"\n{r['imagen']}:")
        print(r["mensaje_alerta"])
