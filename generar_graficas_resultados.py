"""
generar_graficas_resultados.py

Genera gráficas de resultados para la presentación/paper de FireQuant:
1. Evolución de métricas de entrenamiento por época (a partir de results.csv
   que Ultralytics guarda automáticamente).
2. Fire Percentage por imagen de prueba, coloreado por nivel de severidad.

Autor: FireQuant Project
"""

from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd


COLORES_SEVERIDAD = {
    "Low": "#2e7d32",
    "Medium": "#f9a825",
    "High": "#ef6c00",
    "Critical": "#b71c1c",
}


def graficar_evolucion_entrenamiento(
    ruta_run: str,
    salida_png: str = "evolucion_entrenamiento.png",
) -> None:
    """
    Grafica la evolución de mAP50 (caja y máscara) y las pérdidas
    principales a lo largo de las épocas, a partir del results.csv
    generado automáticamente por Ultralytics.

    Args:
        ruta_run: Ruta a la carpeta del run de entrenamiento (ej.
            '.../FireQuant_runs/yolo26n_seg_v2_combinado').
        salida_png: Nombre del archivo de imagen a generar.

    Raises:
        FileNotFoundError: Si no se encuentra results.csv en la ruta.
    """
    ruta_csv = Path(ruta_run) / "results.csv"
    if not ruta_csv.is_file():
        raise FileNotFoundError(f"No se encontró results.csv en: {ruta_run}")

    df = pd.read_csv(ruta_csv)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Gráfica 1: mAP50 de caja y máscara por época ---
    col_map_box = [c for c in df.columns if "mAP50(B)" in c or "mAP50(B" in c]
    col_map_mask = [c for c in df.columns if "mAP50(M)" in c or "mAP50(M" in c]

    if col_map_box:
        axes[0].plot(df["epoch"], df[col_map_box[0]], label="mAP50 (Caja)", color="#1565c0")
    if col_map_mask:
        axes[0].plot(df["epoch"], df[col_map_mask[0]], label="mAP50 (Máscara)", color="#c62828")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("mAP50")
    axes[0].set_title("Evolución de mAP50 durante el entrenamiento")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Gráfica 2: pérdidas principales ---
    col_box_loss = [c for c in df.columns if "box_loss" in c and "train" in c]
    col_seg_loss = [c for c in df.columns if "seg_loss" in c and "train" in c]
    col_cls_loss = [c for c in df.columns if "cls_loss" in c and "train" in c]

    if col_box_loss:
        axes[1].plot(df["epoch"], df[col_box_loss[0]], label="Box Loss", color="#2e7d32")
    if col_seg_loss:
        axes[1].plot(df["epoch"], df[col_seg_loss[0]], label="Seg Loss", color="#6a1b9a")
    if col_cls_loss:
        axes[1].plot(df["epoch"], df[col_cls_loss[0]], label="Cls Loss", color="#e65100")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Pérdida (Loss)")
    axes[1].set_title("Evolución de las pérdidas de entrenamiento")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(salida_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Gráfica guardada en: {salida_png}")


def graficar_fire_percentage_por_imagen(
    resumen: List[Dict],
    salida_png: str = "fire_percentage_por_imagen.png",
) -> None:
    """
    Genera un gráfico de barras del Fire Percentage por imagen de prueba,
    coloreado según el nivel de severidad asignado.

    Args:
        resumen: Lista de diccionarios con resultados por imagen (la salida
            de `procesar_lote()` en procesar_lote_prueba.py). Cada elemento
            debe tener las claves 'imagen', 'fire_percentage', 'severidad'.
        salida_png: Nombre del archivo de imagen a generar.

    Raises:
        ValueError: Si `resumen` está vacío.
    """
    if not resumen:
        raise ValueError("El resumen de resultados está vacío.")

    nombres = [r["imagen"] for r in resumen]
    valores_fp = [r["fire_percentage"] for r in resumen]
    colores = [COLORES_SEVERIDAD.get(r["severidad"], "#757575") for r in resumen]

    fig, ax = plt.subplots(figsize=(12, 6))
    barras = ax.bar(nombres, valores_fp, color=colores)

    for barra, r in zip(barras, resumen):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.3,
            f"{r['fire_percentage']}%\n{r['severidad']}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Imagen de prueba")
    ax.set_ylabel("Fire Percentage (%)")
    ax.set_title("Fire Percentage y Nivel de Severidad por Imagen de Prueba")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(salida_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Gráfica guardada en: {salida_png}")


def imprimir_tabla_metricas_finales(ruta_run: str) -> None:
    """
    Extrae e imprime en formato de tabla limpia las métricas de la última
    época registrada en results.csv, listas para copiar a un documento.

    Args:
        ruta_run: Ruta a la carpeta del run de entrenamiento.

    Raises:
        FileNotFoundError: Si no se encuentra results.csv en la ruta.
    """
    ruta_csv = Path(ruta_run) / "results.csv"
    if not ruta_csv.is_file():
        raise FileNotFoundError(f"No se encontró results.csv en: {ruta_run}")

    df = pd.read_csv(ruta_csv)
    df.columns = [c.strip() for c in df.columns]
    ultima_fila = df.iloc[-1]

    print("=== Métricas finales del entrenamiento ===")
    for columna in df.columns:
        if "metrics" in columna or "loss" in columna:
            print(f"{columna:<30}: {ultima_fila[columna]:.4f}")


if __name__ == "__main__":
    RUTA_RUN = "/content/drive/MyDrive/FireQuant_runs/yolo26n_seg_v2_combinado"

    graficar_evolucion_entrenamiento(RUTA_RUN)
    imprimir_tabla_metricas_finales(RUTA_RUN)

    # Para la gráfica de FP por imagen, usa el 'resumen' que ya generaste
    # con procesar_lote_prueba.py:
    #
    # from procesar_lote_prueba import procesar_lote
    # resumen = procesar_lote(...)
    # graficar_fire_percentage_por_imagen(resumen)
