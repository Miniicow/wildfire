"""
webcam_demo_firequant.py

Demo en vivo de FireQuant usando la cámara del computador. Procesa cada
frame en tiempo real, calcula el Fire Percentage (FP), muestra el nivel
de severidad y la alerta correspondiente superpuestos en video, y al
finalizar genera la gráfica de evolución temporal (FP vs. Tiempo) —
aplicación directa del Objetivo Específico #4 del proyecto.

Controles durante la demo:
    - Presiona 'q' para salir y generar la gráfica final.
    - Apunta la cámara a una foto/video de fuego en otra pantalla (celular,
      otro monitor) para simular una escena de incendio.

"""

import platform
import threading
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

AUDIO_DISPONIBLE = platform.system() == "Windows"
if AUDIO_DISPONIBLE:
    import winsound


UMBRALES_SEVERIDAD = {
    "Low": (0.0, 3.0),
    "Medium": (3.0, 10.0),
    "High": (10.0, 25.0),
    "Critical": (25.0, 100.0),
}

COLORES_SEVERIDAD_BGR = {
    "Low": (0, 200, 0),
    "Medium": (0, 165, 255),
    "High": (0, 100, 255),
    "Critical": (0, 0, 200),
}


def reproducir_alarma_en_segundo_plano(frecuencia: int = 1000, duracion_ms: int = 400) -> None:
    """
    Reproduce un beep de alarma usando winsound (nativo de Windows, sin
    necesidad de instalar ninguna librería adicional), en un hilo separado
    para no bloquear el bucle principal de video.

    Args:
        frecuencia: Frecuencia del tono en Hz.
        duracion_ms: Duración del tono en milisegundos.
    """
    if not AUDIO_DISPONIBLE:
        return
    threading.Thread(target=winsound.Beep, args=(frecuencia, duracion_ms), daemon=True).start()


def generar_mensaje_alerta(nivel: str, fp: float) -> str:
    """
    Genera el mensaje de alerta textual correspondiente al nivel de
    severidad detectado.

    Args:
        nivel: Nivel de severidad ('Low', 'Medium', 'High', 'Critical').
        fp: Valor de Fire Percentage asociado.

    Returns:
        Mensaje de alerta formateado.
    """
    plantillas = {
        "Low": "NIVEL BAJO - Fuego detectado ({fp}% del área). Monitoreo rutinario.",
        "Medium": "NIVEL MEDIO - Fuego en expansión ({fp}% del área). Vigilancia activa.",
        "High": "ALERTA ALTA - Incendio significativo ({fp}% del área). Notificar emergencias.",
        "Critical": "ALERTA CRITICA - Incendio de gran magnitud ({fp}% del área). Protocolo de emergencia.",
    }
    return plantillas.get(nivel, "Nivel desconocido").format(fp=fp)


def clasificar_severidad(fire_percentage: float) -> str:
    """
    Asigna un nivel de severidad discreto según el Fire Percentage.

    Args:
        fire_percentage: Valor de FP en el rango [0, 100].

    Returns:
        Etiqueta de severidad: 'Low', 'Medium', 'High' o 'Critical'.
    """
    for nivel, (lim_inf, lim_sup) in UMBRALES_SEVERIDAD.items():
        if lim_inf <= fire_percentage < lim_sup:
            return nivel
    return "Critical"


def calcular_fp_y_mascara(
    resultado, alto: int, ancho: int
) -> Tuple[float, np.ndarray]:
    """
    Calcula el Fire Percentage y la máscara combinada a partir del
    resultado de inferencia de un solo frame.

    Args:
        resultado: Objeto Results de Ultralytics para un frame.
        alto: Alto del frame en píxeles.
        ancho: Ancho del frame en píxeles.

    Returns:
        Tupla (fire_percentage, mascara_binaria).
    """
    total_pixels = alto * ancho
    mascara_combinada = np.zeros((alto, ancho), dtype=np.uint8)

    if resultado.masks is not None:
        for m in resultado.masks.data.cpu().numpy():
            m_resized = cv2.resize(m.astype(np.uint8), (ancho, alto), interpolation=cv2.INTER_NEAREST)
            mascara_combinada = np.logical_or(mascara_combinada, m_resized).astype(np.uint8)

    fire_pixels = int(np.count_nonzero(mascara_combinada))
    fp = round((fire_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0.0
    return fp, mascara_combinada


def ejecutar_demo_webcam(
    weights_path: str,
    conf_threshold: float = 0.15,
    camara_id: int = 0,
    salida_grafica: str = "evolucion_fp_tiempo.png",
) -> List[Tuple[float, float]]:
    """
    Ejecuta la demo en vivo: abre la cámara, procesa cada frame con el
    modelo FireQuant, muestra overlay de máscara/FP/alerta, y registra
    el historial (tiempo, FP) para graficar al finalizar.

    Args:
        weights_path: Ruta al checkpoint entrenado (best.pt).
        conf_threshold: Umbral de confianza para la inferencia.
        camara_id: Índice de la cámara a usar (0 = cámara por defecto).
        salida_grafica: Ruta donde se guardará la gráfica final de FP vs. tiempo.

    Returns:
        Lista de tuplas (tiempo_segundos, fire_percentage) registradas durante la demo.

    Raises:
        FileNotFoundError: Si el checkpoint no existe.
        RuntimeError: Si no se puede abrir la cámara.
    """
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint en: {weights_path}")

    modelo = YOLO(weights_path)

    captura = cv2.VideoCapture(camara_id)
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con índice {camara_id}")

    historial: List[Tuple[float, float]] = []
    tiempo_inicio = time.time()
    ultimo_disparo_alarma = 0.0
    cooldown_alarma_seg = 2.0  # evita que la alarma suene sin parar cada frame

    if not AUDIO_DISPONIBLE:
        print("Aviso: alarma sonora no disponible en este sistema operativo "
              "(winsound solo funciona en Windows). La alarma será solo visual.")

    print("Demo en vivo iniciada. Presiona 'q' en la ventana de video para salir.")

    try:
        while True:
            ret, frame = captura.read()
            if not ret:
                print("No se pudo leer el frame de la cámara.")
                break

            alto, ancho = frame.shape[:2]
            resultado = modelo.predict(source=frame, conf=conf_threshold, verbose=False)[0]

            fp, mascara = calcular_fp_y_mascara(resultado, alto, ancho)
            nivel = clasificar_severidad(fp)
            color = COLORES_SEVERIDAD_BGR[nivel]

            tiempo_transcurrido = round(time.time() - tiempo_inicio, 2)
            historial.append((tiempo_transcurrido, fp))

            overlay = frame.copy()
            overlay[mascara == 1] = color
            frame_final = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            # --- Disparo de alarma (sonido + borde parpadeante) en niveles High/Critical ---
            alarma_activa = nivel in ("High", "Critical")
            if alarma_activa and (time.time() - ultimo_disparo_alarma) > cooldown_alarma_seg:
                ultimo_disparo_alarma = time.time()
                print(f"[ALARMA] {generar_mensaje_alerta(nivel, fp)}")
                reproducir_alarma_en_segundo_plano()

            if alarma_activa:
                # Borde rojo grueso parpadeante como refuerzo visual de alarma
                parpadeo = int(time.time() * 4) % 2 == 0
                if parpadeo:
                    cv2.rectangle(frame_final, (0, 0), (ancho - 1, alto - 1), (0, 0, 255), 12)
                cv2.putText(
                    frame_final, "!! ALERTA !!", (ancho // 2 - 100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA,
                )

            texto = f"FP: {fp}%  |  Nivel: {nivel}"
            cv2.putText(frame_final, texto, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            cv2.putText(
                frame_final, "Presiona 'q' para salir", (15, alto - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow("FireQuant - Demo en vivo", frame_final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        captura.release()
        cv2.destroyAllWindows()

    if historial:
        _graficar_evolucion(historial, salida_grafica)

    return historial


def _graficar_evolucion(historial: List[Tuple[float, float]], salida_png: str) -> None:
    """
    Genera y guarda la gráfica de Fire Percentage vs. Tiempo a partir del
    historial registrado durante la demo (Objetivo Específico #4).

    Args:
        historial: Lista de tuplas (tiempo_segundos, fire_percentage).
        salida_png: Ruta del archivo de imagen a generar.
    """
    tiempos = [h[0] for h in historial]
    valores_fp = [h[1] for h in historial]

    plt.figure(figsize=(10, 5))
    plt.plot(tiempos, valores_fp, color="#c62828", linewidth=1.5)
    plt.fill_between(tiempos, valores_fp, color="#c62828", alpha=0.15)
    plt.xlabel("Tiempo (segundos)")
    plt.ylabel("Fire Percentage (%)")
    plt.title("Evolución del Fire Percentage durante la demo en vivo")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGráfica de evolución temporal guardada en: {salida_png}")


if __name__ == "__main__":
    ejecutar_demo_webcam(
        weights_path="best.pt",  # ajusta a la ruta local de tu modelo descargado
        conf_threshold=0.15,
    )
