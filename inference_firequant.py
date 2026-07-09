"""
inference_firequant.py

Módulo de inferencia para FireQuant. Aplica el modelo YOLOv26n-seg entrenado
sobre imágenes nuevas (fuera del dataset de entrenamiento), extrae la máscara
binaria de fuego, calcula el Fire Percentage (FP) según:

    FP = (P_fire / P_total) * 100

y clasifica la severidad del incendio en niveles discretos (Low, Medium,
High, Critical) para la activación de alertas tempranas.

Autor: FireQuant Project
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO


# ==============================================================
# Umbrales de severidad — Objetivo Específico #3 del proyecto
# Ajustables según calibración empírica con el equipo de emergencias.
# ==============================================================
UMBRALES_SEVERIDAD = {
    "Low": (0.0, 3.0),
    "Medium": (3.0, 10.0),
    "High": (10.0, 25.0),
    "Critical": (25.0, 100.0),
}


@dataclass
class ResultadoFireQuant:
    """
    Encapsula el resultado de la cuantificación de fuego para una imagen.

    Attributes:
        fire_pixels: Cantidad de píxeles clasificados como fuego.
        total_pixels: Cantidad total de píxeles de la imagen (alto x ancho).
        fire_percentage: Porcentaje de área ocupada por el fuego (FP).
        severity_level: Nivel de severidad asignado según FP.
        num_instances: Número de instancias/regiones de fuego detectadas.
        mean_confidence: Confianza promedio de las detecciones (0-1).
    """
    fire_pixels: int
    total_pixels: int
    fire_percentage: float
    severity_level: str
    num_instances: int
    mean_confidence: float


class FireQuantEstimator:
    """
    Encapsula el modelo YOLOv26n-seg entrenado y expone métodos para
    estimar el Fire Percentage y la severidad sobre imágenes o frames
    individuales.
    """

    def __init__(self, weights_path: str, fire_class_id: int = 0, conf_threshold: float = 0.35):
        """
        Inicializa el estimador cargando los pesos entrenados.

        Args:
            weights_path: Ruta al checkpoint entrenado (best.pt).
            fire_class_id: ID de la clase 'fuego' según data.yaml
                (por defecto 0 si el dataset tiene una sola clase).
            conf_threshold: Umbral mínimo de confianza para considerar
                una detección válida.

        Raises:
            FileNotFoundError: Si el archivo de pesos no existe.
        """
        if not Path(weights_path).is_file():
            raise FileNotFoundError(f"No se encontró el checkpoint en: {weights_path}")

        self.modelo = YOLO(weights_path)
        self.fire_class_id = fire_class_id
        self.conf_threshold = conf_threshold

    @staticmethod
    def _clasificar_severidad(fire_percentage: float) -> str:
        """
        Asigna un nivel de severidad discreto según el Fire Percentage,
        siguiendo la lógica de umbrales del Objetivo Específico #3.

        Args:
            fire_percentage: Valor de FP en el rango [0, 100].

        Returns:
            Etiqueta de severidad: 'Low', 'Medium', 'High' o 'Critical'.
        """
        for nivel, (lim_inf, lim_sup) in UMBRALES_SEVERIDAD.items():
            if lim_inf <= fire_percentage < lim_sup:
                return nivel
        return "Critical"  # fallback si FP == 100.0 exactamente

    @staticmethod
    def generar_mensaje_alerta(resultado: "ResultadoFireQuant") -> str:
        """
        Genera el mensaje de alerta temprana correspondiente al nivel de
        severidad detectado, para notificación a cuerpos de emergencia
        (Objetivo Específico #3: activación de alertas tempranas dinámicas).

        Args:
            resultado: Resultado ya calculado por `estimar()`.

        Returns:
            Mensaje de alerta formateado, listo para mostrar en consola,
            enviar por notificación, o registrar en log.
        """
        plantillas = {
            "Low": "🟢 NIVEL BAJO — Fuego detectado ({fp}% del área). Monitoreo rutinario.",
            "Medium": "🟡 NIVEL MEDIO — Fuego en expansión ({fp}% del área). Se recomienda vigilancia activa.",
            "High": "🟠 ALERTA ALTA — Incendio significativo ({fp}% del área). Notificar a cuerpos de emergencia.",
            "Critical": "🔴 ALERTA CRÍTICA — Incendio de gran magnitud ({fp}% del área). Activación inmediata de protocolo de emergencia.",
        }
        plantilla = plantillas.get(resultado.severity_level, "Nivel desconocido")
        return plantilla.format(fp=resultado.fire_percentage)

    def estimar(self, imagen: Union[str, np.ndarray]) -> ResultadoFireQuant:
        """
        Ejecuta la inferencia de segmentación sobre una imagen y calcula
        el Fire Percentage (FP) mediante conteo directo de píxeles de la
        máscara binaria combinada de todas las instancias de fuego.

        Args:
            imagen: Ruta a un archivo de imagen, o un array de NumPy
                (formato BGR, como lo entrega OpenCV) ya cargado en memoria.

        Returns:
            Instancia de ResultadoFireQuant con las métricas calculadas.

        Raises:
            FileNotFoundError: Si `imagen` es una ruta que no existe.
            ValueError: Si la imagen no pudo ser leída/decodificada.
            RuntimeError: Si la inferencia del modelo falla.
        """
        # -----------------------------------------------------------
        # 1. Carga y validación de la imagen
        # -----------------------------------------------------------
        if isinstance(imagen, str):
            if not Path(imagen).is_file():
                raise FileNotFoundError(f"No se encontró la imagen: {imagen}")
            frame = cv2.imread(imagen)
            if frame is None:
                raise ValueError(f"OpenCV no pudo decodificar la imagen: {imagen}")
        elif isinstance(imagen, np.ndarray):
            frame = imagen
        else:
            raise ValueError("El parámetro 'imagen' debe ser una ruta (str) o un array de NumPy.")

        alto, ancho = frame.shape[:2]
        total_pixels = alto * ancho

        # -----------------------------------------------------------
        # 2. Inferencia
        # -----------------------------------------------------------
        try:
            resultados = self.modelo.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Error durante la inferencia del modelo: {exc}") from exc

        resultado = resultados[0]

        # -----------------------------------------------------------
        # 3. Extracción y combinación de máscaras de la clase 'fuego'
        # -----------------------------------------------------------
        mascara_combinada = np.zeros((alto, ancho), dtype=np.uint8)
        num_instances = 0
        confidencias = []

        if resultado.masks is not None and resultado.boxes is not None:
            clases = resultado.boxes.cls.cpu().numpy()
            confs = resultado.boxes.conf.cpu().numpy()
            # Las máscaras vienen en la resolución de inferencia; se
            # reescalan al tamaño original de la imagen (data ya viene
            # alineada por Ultralytics vía resultado.masks.data + orig_shape).
            mascaras_raw = resultado.masks.data.cpu().numpy()  # (N, H_infer, W_infer)

            for idx, clase_id in enumerate(clases):
                if int(clase_id) != self.fire_class_id:
                    continue

                mascara_instancia = mascaras_raw[idx]
                mascara_resized = cv2.resize(
                    mascara_instancia.astype(np.uint8),
                    (ancho, alto),
                    interpolation=cv2.INTER_NEAREST,
                )
                mascara_combinada = np.logical_or(
                    mascara_combinada, mascara_resized
                ).astype(np.uint8)

                num_instances += 1
                confidencias.append(float(confs[idx]))

        # -----------------------------------------------------------
        # 4. Cálculo del Fire Percentage: FP = (P_fire / P_total) * 100
        # -----------------------------------------------------------
        fire_pixels = int(np.count_nonzero(mascara_combinada))
        fire_percentage = (fire_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        severidad = self._clasificar_severidad(fire_percentage)
        confianza_media = float(np.mean(confidencias)) if confidencias else 0.0

        return ResultadoFireQuant(
            fire_pixels=fire_pixels,
            total_pixels=total_pixels,
            fire_percentage=round(fire_percentage, 2),
            severity_level=severidad,
            num_instances=num_instances,
            mean_confidence=round(confianza_media, 3),
        )

    def visualizar(
        self,
        imagen: Union[str, np.ndarray],
        resultado: ResultadoFireQuant,
        salida_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Genera una visualización con la máscara de fuego superpuesta en
        rojo semitransparente y un overlay de texto con el FP y la severidad.

        Args:
            imagen: Ruta o array de la imagen original.
            resultado: Resultado ya calculado por `estimar()`.
            salida_path: Si se indica, guarda la imagen anotada en esa ruta.

        Returns:
            Imagen anotada como array de NumPy (BGR).

        Raises:
            ValueError: Si la imagen no pudo cargarse.
        """
        if isinstance(imagen, str):
            frame = cv2.imread(imagen)
            if frame is None:
                raise ValueError(f"No se pudo cargar la imagen: {imagen}")
        else:
            frame = imagen.copy()

        # Re-ejecuta predicción solo para obtener la máscara visual
        # (evita recalcular estadísticas, ya provistas en `resultado`).
        resultados = self.modelo.predict(source=frame, conf=self.conf_threshold, verbose=False)
        anotada = resultados[0].plot()

        color_severidad = {
            "Low": (0, 200, 0),
            "Medium": (0, 165, 255),
            "High": (0, 0, 255),
            "Critical": (0, 0, 139),
        }
        color = color_severidad.get(resultado.severity_level, (255, 255, 255))

        texto = f"FP: {resultado.fire_percentage}%  |  Nivel: {resultado.severity_level}"
        cv2.putText(
            anotada, texto, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
        )

        if salida_path:
            cv2.imwrite(salida_path, anotada)

        return anotada


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # Ejemplo de uso sobre una imagen nueva (fuera del dataset)
    # ---------------------------------------------------------------
    WEIGHTS_PATH = "FireQuant_runs/yolo26n_seg_v1/weights/best.pt"
    IMAGEN_PRUEBA = "/content/imagen_nueva.jpg"

    estimador = FireQuantEstimator(weights_path=WEIGHTS_PATH, conf_threshold=0.35)
    resultado = estimador.estimar(IMAGEN_PRUEBA)

    print(f"Píxeles de fuego:      {resultado.fire_pixels}")
    print(f"Píxeles totales:       {resultado.total_pixels}")
    print(f"Fire Percentage (FP):  {resultado.fire_percentage}%")
    print(f"Nivel de severidad:    {resultado.severity_level}")
    print(f"Instancias detectadas: {resultado.num_instances}")
    print(f"Confianza promedio:    {resultado.mean_confidence}")
    print()
    print(estimador.generar_mensaje_alerta(resultado))

    estimador.visualizar(IMAGEN_PRUEBA, resultado, salida_path="/content/resultado_anotado.jpg")
