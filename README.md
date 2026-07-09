# Wildfire Fire Level Estimation

Sistema de monitoreo inteligente y cuantificación analítica de incendios forestales basado en visión por computadora. El proyecto utiliza **YOLOv26n-seg** para segmentar el fuego a nivel de píxel, calcular el porcentaje de área ocupada por las llamas (*Fire Percentage*) y clasificar la severidad del incendio en niveles discretos (Low, Medium, High, Critical), generando alertas automáticas en tiempo real.

## ¿Qué hace este proyecto?

```
Imagen / Video → Segmentación (YOLOv26n-seg) → Máscara de fuego →
Conteo de píxeles → Fire Percentage (FP) → Nivel de severidad → Alerta
```

**Fórmula central:**
```
FP = (Píxeles de fuego / Píxeles totales de la imagen) × 100
```

## Estructura del repositorio y entorno de ejecución

⚠️ **Importante:** este proyecto se desarrolló combinando dos entornos distintos, y no todos los scripts corren en el mismo lugar. Antes de ejecutar cualquier archivo, revisa en qué entorno está pensado:

| Categoría | Entorno recomendado | Por qué |
|---|---|---|
| Scripts de **descarga y preparación de datasets** | Google Colab | Necesitan descargar datasets grandes (D-Fire, ~5.3GB) y correr SAM sobre GPU; se apoyan en `google.colab.drive` para persistencia en Google Drive |
| Scripts de **entrenamiento** | Google Colab (GPU) | El modelo se entrenó en Colab (T4 → L4) por la necesidad de GPU; no requieren nada exclusivo de Colab en el código, pero entrenar en CPU local sería extremadamente lento |
| **Demos e inferencia** (`app.py`, `webcam_demo_firequant.py`) | **Local** (tu computador) | Diseñados para correr en local: la app de Streamlit y la demo de cámara necesitan acceso directo al hardware (webcam) o a un servidor local, no funcionan igual dentro de Colab |
| `inference_firequant.py` | Ambos | Es agnóstico al entorno; funciona igual en Colab o en local, siempre que tenga la ruta correcta al modelo `best.pt` |

### Sobre las rutas de Google Drive

Varios scripts de descarga/entrenamiento (`download_dfire.py`, `combinar_datasets_firequant.py`, `train_firequant_yolo26.py`) usan rutas del tipo `/content/drive/MyDrive/...`. **Esas rutas solo existen dentro de una sesión de Google Colab con Drive montado** (`from google.colab import drive; drive.mount('/content/drive')`). Si copias estos scripts a tu computador local, tendrás que:
1. Eliminar o comentar la parte de `drive.mount(...)`.
2. Cambiar las rutas `/content/drive/MyDrive/...` por una ruta local de tu preferencia (ej. `C:/Users/tu_usuario/firequant_data/...`).

## Descripción de cada archivo

| Archivo | Entorno | Función |
|---|---|---|
| `main.ipynb` | Colab | Notebook principal donde se ejecutó todo el flujo de descarga, combinación de datasets y entrenamiento. |
| `download_dfire.py` | Colab | Descarga el dataset D-Fire desde su espejo en Kaggle. |
| `dfire_sam_to_yolo_seg.py` | Colab (GPU) | Genera pseudo-máscaras de segmentación para D-Fire usando SAM (Segment Anything Model), guiado por las cajas delimitadoras originales. |
| `diagnostico_dfire.py`, `diagnostico_dfire_v2.py` | Colab | Scripts de depuración usados para verificar la estructura de carpetas y el emparejamiento imagen/etiqueta de D-Fire. No son necesarios para producción, se conservan como referencia del proceso. |
| `verificar_mascaras_sa...` (`verificar_mascaras_sam.py`) | Colab | Inspección visual de la calidad de las máscaras generadas por SAM antes de usarlas en el entrenamiento. |
| `combinar_datasets_fir...` (`combinar_datasets_firequant.py`) | Colab | Combina el dataset de Roboflow (anotado manualmente) con el de D-Fire+SAM en una sola estructura de entrenamiento YOLO-seg. |
| `inference_firequant.py` | Colab o Local | Pipeline de inferencia: carga el modelo, calcula el Fire Percentage, clasifica la severidad y genera el mensaje de alerta. |
| `procesar_lote_prueba...` (`procesar_lote_prueba.py`) | Colab o Local | Corre la inferencia sobre una carpeta completa de imágenes de prueba y genera una tabla resumen + imágenes anotadas. |
| `generar_graficas_resul...` (`generar_graficas_resultados.py`) | Colab o Local | Genera las gráficas de evolución del entrenamiento (mAP, pérdidas) y de Fire Percentage por imagen. |
| `evolucion_fp_tiempo....` (imagen) | — | Gráfica de ejemplo de Fire Percentage a lo largo del tiempo, generada por la demo de cámara. |
| `app.py` | **Local** | Aplicación web en Streamlit: sube una imagen y visualiza la segmentación, el Fire Percentage, el nivel de severidad y la alerta. |
| `webcam_demo_firequant.py` | **Local** | Demo en tiempo real usando la cámara del computador: procesa video en vivo, calcula FP por frame, dispara alarma visual y sonora en niveles altos, y grafica la evolución temporal al finalizar. |
| `best.pt` | Ambos | Pesos del modelo YOLOv26n-seg ya entrenado, listo para inferencia. |

## Cómo correr las demos (local)

### Requisitos
```bash
pip install ultralytics opencv-python matplotlib streamlit
```
En Windows, la alarma sonora de `webcam_demo_firequant.py` usa `winsound` (ya incluido en Python, no requiere instalación adicional). En otros sistemas operativos, la alarma sonora se desactiva automáticamente y solo queda la alerta visual.

### App web (Streamlit)
```bash
python -m streamlit run app.py
```
Se abre en `http://localhost:8501`. Sube una imagen y visualiza el resultado.

### Demo con cámara en vivo
```bash
python webcam_demo_firequant.py
```
Apunta la cámara a una imagen o video de fuego. Presiona `q` para salir y generar la gráfica de evolución temporal.

Ambos scripts esperan encontrar `best.pt` en la misma carpeta (o puedes especificar la ruta desde el panel lateral de la app, o editando la variable correspondiente en el script de la cámara).

## Dataset y metodología

- **Roboflow** (201 imágenes): dataset original, anotado manualmente con máscaras de segmentación.
- **D-Fire + SAM** (5,822 imágenes): dataset público de detección de fuego (solo con cajas delimitadoras originalmente); las máscaras de segmentación se generaron automáticamente usando SAM (Segment Anything Model, Meta AI), utilizando las cajas como guía geométrica.
- **Total combinado:** 6,023 imágenes, usadas para reentrenar YOLOv26n-seg y mitigar el overfitting observado con el dataset original de un solo video.

## Resultados

| Métrica (máscara, conjunto de test) | Valor |
|---|---|
| Precision | 0.646 |
| Recall | 0.494 |
| mAP50 | 0.535 |
| mAP50-95 | 0.270 |

## Umbrales de severidad

| Nivel | Rango de Fire Percentage |
|---|---|
| Low | 0.0% – 3.0% |
| Medium | 3.0% – 10.0% |
| High | 10.0% – 25.0% |
| Critical | 25.0% – 100.0% |

## Limitaciones conocidas

- El modelo tiende a subestimar el área de fuego en escenas panorámicas de gran escala, dado que el dataset de entrenamiento está compuesto mayormente por fuego a media distancia.
- Las pseudo-máscaras generadas por SAM no han sido validadas cuantitativamente (IoU) contra anotación manual.
- El módulo de análisis temporal en video (evolución del Fire Percentage a lo largo de una secuencia real, más allá de la demo de cámara) está en desarrollo.

## Créditos

- Modelo de segmentación: [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics)
- Generación de pseudo-máscaras: [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), Meta AI
- Dataset D-Fire: espejo público en Kaggle (`sayedgamal99/smoke-fire-detection-yolo`)
