"""
Demo de detección TUNEL+ / TUNEL- — interfaz simple para probar el modelo.

Requiere: pip install streamlit ultralytics pillow

USO:
    streamlit run app.py

Deja este archivo en la raíz de tu carpeta de proyecto (junto a
runs_tunel/v1/weights/best.pt, o ajusta la ruta MODEL_PATH abajo).
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ---------- CONFIGURACIÓN ----------
MODEL_PATH = "best.pt"  # coloca el archivo best.pt en la raíz del repo
CONF_THRESHOLD = 0.25
CLASS_NAMES = {0: "TUNEL+", 1: "TUNEL-"}
CLASS_COLORS = {0: (255, 60, 60), 1: (60, 200, 60)}  # rojo / verde para diferenciar
# ------------------------------------

st.set_page_config(page_title="TUNEL AI — Demo de detección", page_icon="🧬")


@st.cache_resource
def cargar_modelo(path):
    return YOLO(path)


st.title("🧬 TUNEL AI — Demo de detección")
st.caption("Detección y conteo de espermatozoides TUNEL+ / TUNEL- con YOLOv8")

try:
    model = cargar_modelo(MODEL_PATH)
except Exception as e:
    st.error(f"No pude cargar el modelo en '{MODEL_PATH}'. Verifica la ruta. Error: {e}")
    st.stop()

archivo = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")

    with st.spinner("Analizando imagen..."):
        resultados = model.predict(imagen, conf=CONF_THRESHOLD, agnostic_nms=True, verbose=False)[0]

    # Dibujar cajas con PIL (sin depender de OpenCV / librerías de sistema)
    img_dibujada = imagen.copy()
    draw = ImageDraw.Draw(img_dibujada)
    conteo = {0: 0, 1: 0}

    for box in resultados.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{CLASS_NAMES.get(cls, cls)} {conf:.2f}"
        draw.text((x1, max(y1 - 12, 0)), label, fill=color)
        conteo[cls] = conteo.get(cls, 0) + 1

    st.image(img_dibujada, use_container_width=True, caption="Detecciones")

    col1, col2, col3 = st.columns(3)
    col1.metric("TUNEL+", conteo.get(0, 0))
    col2.metric("TUNEL-", conteo.get(1, 0))
    total = conteo.get(0, 0) + conteo.get(1, 0)
    pct = (conteo.get(0, 0) / total * 100) if total > 0 else 0
    col3.metric("% Fragmentación", f"{pct:.1f}%")

    st.caption(
        f"Total detectado: {total} | Umbral de confianza: {CONF_THRESHOLD} "
        "(ajustable en el código, MODEL_PATH y CONF_THRESHOLD)"
    )
