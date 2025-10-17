import streamlit as st
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="Skin Analysis MVP v2", layout="centered")

st.title("🧴 Skin Analysis MVP v2 (face-mesh mask)")
st.write("В этой версии добавлены: точное выделение лица через MediaPipe FaceMesh, исключение глаз/губ/бровей, аккуратный кроп.")

uploaded = st.file_uploader("Загрузите изображение лица (без фильтров, ровный свет)", type=["jpg", "jpeg", "png"])

# ---------- Utils ----------
def to_cv(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def resize_long_side(bgr, max_side=1280):
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return bgr

# ---------- Face Mesh Mask ----------
mp_face_mesh = mp.solutions.face_mesh

# Common landmark index groups for FaceMesh (468 pts)
# Polygons (approx): face oval, lips, eyes, eyebrows — to exclude non‑skin.
# Indices are from MediaPipe standard topology.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183,78]
LIPS_INNER = [78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70,63,105,66,107,55,65,52,53,46]
RIGHT_BROW = [336,296,334,293,300,383,276,283,282,295]

def landmarks_to_points(landmarks, w, h):
    pts = []
    for lm in landmarks:
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)

def poly_from_indices(landmarks, indices, w, h):
    pts = [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices]
    return np.array(pts, dtype=np.int32)

def make_skin_mask(bgr):
    """Return (mask_skin, face_bbox) where mask is boolean (H,W)."""
    h, w = bgr.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return np.zeros((h,w), dtype=np.uint8), (0,0,w,h)
        lm = res.multi_face_landmarks[0].landmark

        # Base: face oval as big polygon
        face_poly = poly_from_indices(lm, FACE_OVAL, w, h)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_poly], 255)

        # Exclude: eyes, brows, lips (cut out)
        for idxs in [LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS_OUTER, LIPS_INNER]:
            poly = poly_from_indices(lm, idxs, w, h)
            cv2.fillPoly(mask, [poly], 0)

        # Optional: shave hairline by eroding a little
        mask = cv2.erode(mask, np.ones((7,7), np.uint8), iterations=1)
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (7,7), 0)
        mask = (mask > 128).astype(np.uint8) * 255

        # Face bbox for convenient crop
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            bbox = (0,0,w,h)
        else:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            # small padding
            pad = int(0.06 * max(x2-x1, y2-y1))
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(w, x2+pad), min(h, y2+pad)
            bbox = (x1, y1, x2, y2)
        return mask, bbox

# ---------- Metrics ----------
def redness_metric(bgr, mask):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    a = a.astype(np.float32)
    m = mask > 0
    if not m.any():
        return 0.0, bgr
    # normalize a inside skin only
    a_s = a[m]
    a_norm = (a - a_s.min()) / (a_s.max() - a_s.min() + 1e-6)
    mean_red = float(a_norm[m].mean())

    heat = (a_norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat[~m] = 0
    overlay = cv2.addWeighted(bgr, 0.6, heat, 0.4, 0)
    return mean_red, overlay

def shine_metric(bgr, mask):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0]
    m = mask > 0
    if not m.any():
        return 0.0, bgr
    Lskin = L[m]
    thr = np.quantile(Lskin, 0.9)  # top 10% brightness
    shine_mask = (L > thr) & m
    ratio = float(shine_mask.sum()) / float(m.sum())
    vis = bgr.copy()
    vis[shine_mask] = (0, 255, 255)  # yellow
    return ratio, vis

def texture_metric(bgr, mask):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)  # denoise while keeping edges
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    m = mask > 0
    val = float(lap[m].var()) if m.any() else 0.0

    lap_norm = lap.copy()
    if lap_norm.max() > 0:
        lap_norm = (lap_norm / lap_norm.max() * 255).astype(np.uint8)
    heat = cv2.applyColorMap(lap_norm, cv2.COLORMAP_MAGMA)
    heat[~m] = 0
    overlay = cv2.addWeighted(bgr, 0.6, heat, 0.4, 0)
    return val, overlay

def spot_metric(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    m = mask > 0
    if not m.any():
        return 0, bgr
    v_med = np.median(V[m])
    v_thr = max(0, v_med - 25)
    s_thr = np.quantile(S[m], 0.4)
    spots = (V < v_thr) & (S > s_thr) & m

    spots_uint = (spots.astype(np.uint8) * 255)
    spots_uint = cv2.morphologyEx(spots_uint, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(spots_uint, connectivity=8)
    count = 0
    vis = bgr.copy()
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        # avoid long skinny fragments (hair/edges)
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if aspect > 5:
            continue
        count += 1
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 1)
    return int(count), vis

if uploaded is not None:
    img = Image.open(uploaded)
    bgr_full = to_cv(img)
    bgr_full = resize_long_side(bgr_full, 1400)

    mask_full, bbox = make_skin_mask(bgr_full)
    x1,y1,x2,y2 = bbox
    face = bgr_full[y1:y2, x1:x2]
    mask = mask_full[y1:y2, x1:x2]

    st.subheader("Маска лица (исключены глаза/губы/брови)")
    st.image(mask, use_column_width=True, clamp=True)

    st.subheader("Кадр для анализа (кроп по лицу)")
    st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), use_column_width=True)

    red_score, red_vis = redness_metric(face, mask)
    shine_score, shine_vis = shine_metric(face, mask)
    text_score, text_vis = texture_metric(face, mask)
    spot_count, spot_vis = spot_metric(face, mask)

    st.subheader("Результаты")
    st.write({
        "redness_score_0_1": round(red_score, 3),
        "shine_ratio_0_1": round(shine_score, 3),
        "texture_var": round(text_score, 1),
        "spots_count": int(spot_count)
    })

    st.caption("Теплокарта покраснений")
    st.image(cv2.cvtColor(red_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.caption("Зоны блеска (жёлтым)")
    st.image(cv2.cvtColor(shine_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.caption("Текстурная карта (морщины/поры proxy)")
    st.image(cv2.cvtColor(text_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.caption("Обнаруженные пятна/высыпания (красные рамки)")
    st.image(cv2.cvtColor(spot_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.info("⚠️ Дисклеймер: Это не медицинская диагностика. Метрики приблизительные и зависят от освещения/камеры.")
else:
    st.info("Загрузите фото, чтобы увидеть анализ.")