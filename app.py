
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from skimage.morphology import remove_small_objects, dilation, disk
from skimage.measure import label, regionprops
from skimage.filters import gabor

st.set_page_config(page_title="Skin Analysis v3.1 — presets & grouping", layout="wide")
st.title("🧴 Skin Analysis v3.1 — пресеты кожи, фильтрация и группировка")

uploaded = st.file_uploader("Загрузите фото лица (без фильтров, ровный свет)", type=["jpg","jpeg","png"])

# ===== Presets =====
PRESETS = {
    "Нормальная/комбинированная": dict(
        red_thr=0.55, acne_log_q=0.70, min_acne_area=60, min_spot_area=40,
        pores_q=0.88, pores_sigma=(1.0, 2.5),
        pig_b_q=0.70, pig_L_q=0.35, pig_S_q=0.35,
        ery_thr=0.52, milia_V_q=0.88, milia_S_q=0.30
    ),
    "Жирная/проблемная": dict(
        red_thr=0.50, acne_log_q=0.60, min_acne_area=80, min_spot_area=50,
        pores_q=0.90, pores_sigma=(1.2, 2.8),
        pig_b_q=0.72, pig_L_q=0.35, pig_S_q=0.35,
        ery_thr=0.50, milia_V_q=0.90, milia_S_q=0.28
    ),
    "Чувствительная/тонкая": dict(
        red_thr=0.58, acne_log_q=0.72, min_acne_area=50, min_spot_area=35,
        pores_q=0.86, pores_sigma=(0.9, 2.2),
        pig_b_q=0.68, pig_L_q=0.38, pig_S_q=0.32,
        ery_thr=0.54, milia_V_q=0.86, milia_S_q=0.32
    ),
}

# ===== Utils =====
def to_cv(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def resize_long_side(bgr, max_side=1400):
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return bgr

def clahe_L(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    lab2 = cv2.merge([Lc,a,b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

# ===== Face mask =====
mp_face_mesh = mp.solutions.face_mesh
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
             152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183]
LIPS_INNER = [78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183]
LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LEFT_BROW = [70,63,105,66,107,55,65,52,53,46]
RIGHT_BROW = [336,296,334,293,300,383,276,283,282,295]

def poly_from_indices(lm, idxs, w, h):
    return np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in idxs], dtype=np.int32)

def make_skin_mask(bgr):
    h, w = bgr.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return np.zeros((h,w), np.uint8), (0,0,w,h)
        lm = res.multi_face_landmarks[0].landmark
        base = np.zeros((h,w), np.uint8)
        cv2.fillPoly(base, [poly_from_indices(lm, FACE_OVAL, w, h)], 255)
        for idxs in [LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS_OUTER, LIPS_INNER]:
            cv2.fillPoly(base, [poly_from_indices(lm, idxs, w, h)], 0)
        base = cv2.erode(base, np.ones((7,7), np.uint8), 1)
        base = cv2.GaussianBlur(base, (7,7), 0)
        base = (base>128).astype(np.uint8)*255
        ys, xs = np.where(base>0)
        if xs.size==0: return base, (0,0,w,h)
        x1,x2 = xs.min(), xs.max(); y1,y2 = ys.min(), ys.max()
        pad = int(0.06*max(x2-x1, y2-y1)); x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(w,x2+pad); y2=min(h,y2+pad)
        return base, (x1,y1,x2,y2)

# ===== Core helpers =====
def specular_mask(bgr, skin_mask):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0]; m = skin_mask>0
    if not m.any(): return np.zeros_like(L, dtype=bool)
    thr = np.quantile(L[m], 0.9)
    return (L>thr) & m

def redness_map(bgr, skin_mask):
    bgr_s = cv2.bilateralFilter(bgr, 9, 50, 50)
    lab = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2LAB)
    L,a,b = cv2.split(lab)
    m = skin_mask>0
    a_s = a[m]
    a_norm = np.zeros_like(a, dtype=np.float32)
    if a_s.size>0:
        a_norm = (a - a_s.min())/(a_s.max()-a_s.min()+1e-6)
    a_norm[~m]=0
    return a_norm

def group_mask(binary_mask, radius=6, min_group_area=150):
    if binary_mask.dtype != bool:
        binary_mask = binary_mask.astype(bool)
    blob = dilation(binary_mask, disk(radius))
    lab = label(blob)
    keep = np.zeros_like(blob, dtype=bool)
    for r in regionprops(lab):
        if r.area >= min_group_area:
            rr, cc = r.coords[:,0], r.coords[:,1]
            keep[rr, cc] = True
    return keep

def draw_boxes_from_mask(bgr, mask, color, thickness=2):
    vis = bgr.copy()
    lab = label(mask)
    for p in regionprops(lab):
        minr, minc, maxr, maxc = p.bbox
        cv2.rectangle(vis, (minc, minr), (maxc, maxr), color, thickness)
    return vis

# ===== Detectors =====
def detect_acne(bgr, skin_mask, cfg):
    red = redness_map(bgr, skin_mask)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    log = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0,0), 1.2), cv2.CV_32F))
    m = skin_mask>0; shine = specular_mask(bgr, skin_mask)
    cand = (red>cfg["red_thr"]) & (log>np.quantile(log[m], cfg["acne_log_q"])) & (~shine) & m
    # убрать мелкие
    labl = label(cand); keep = np.zeros_like(cand, bool)
    for p in regionprops(labl):
        if p.area >= cfg["min_acne_area"]:
            rr, cc = p.coords[:,0], p.coords[:,1]
            keep[rr, cc] = True
    # сгруппировать очаги
    groups = group_mask(keep, radius=8, min_group_area=max(200, cfg["min_acne_area"]*3))
    vis = draw_boxes_from_mask(bgr, groups, (0,0,255), 2)
    return groups, vis

def detect_milia(bgr, skin_mask, cfg):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv); m = skin_mask>0
    if not m.any(): return np.zeros_like(m), bgr
    v_thr = np.quantile(V[m], cfg["milia_V_q"]); s_thr = np.quantile(S[m], cfg["milia_S_q"])
    cand = (V>v_thr) & (S<s_thr) & m & (~specular_mask(bgr, skin_mask))
    # убрать мелкие/слишком большие
    labl = label(cand); keep = np.zeros_like(cand, bool)
    for p in regionprops(labl):
        if 25 <= p.area <= 300:
            keep[p.coords[:,0], p.coords[:,1]] = True
    groups = group_mask(keep, radius=5, min_group_area=120)
    vis = draw_boxes_from_mask(bgr, groups, (255,255,255), 2)
    return groups, vis

def detect_wrinkles(bgr, skin_mask):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    m = skin_mask>0
    mags = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        fr, fi = gabor(gray, frequency=0.25, theta=theta)
        mags.append(np.sqrt(fr**2 + fi**2))
    mag = np.mean(mags, axis=0)
    t = np.quantile(mag[m], 0.85) if m.any() else 255
    wr = (mag>t) & m
    wr = remove_small_objects(wr, 80)
    vis = bgr.copy(); vis[wr]=(255,0,255)
    return wr, vis

def detect_pigmentation(bgr, skin_mask, cfg):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L,a,b = cv2.split(lab); m = skin_mask>0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    b_thr = np.quantile(b[m], cfg["pig_b_q"]); L_thr = np.quantile(L[m], cfg["pig_L_q"]); S_thr = np.quantile(S[m], cfg["pig_S_q"])
    pig = ((b>b_thr) | ((L<L_thr) & (S>S_thr))) & m & (~specular_mask(bgr, skin_mask))
    pig = remove_small_objects(pig, cfg["min_spot_area"])
    pig = group_mask(pig, radius=7, min_group_area=260)
    vis = bgr.copy(); vis[pig]=(0,128,255)
    return pig, vis

def detect_erythema(bgr, skin_mask, cfg):
    red = redness_map(bgr, skin_mask); m = skin_mask>0
    mask = (red>cfg["ery_thr"]) & m & (~specular_mask(bgr, skin_mask))
    mask = group_mask(mask, radius=8, min_group_area=260)
    heat_u8 = (np.clip(red,0,1)*255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET); cm[~m]=0
    vis = cv2.addWeighted(bgr, 0.6, cm, 0.4, 0)
    return mask, vis

def detect_pores(bgr, skin_mask, cfg):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s1, s2 = cfg["pores_sigma"]
    g1 = cv2.GaussianBlur(gray,(0,0), s1); g2 = cv2.GaussianBlur(gray,(0,0), s2)
    band = cv2.subtract(g1, g2); band = cv2.normalize(np.abs(band), None, 0, 255, cv2.NORM_MINMAX)
    m = skin_mask>0; thr = np.quantile(band[m], cfg["pores_q"]) if m.any() else 255
    pr = (band>thr) & m & (~specular_mask(bgr, skin_mask))
    pr = remove_small_objects(pr, 12)
    pr = group_mask(pr, radius=5, min_group_area=140)
    vis = bgr.copy(); vis[pr]=(0,255,0)
    return pr, vis

def dehydration_score(bgr, skin_mask):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(cv2.bilateralFilter(gray,7,50,50), cv2.CV_32F, ksize=3))
    m = skin_mask>0
    if not m.any(): return 0.0
    text = float(lap[m].mean())
    shine_ratio = float(specular_mask(bgr, skin_mask).sum())/float(m.sum())
    score = np.clip((text/110.0) * (1.0 - 1.1*shine_ratio), 0, 1)
    return score

def post_acne(bgr, skin_mask, cfg):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    log = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray,(0,0),1.2), cv2.CV_32F))
    red = redness_map(bgr, skin_mask)
    pig, _ = detect_pigmentation(bgr, skin_mask, cfg)
    m = skin_mask>0
    flat = (log < np.quantile(log[m], 0.60)) & m & (~specular_mask(bgr, skin_mask))
    mask = ((red>cfg["red_thr"]-0.05) | pig) & flat
    mask = remove_small_objects(mask, cfg["min_spot_area"])
    mask = group_mask(mask, radius=6, min_group_area=180)
    vis = draw_boxes_from_mask(bgr, mask, (0,128,255), 2)
    return mask, vis

# ===== Main =====
preset_name = st.sidebar.selectbox("Тип кожи / пресет", list(PRESETS.keys()), index=1)
cfg = PRESETS[preset_name]

st.sidebar.write("Порог чувствительности (тонкая настройка):")
cfg["red_thr"] = st.sidebar.slider("Redness threshold (0..1)", 0.3, 0.8, float(cfg["red_thr"]), 0.01)
cfg["acne_log_q"] = st.sidebar.slider("Acne LoG quantile", 0.5, 0.9, float(cfg["acne_log_q"]), 0.01)
cfg["min_acne_area"] = st.sidebar.slider("Min acne blob area", 10, 200, int(cfg["min_acne_area"]), 5)
cfg["pores_q"] = st.sidebar.slider("Pores band quantile", 0.75, 0.98, float(cfg["pores_q"]), 0.01)

if uploaded is not None:
    bgr0 = to_cv(Image.open(uploaded))
    bgr0 = resize_long_side(bgr0, 1400)
    bgr = clahe_L(bgr0)
    skin_mask, bbox = make_skin_mask(bgr)
    x1,y1,x2,y2 = bbox
    face = bgr[y1:y2, x1:x2]
    mask = skin_mask[y1:y2, x1:x2] > 0

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Маска лица")
        st.image((mask.astype(np.uint8)*255), clamp=True, use_column_width=True)
    with col2:
        st.subheader("Кадр для анализа")
        st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Run detectors
    ery_mask, ery_vis = detect_erythema(face, mask, cfg)
    acne_mask, acne_vis = detect_acne(face, mask, cfg)
    pig_mask, pig_vis = detect_pigmentation(face, mask, cfg)
    milia_mask, milia_vis = detect_milia(face, mask, cfg)
    wr_mask, wr_vis = detect_wrinkles(face, mask)
    pr_mask, pr_vis = detect_pores(face, mask, cfg)
    dh_score = dehydration_score(face, mask)
    pa_mask, pa_vis = post_acne(face, mask, cfg)

    st.subheader("Итоги (укрупнённые зоны)")
    st.write({
        "Акне (зоны)": int(label(acne_mask).max()),
        "Купероз/покраснение (площадь)": int(ery_mask.sum()),
        "Милиумы (зоны)": int(label(milia_mask).max()),
        "Морщины (площадь)": int(wr_mask.sum()),
        "Пигментация (площадь)": int(pig_mask.sum()),
        "Расширенные поры (площадь)": int(pr_mask.sum()),
        "Обезвоженность (0..1)": round(float(dh_score), 3),
        "Постакне (зоны)": int(label(pa_mask).max()),
    })

    c1,c2 = st.columns(2); c3,c4 = st.columns(2); c5,c6 = st.columns(2)
    with c1: st.caption("Покраснение (heatmap/маска)"); st.image(cv2.cvtColor(ery_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    with c2: st.caption("Акне (укрупнённые зоны)"); st.image(cv2.cvtColor(acne_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    with c3: st.caption("Пигментация (зоны)"); st.image(cv2.cvtColor(pig_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    with c4: st.caption("Милиумы (зоны)"); st.image(cv2.cvtColor(milia_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    with c5: st.caption("Морщины"); st.image(cv2.cvtColor(wr_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    with c6: st.caption("Расширенные поры"); st.image(cv2.cvtColor(pr_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.caption("Постакне (оранжевые рамки)")
    st.image(cv2.cvtColor(pa_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.info("⚠️ Немедицинский прототип. Для клинических выводов обратитесь к врачу.")
else:
    st.info("Загрузите фото, чтобы увидеть анализ.")
