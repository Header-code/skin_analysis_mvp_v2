
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from skimage.filters import frangi, gabor
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

st.set_page_config(page_title="Skin Analysis MVP v3", layout="wide")
st.title("🧴 Skin Analysis MVP v3 — мульти-анализ")
st.caption("Немедицинский прототип. Эвристики CV + FaceMesh.")

uploaded = st.file_uploader("Загрузите фото лица (без фильтров, ровный свет)", type=["jpg","jpeg","png"])

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

def acne_detect(bgr, skin_mask):
    red = redness_map(bgr, skin_mask)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    log = cv2.GaussianBlur(gray, (0,0), 1.2)
    log = cv2.Laplacian(log, cv2.CV_32F, ksize=3); log = np.abs(log)
    m = skin_mask>0
    shine = specular_mask(bgr, skin_mask)
    score = (red>0.55) & (log>np.quantile(log[m], 0.65)) & (~shine) & m
    labl = label(score)
    props = regionprops(labl)
    boxes=[]; vis=bgr.copy()
    for p in props:
        if p.area<20 or p.area>3500: continue
        minr,minc,maxr,maxc = p.bbox
        boxes.append((minc,minr,maxc-minc,maxr-minr))
        cv2.rectangle(vis,(minc,minr),(maxc,maxr),(0,0,255),1)
    return boxes, vis

def erythema_detect(bgr, skin_mask):
    red = redness_map(bgr, skin_mask)
    mask = red>0.5
    mask = mask & (skin_mask>0) & (~specular_mask(bgr, skin_mask))
    return red, mask

def pigmentation_detect(bgr, skin_mask):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L,a,b = cv2.split(lab)
    m = skin_mask>0
    if not m.any(): return np.zeros_like(L, bool), bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    b_thr = np.quantile(b[m], 0.65)
    L_thr = np.quantile(L[m], 0.35)
    S_thr = np.quantile(S[m], 0.35)
    pig = ((b>b_thr) | (L<L_thr)&(S>S_thr)) & m & (~specular_mask(bgr, skin_mask))
    pig = remove_small_objects(label(pig), 40)>0
    vis = bgr.copy(); vis[pig]=(0,128,255)
    return pig, vis

def milia_detect(bgr, skin_mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv); m = skin_mask>0
    if not m.any(): return [], bgr
    v_thr = np.quantile(V[m], 0.85)
    s_thr = np.quantile(S[m], 0.35)
    cand = (V>v_thr) & (S<s_thr) & m
    labl = label(cand)
    props = regionprops(labl)
    boxes=[]; vis=bgr.copy()
    for p in props:
        if p.area<8 or p.area>150: continue
        rmin,cmin,rmax,cmax = p.bbox
        circ = (4*np.pi*p.area) / (p.perimeter**2 + 1e-6)
        if circ<0.4: continue
        boxes.append((cmin,rmin,cmax-cmin,rmax-rmin))
        cv2.rectangle(vis,(cmin,rmin),(cmax,rmax),(255,255,255),1)
    return boxes, vis

def wrinkle_detect(bgr, skin_mask):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    m = skin_mask>0
    mags = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        fr, fi = gabor(gray, frequency=0.25, theta=theta)
        mags.append(np.sqrt(fr**2 + fi**2))
    mag = np.mean(mags, axis=0)
    t = np.quantile(mag[m], 0.8) if m.any() else 255
    wr = (mag>t) & m
    wr = remove_small_objects(label(wr), 60)>0
    vis = bgr.copy(); vis[wr]=(255,0,255)
    return wr, vis

def pores_detect(bgr, skin_mask):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(gray,(0,0),1.0)
    g2 = cv2.GaussianBlur(gray,(0,0),2.5)
    band = cv2.subtract(g1, g2)
    band = cv2.normalize(np.abs(band), None, 0, 255, cv2.NORM_MINMAX)
    m = skin_mask>0
    thr = np.quantile(band[m], 0.85) if m.any() else 255
    pr = (band>thr) & m & (~specular_mask(bgr, skin_mask))
    pr = remove_small_objects(label(pr), 10)>0
    vis = bgr.copy(); vis[pr]=(0,255,0)
    return pr, vis

def dehydration_indicator(bgr, skin_mask):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(cv2.bilateralFilter(gray,7,50,50), cv2.CV_32F, ksize=3)
    lap = np.abs(lap); m = skin_mask>0
    if not m.any(): return 0.0
    text = float(lap[m].mean())
    shine_ratio = float(specular_mask(bgr, skin_mask).sum())/float(m.sum())
    score = np.clip((text/100.0) * (1.0 - shine_ratio*1.2), 0, 1)
    return score

def post_acne_detect(bgr, skin_mask, acne_boxes):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv); m = skin_mask>0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    log = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray,(0,0),1.2), cv2.CV_32F))
    red = redness_map(bgr, skin_mask)
    pig,_ = pigmentation_detect(bgr, skin_mask)
    flat = (log < np.quantile(log[m], 0.55)) & m & (~specular_mask(bgr, skin_mask))
    mask = ((red>0.45) | pig) & flat
    labl = label(mask); props = regionprops(labl)
    vis = bgr.copy(); boxes=[]
    for p in props:
        if p.area<25 or p.area>4000: continue
        rmin,cmin,rmax,cmax = p.bbox
        boxes.append((cmin,rmin,cmax-cmin,rmax-rmin))
        cv2.rectangle(vis,(cmin,rmin),(cmax,rmax),(0,128,255),1)
    return boxes, vis

def overlay_heat(img, heat, m, alpha=0.45, cmap=cv2.COLORMAP_JET):
    heat_u8 = (np.clip(heat,0,1)*255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, cmap)
    cm[~m] = 0
    return cv2.addWeighted(img, 1-alpha, cm, alpha, 0)

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

    acne_boxes, acne_vis = acne_detect(face, mask)
    ery_heat, ery_mask = erythema_detect(face, mask)
    pig_mask, pig_vis = pigmentation_detect(face, mask)
    milia_boxes, milia_vis = milia_detect(face, mask)
    wrinkle_mask, wrinkle_vis = wrinkle_detect(face, mask)
    pores_mask, pores_vis = pores_detect(face, mask)
    dehydr_score = dehydration_indicator(face, mask)
    post_boxes, post_vis = post_acne_detect(face, mask, acne_boxes)

    ery_vis = overlay_heat(face, ery_heat, mask)

    st.subheader("Итоги")
    st.write({
        "Акне (области)": len(acne_boxes),
        "Купероз (площадь)": int(ery_mask.sum()),
        "Милиумы (области)": len(milia_boxes),
        "Морщины (площадь)": int(wrinkle_mask.sum()),
        "Пигментация (площадь)": int(pig_mask.sum()),
        "Покраснение (ср. по маске 0..1)": round(float(ery_heat[mask].mean()),3),
        "Птоз": "— (не диагностируем в MVP)",
        "Расширенные поры (площадь)": int(pores_mask.sum()),
        "Обезвоженность (0..1)": round(float(dehydr_score),3),
        "Постакне (области)": len(post_boxes)
    })

    c1,c2 = st.columns(2)
    with c1: st.caption("Покраснение (heatmap)"); st.image(cv2.cvtColor(ery_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    with c2: st.caption("Акне (красные рамки)"); st.image(cv2.cvtColor(acne_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    c3,c4 = st.columns(2)
    with c3: st.caption("Пигментация (оранжевым)"); st.image(cv2.cvtColor(pig_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    with c4: st.caption("Милиумы (белые рамки)"); st.image(cv2.cvtColor(milia_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    c5,c6 = st.columns(2)
    with c5: st.caption("Морщины (пурпур)"); st.image(cv2.cvtColor(wrinkle_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    with c6: st.caption("Расширенные поры (зелёным)"); st.image(cv2.cvtColor(pores_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    st.caption("Постакне (оранжевые рамки)")
    st.image(cv2.cvtColor(post_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.info("⚠️ Не является медицинской диагностикой. Результат зависит от освещения, позы и камеры.")
else:
    st.info("Загрузите фото, чтобы увидеть анализ.")
