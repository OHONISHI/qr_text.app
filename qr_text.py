import cv2
import numpy as np
import streamlit as st

st.title("QRコードを撮影してください")

def digital_zoom_center(bgr, zoom=1.0):
    if zoom <= 1.0:
        return bgr
    h, w = bgr.shape[:2]
    nh, nw = int(h/zoom), int(w/zoom)
    y1 = (h - nh)//2
    x1 = (w - nw)//2
    crop = bgr[y1:y1+nh, x1:x1+nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)

# ← ここでスライダーを出しておく
zoom = st.slider("デジタルズーム", 1.0, 3.0, 1.0, 0.1)

def is_blurry(gray, thresh=120.0):
    val = cv2.Laplacian(gray, cv2.CV_64F).var()
    return val < thresh, val

def preprocess_candidates(bgr, target_sizes=(900, 1200, 1500),
                          block_sizes=(21, 31, 41), Cs=(2, 5)):
    h, w = bgr.shape[:2]
    base_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    den  = cv2.bilateralFilter(base_gray, d=7, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(den)
    blur = cv2.GaussianBlur(clahe, (0,0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)

    cands = []
    for tgt in target_sizes:
        scale = max(tgt / max(h, w), 1.0)
        img  = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        g    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cands.append(img)
        cands.append(cv2.cvtColor(cv2.resize(sharp, (img.shape[1], img.shape[0])), cv2.COLOR_GRAY2BGR))

        for bs in block_sizes:
            if bs % 2 == 0: bs += 1
            for C in Cs:
                th = cv2.adaptiveThreshold(
                    cv2.resize(sharp, (img.shape[1], img.shape[0])),
                    255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, C
                )
                th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
                cands.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))

        _, th_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cands.append(cv2.cvtColor(th_otsu, cv2.COLOR_GRAY2BGR))
    return cands

def decode_qr(bgr):
    det = cv2.QRCodeDetector()
    blurry, var = is_blurry(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    maybe_blurry = blurry

    for cand in preprocess_candidates(bgr):
        for rot in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            img = cv2.rotate(cand, rot) if rot is not None else cand

            ok, infos, pts, _ = det.detectAndDecodeMulti(img)
            if ok and any(infos):
                texts = [t for t in infos if t]
                return texts, maybe_blurry

            txt, pts, _ = det.detectAndDecode(img)
            if txt:
                return [txt], maybe_blurry

    return [], maybe_blurry

img_file = st.camera_input("")
file_up  = st.file_uploader("または写真から選択", type=["png","jpg","jpeg"])

src = img_file.getvalue() if img_file else (file_up.getvalue() if file_up else None)

if src:
    bgr = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("画像の読み込みに失敗しました。別の画像でお試しください。")
    else:
        # ★ここでデジタルズームを適用（読み込み後・解析前）
        bgr = digital_zoom_center(bgr, zoom=zoom)

        with st.spinner("解析中..."):
            texts, maybe_blurry = decode_qr(bgr)

        if texts:
            st.success(f"読み取り成功（{len(texts)}件）")
            for i, t in enumerate(texts, 1):
                st.text_input(f"QR内容 {i}", value=t)
            if maybe_blurry:
                st.info("※ 画像がややボケ気味でした。次回はもう少し近づく/明るく撮ると安定します。")
        else:
            st.warning("QRコードが見つかりませんでした。明るく・近づけて・正面から再撮影してください。")
