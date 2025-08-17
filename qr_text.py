import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="QRコード読み取り（高精度）", layout="centered")
st.title("QRコードを撮影してください")

@st.cache_resource
def get_detector():
    return cv2.QRCodeDetector()

def preprocess_variants(bgr, upscale_to=900, block_size=31, C=2):
    # block_size は奇数に補正
    if block_size % 2 == 0:
        block_size += 1

    h, w = bgr.shape[:2]
    max_side = max(h, w)

    # 小さいQR対策：拡大
    if max_side < upscale_to:
        scale = upscale_to / max_side
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    # 前処理（ノイズ低減→コントラスト↑→軽いシャープ）
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    den  = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(den)
    blur = cv2.GaussianBlur(clahe, (0,0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)

    # 2値化も併用（効くケースと効かないケースがあるので候補に入れる）
    thr = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, C
    )

    # 小さい文字に強い拡大、ブレ気味に効く縮小の両方を試す
    variants = [
        bgr,
        cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR),
        cv2.resize(bgr, (max(1, w//2), max(1, h//2)), interpolation=cv2.INTER_AREA)
    ]
    return variants

def try_decode_all(img_bgr, upscale_to=900, block_size=31, debug=False):
    det = get_detector()
    results = []
    annotated = None

    for var in preprocess_variants(img_bgr, upscale_to, block_size):
        for rot in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            test = cv2.rotate(var, rot) if rot is not None else var

            # 複数QR
            ok, infos, points, _ = det.detectAndDecodeMulti(test)
            if ok and points is not None:
                texts = [t for t in infos if t]
                if texts:
                    annotated = test.copy()
                    for p in points:
                        p = p.astype(int).reshape(-1, 2)
                        cv2.polylines(annotated, [p], True, (0, 255, 0), 2)
                    results.extend(texts)
                    return list(dict.fromkeys(results)), annotated  # 重複除去して即返す

            # 単一QR
            txt, pts, _ = det.detectAndDecode(test)
            if txt:
                annotated = test.copy()
                if pts is not None:
                    pts = pts.astype(int).reshape(-1, 2)
                    cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                results.append(txt)
                return list(dict.fromkeys(results)), annotated

    return [], annotated

# ---- UI（調整用） ----
col1, col2 = st.columns(2)
upscale_to = col1.slider("拡大の目標ピクセル（長辺）", 700, 1600, 1000, 50)
block_size = col2.slider("2値化ブロックサイズ（奇数）", 11, 51, 31, 2)
debug_view = st.toggle("デバッグ画像を表示", value=False)

img_file = st.camera_input("カメラの許可が必要です（許可後に撮影）")
file_up = st.file_uploader("または画像ファイルを選択", type=["png", "jpg", "jpeg"])

src_bytes = img_file.getvalue() if img_file else (file_up.getvalue() if file_up else None)

if src_bytes:
    bgr = cv2.imdecode(np.frombuffer(src_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("画像の読み込みに失敗しました。別の画像でお試しください。")
    else:
        with st.spinner("解析中..."):
            texts, ann = try_decode_all(bgr, upscale_to=upscale_to, block_size=block_size, debug=debug_view)

        if texts:
            st.success(f"読み取り成功（{len(texts)}件）")
            for i, t in enumerate(texts, 1):
                st.text_input(f"QR内容 {i}", value=t)
        else:
            st.warning("読み取れませんでした。明るく・近づけて・なるべく正面から再撮影してください。")

        if debug_view and ann is not None:
            rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption="検出結果（枠表示）", use_column_width=True)
