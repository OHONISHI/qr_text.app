import cv2
import numpy as np
import streamlit as st

def preprocess_variants(bgr):
    # 1) 必要なら拡大（小さいQR対策）
    h, w = bgr.shape[:2]
    max_side = max(h, w)
    if max_side < 900:  # 小さければ拡大
        scale = 900 / max_side
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    # 2) グレースケール → ノイズ低減 → コントラスト強調 → ちょいシャープ
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    den  = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(den)
    blur = cv2.GaussianBlur(clahe, (0,0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)

    # 3) 2値化（効く場合と効かない場合があるので両方試す）
    thr = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 2)

    # 検出器はBGR/GRAYどちらでも動くので候補を複数返す
    return [bgr, cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)]

def try_decode_all(img_bgr):
    det = cv2.QRCodeDetector()
    for var in preprocess_variants(img_bgr):
        # 0/90/180/270度 回転も試す（傾き対策）
        for rot in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            test = cv2.rotate(var, rot) if rot is not None else var

            # 複数QR
            ok, infos, pts, _ = det.detectAndDecodeMulti(test)
            if ok:
                texts = [t for t in infos if t]
                if texts:
                    return texts[0]  # 1個でも取れたら返す

            # 単一QR
            txt, pts, _ = det.detectAndDecode(test)
            if txt:
                return txt
    return ""

st.title("QRコードを撮影してください")
img_file = st.camera_input("")

if img_file:
    bytes_data = img_file.getvalue()
    bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    qr = try_decode_all(bgr)
    if qr:
        st.success(f"QRコード内容：{qr}")
        st.text_input("自動入力欄", value=qr)
    else:
        st.warning("うまく読み取れませんでした。もう一度、明るく近づけて撮影してみてください。")

