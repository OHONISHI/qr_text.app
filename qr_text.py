import os
import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="QR Reader (WebRTC)", layout="centered")
st.title("QRコード リアルタイム読み取り（MediaStream / WebRTC）")

# ----------------- ユーティリティ -----------------
def digital_zoom_center(bgr, zoom=1.0):
    if zoom <= 1.0:
        return bgr
    h, w = bgr.shape[:2]
    nh, nw = int(h/zoom), int(w/zoom)
    y1 = (h - nh)//2
    x1 = (w - nw)//2
    crop = bgr[y1:y1+nh, x1:x1+nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)

def preprocess_variants(bgr):
    h, w = bgr.shape[:2]
    max_side = max(h, w)
    if max_side < 1000:
        scale = 1000 / max_side
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    den  = cv2.bilateralFilter(gray, 7, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(den)
    blur = cv2.GaussianBlur(clahe, (0,0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    thr = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)

    return [
        bgr,
        cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR),
    ]

def decode_once(img_bgr, det):
    for var in preprocess_variants(img_bgr):
        for rot in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            test = cv2.rotate(var, rot) if rot is not None else var

            ok, infos, points, _ = det.detectAndDecodeMulti(test)
            if ok and any(infos):
                texts = [t for t in infos if t]
                return texts[0], test, points

            txt, pts, _ = det.detectAndDecode(test)
            if txt:
                return txt, test, np.array([pts]) if pts is not None else None
    return "", None, None

# ----------------- UI（調整用） -----------------
c1, c2 = st.columns(2)
zoom = c1.slider("デジタルズーム", 1.0, 2.5, 1.0, 0.1)
want_draw_box = c2.checkbox("検出枠を描画", value=True)
st.info("コツ：明るく・近づけて・正面から。取り込めた瞬間に自動で表示されます。")

# ----------------- WebRTC 映像処理 -----------------
class QRTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = cv2.QRCodeDetector()
        self.last_text = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # デジタルズーム（ソフト拡大）
        img = digital_zoom_center(img, zoom=zoom)

        if self.last_text is None:
            text, test_img, points = decode_once(img, self.detector)
            if text:
                self.last_text = text
                if want_draw_box and points is not None:
                    draw = img.copy()
                    for p in points:
                        p = p.astype(int).reshape(-1, 2)
                        cv2.polylines(draw, [p], True, (0,255,0), 2)
                    img = draw

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# カメラ設定：背面カメラ優先・高解像度希望
webrtc_ctx = webrtc_streamer(
    key="qr-reader",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=QRTransformer,
    media_stream_constraints={
        "video": {
            "width":  {"ideal": 1280},
            "height": {"ideal": 720},
            "facingMode": "environment"  # スマホで背面カメラを優先
        },
        "audio": False,
    },
    async_processing=True,
)

# 結果の表示
if webrtc_ctx and webrtc_ctx.video_transformer:
    if webrtc_ctx.video_transformer.last_text:
        st.success("読み取り成功!")
        st.text_input("QR内容", value=webrtc_ctx.video_transformer.last_text)
        if st.button("もう一度読み取る"):
            webrtc_ctx.video_transformer.last_text = None
