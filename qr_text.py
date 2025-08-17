import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="QR撮影（1枚スナップ）")

# --- セッション初期化 ---
st.session_state.setdefault("camera_on", True)
st.session_state.setdefault("qr_text", "")     # ← ここに読み取り結果を保存
st.session_state.setdefault("shot_image", None)

st.title("QRコードを “撮影” して読み取る")

# ------ ユーティリティ ------
def digital_zoom_center(bgr, zoom=1.0):
    if zoom <= 1.0: return bgr
    h, w = bgr.shape[:2]
    nh, nw = int(h/zoom), int(w/zoom)
    y1, x1 = (h-nh)//2, (w-nw)//2
    crop = bgr[y1:y1+nh, x1:x1+nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)

def decode_once(img_bgr):
    det = cv2.QRCodeDetector()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # そのまま / グレイ / 自適応2値化
    variants = [
        img_bgr,
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        ), cv2.COLOR_GRAY2BGR),
    ]
    for var in variants:
        ok, infos, _, _ = det.detectAndDecodeMulti(var)
        if ok and any(infos):
            return [t for t in infos if t][0]
        txt, _, _ = det.detectAndDecode(var)
        if txt:
            return txt
    return ""

# ------ UI（調整） ------
c1, c2 = st.columns(2)
zoom = c1.slider("デジタルズーム", 1.0, 2.5, 1.5, 0.1)
stop_after_shot = c2.checkbox("撮影後にカメラを停止", True)

# ------ WebRTC（プレビュー＋最新フレーム保持） ------
class SnapTransformer(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img
        return frame

webrtc_ctx = None
if st.session_state["camera_on"]:
    webrtc_ctx = webrtc_streamer(
        key="qr-snapshot",
        mode=WebRtcMode.SENDRECV,
        video_transformer_


