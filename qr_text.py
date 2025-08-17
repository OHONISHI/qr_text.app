import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="QR撮影（1枚スナップ）")

# --- 初期値 ---
if "camera_on" not in st.session_state:
    st.session_state["camera_on"] = True

def digital_zoom_center(bgr, zoom=1.0):
    if zoom <= 1.0: return bgr
    h, w = bgr.shape[:2]
    nh, nw = int(h/zoom), int(w/zoom)
    y1, x1 = (h-nh)//2, (w-nw)//2
    crop = bgr[y1:y1+nh, x1:x1+nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)

def decode_once(img_bgr):
    det = cv2.QRCodeDetector()
    variants = [
        img_bgr,
        cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cv2.adaptiveThreshold(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
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

st.title("QRコードを “撮影” して読み取る")

c1, c2 = st.columns(2)
zoom = c1.slider("デジタルズーム", 1.0, 2.5, 1.2, 0.1)
stop_after_shot = c2.checkbox("撮影後にカメラを停止", True)

class SnapTransformer(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img
        return frame

# --- カメラ（描画/非表示をセッションで制御） ---
webrtc_ctx = None
if st.session_state["camera_on"]:
    webrtc_ctx = webrtc_streamer(
        key="qr-snapshot",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SnapTransformer,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "facingMode": "environment"},
            "audio": False,
        },
        async_processing=True,
    )

# --- 撮影ボタン ---
if webrtc_ctx and webrtc_ctx.state.playing:
    if st.button("📸 撮影"):
        vt = webrtc_ctx.video_transformer
        if vt and vt.latest_frame is not None:
            shot = vt.latest_frame.copy()
            shot = digital_zoom_center(shot, zoom=zoom)
            with st.spinner("解析中..."):
                text = decode_once(shot)

            # ↓ 非推奨を回避：use_container_width に変更
            st.image(cv2.cvtColor(shot, cv2.COLOR_BGR2RGB),
                     caption="撮影画像", use_container_width=True)

            if text:
                st.success("読み取り成功")
                st.text_input("QR内容", value=text)
            else:
                st.warning("QRを読み取れませんでした。明るく・近づけて・正面から再試行してください。")

            # 撮影後に停止（= 次回再実行で webrtc_streamer を描画しない）
            if stop_after_shot:
                st.session_state["camera_on"] = False
                st.rerun()
        else:
            st.info("映像の準備中です。数秒待ってから撮影してください。")

