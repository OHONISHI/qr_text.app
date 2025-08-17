import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="QR撮影（1枚スナップ）")
st.title("QRコードを “撮影” して読み取る")

# ---- お好みのユーティリティ（簡略版） ----
def digital_zoom_center(bgr, zoom=1.0):
    if zoom <= 1.0: return bgr
    h, w = bgr.shape[:2]
    nh, nw = int(h/zoom), int(w/zoom)
    y1, x1 = (h-nh)//2, (w-nw)//2
    crop = bgr[y1:y1+nh, x1:x1+nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)

def decode_once(img_bgr):
    det = cv2.QRCodeDetector()
    # そのまま→グレイ→二値の簡易3パターン
    variants = [
        img_bgr,
        cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cv2.adaptiveThreshold(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        ), cv2.COLOR_GRAY2BGR),
    ]
    for var in variants:
        # 複数QR
        ok, infos, pts, _ = det.detectAndDecodeMulti(var)
        if ok and any(infos):
            texts = [t for t in infos if t]
            return texts[0]
        # 単一QR
        txt, _, _ = det.detectAndDecode(var)
        if txt:
            return txt
    return ""

# ---- UI ----
c1, c2 = st.columns(2)
zoom = c1.slider("デジタルズーム", 1.0, 2.5, 1.2, 0.1)
stop_after_shot = c2.checkbox("撮影後にカメラを停止", True)
st.caption("プレビューで構図を合わせ、📸 撮影 を押すとその1枚だけ解析します。")

# ---- WebRTC: プレビューのみ、撮影はボタンで1枚 ----
class SnapTransformer(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None  # 直近フレーム（BGR）

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img  # 常に更新（撮影ボタン押下時に利用）
        return frame  # 返す映像はそのまま（プレビュー用途）

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

# ---- 撮影ボタン：最新フレームを1枚だけ解析 ----
if webrtc_ctx and webrtc_ctx.state.playing:
    if st.button("📸 撮影"):
        vt = webrtc_ctx.video_transformer
        if vt and vt.latest_frame is not None:
            shot = vt.latest_frame.copy()
            shot = digital_zoom_center(shot, zoom=zoom)

            with st.spinner("解析中..."):
                text = decode_once(shot)

            st.image(cv2.cvtColor(shot, cv2.COLOR_BGR2RGB), caption="撮影画像", use_column_width=True)
            if text:
                st.success("読み取り成功")
                st.text_input("QR内容", value=text)
            else:
                st.warning("QRを読み取れませんでした。もう一度、明るく近づけて正面から試してください。")

            if stop_after_shot:
                webrtc_ctx.stop()  # ← 写真モード完了
        else:
            st.info("映像がまだ来ていません。数秒待ってから再度お試しください。")
