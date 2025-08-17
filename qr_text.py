import streamlit as st
import cv2
import numpy as np

img_file_buffer = st.camera_input("QRコードを撮影してください")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    qrcode_detector = cv2.QRCodeDetector()
    _, decoded_info, _, _ = qrcode_detector.detectAndDecodeMulti(cv2_img)

    # 1つでもQR文字列が見つかれば
    qr_text = ""
    for txt in decoded_info:
        if txt:
            qr_text = txt
            break
    if qr_text:
        st.success(f"QRコード内容：{qr_text}")
        # ここで自動入力欄にテキスト反映
        st.text_input("自動入力欄", value=qr_text)
    else:
        st.warning("QRコードが見つかりませんでした")
