import streamlit as st
import cv2
import numpy as np
import tempfile

def main():
    st.title("選取廣告位置")
    uploaded_file = st.file_uploader("上傳要插入廣告之視頻或照片...", type=['jpg', 'png', 'jpeg', 'mp4', 'avi'])

    if uploaded_file is not None:
        # Determine if the uploaded file is a video
        if uploaded_file.name.split('.')[-1] in ['mp4', 'avi']:
            # Handle video file by saving to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1])
            tfile.write(uploaded_file.getvalue())
            vid = cv2.VideoCapture(tfile.name)
            ret, frame = vid.read()
            vid.release()
            tfile.close()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = frame
            else:
                st.error("Failed to read video file")
                return
        else:
            # Handle image file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display image or video frame in Streamlit
        st.image(img, use_column_width=True, channels='RGB')

        # We use OpenCV to detect mouse clicks on the image
        if st.button('Pick Points'):
            points = pick_points(img)
            if points:
                st.write(f"Selected Points: {points}")

def pick_points(image):
    """ OpenCV routine to pick points on an image """
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            points.append((x, y))
            cv2.imshow("image", image)

    cv2.imshow('image', image)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points

if __name__ == "__main__":
    main()
