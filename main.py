import streamlit as st
import cv2
import numpy as np

def main():
    st.title("選取廣告位置")
    uploaded_file = st.file_uploader("上傳要插入廣告之照片...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display image in Streamlit
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