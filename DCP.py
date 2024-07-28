import streamlit as st
import PIL.Image as Image
import skimage.io as io
import numpy as np
import cv2
import io as io_bytes

from gf import guided_filter 

class HazeRemoval:
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.r = r
        self.eps = eps

    def open_image(self, img_path):
        img = Image.open(img_path)
        self.src = np.array(img).astype(np.double)/255.
        self.rows, self.cols, _ = self.src.shape
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
        self.Alight = np.zeros((3), dtype=np.double)
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
        self.dst = np.zeros_like(self.src, dtype=np.double)

    def open_image_array(self, img_array):
        self.src = img_array.astype(np.double)/255.
        self.rows, self.cols, _ = self.src.shape
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
        self.Alight = np.zeros((3), dtype=np.double)
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
        self.dst = np.zeros_like(self.src, dtype=np.double)

    def get_dark_channel(self):
        tmp = self.src.min(axis=2)
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - self.radius)
                rmax = min(i + self.radius, self.rows - 1)
                cmin = max(0, j - self.radius)
                cmax = min(j + self.radius, self.cols - 1)
                self.dark[i, j] = tmp[rmin:rmax + 1, cmin:cmax + 1].min()

    def get_air_light(self):
        flat = self.dark.flatten()
        flat.sort()
        num = int(self.rows * self.cols * 0.001)
        threshold = flat[-num]
        tmp = self.src[self.dark >= threshold]
        tmp.sort(axis=0)
        self.Alight = tmp[-num:, :].mean(axis=0)

    def get_transmission(self):
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - self.radius)
                rmax = min(i + self.radius, self.rows - 1)
                cmin = max(0, j - self.radius)
                cmax = min(j + self.radius, self.cols - 1)
                pixel = (self.src[rmin:rmax + 1, cmin:cmax + 1] / self.Alight).min()
                self.tran[i, j] = 1. - self.omega * pixel

    def guided_filter(self):
        self.gtran = guided_filter(self.src, self.tran, self.r, self.eps)

    def recover(self):
        self.gtran[self.gtran < self.t0] = self.t0
        t = self.gtran.reshape(*self.gtran.shape, 1).repeat(3, axis=2)
        self.dst = (self.src.astype(np.double) - self.Alight) / t + self.Alight
        self.dst *= 255
        self.dst[self.dst > 255] = 255
        self.dst[self.dst < 0] = 0
        self.dst = self.dst.astype(np.uint8)

    def process(self, img_path):
        self.open_image(img_path)
        self.get_dark_channel()
        self.get_air_light()
        self.get_transmission()
        self.guided_filter()
        self.recover()
        return self.dst

    def process_frame(self, frame):
        self.open_image_array(frame)
        self.get_dark_channel()
        self.get_air_light()
        self.get_transmission()
        self.guided_filter()
        self.recover()
        return self.dst

st.title("Dark Channel Prior Haze Removal Application")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    img = Image.open(uploaded_file)
    with col1:
        st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button("Remove Haze"):
        hr = HazeRemoval()
        dst = hr.process(uploaded_file)
        
        with col2:
            st.image(dst, caption='Dehazed Image', use_column_width=True)
        
        im = Image.fromarray(dst)
        buf = io_bytes.BytesIO()
        im.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Dehazed Image",
            data=byte_im,
            file_name="dehazed_image.png",
            mime="image/png"
        )
