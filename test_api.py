from PIL import Image
import requests
from io import BytesIO
import math
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import pickle
import numpy as np
import json


def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model


# Ham tien xu ly, chuyen doi hinh anh thanh tensor
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_vector(model, img):
    img_tensor = image_preprocess(img)
    # Trich dac trung
    vector = model.predict(img_tensor)[0]
    # Chuan hoa vector = chia chia L2 norm (tu google search)
    vector = vector / np.linalg.norm(vector)
    return vector


def search_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # Khoi tao model
    model = get_extract_model()
    # Trich dac trung anh search
    search_vector = extract_vector(model, img)

    # Load vector
    vectors = pickle.load(open("vectors.pkl", "rb"))
    paths = pickle.load(open("paths.pkl", "rb"))
    with open("data_file.json", "r") as read_file:
        link = json.load(read_file)
    # Tinh khoang cach tu search_vector den tat ca cac vector
    distance = np.linalg.norm(vectors - search_vector, axis=1)

    # Sap xep va lay ra K vector co khoang cach ngan nhat
    K = 2
    ids = np.argsort(distance)[:K]
    # Tao oputput
    nearest_image = [(paths[id], link[id]['url']) for id in ids]
    link_arr = []
    for i in range(K):
        link_product = nearest_image[i][1]
        link_arr.append(link_product)
    link_json = {
        "url": link_arr
    }

    return link_json


