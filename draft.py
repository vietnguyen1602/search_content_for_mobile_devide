# import thu vien
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import pickle
import numpy as np
import json

# # Ham tao model
# def get_extract_model():
#     vgg16_model = VGG16(weights="imagenet")
#     extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output)
#     return extract_model
#
# # Ham tien xu ly, chuyen doi hinh anh thanh tensor
# def image_preprocess(img):
#     img = img.resize((224, 224))
#     img = img.convert("RGB")
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return x
#
# def extract_vector(model, image_path):
#     print("Xu ly : ", image_path)
#     img = Image.open(image_path)
#     img_tensor = image_preprocess(img)
#
#     # Trich dac trung
#     vector = model.predict(img_tensor)[0]
#     # Chuan hoa vector = chia chia L2 norm (tu google search)
#     vector = vector / np.linalg.norm(vector)
#     return vector
#
#
#
#
# # Dinh nghia thu muc data
#
# data_folder = "dataset"
# text_folder = "lable"
# # Khoi tao model
# model = get_extract_model()
#
# vectors = []
# paths = []
# texts = []
#
# for image_path in os.listdir(data_folder):
#     # Noi full path
#     image_path_full = os.path.join(data_folder, image_path + ".txt")
#     # Trich dac trung
#     #image_vector = extract_vector(model, image_path_full)
#     # Add dac trung va full path vao list
#     #vectors.append(image_vector)
#     texts.append(image_path_full)
#
# # save vao file
#vector_file = "vectors.pkl"
#path_file = "paths.pkl"
# text_file = "text.pkl"
# pickle.dump(vectors, open(vector_file, "wb"))
# pickle.dump(paths, open(path_file, "wb"))
# pickle.dump(texts, open(text_file, "wb"))
path = pickle.load(open("paths.pkl", "rb"))
text = pickle.load(open("text.pkl", "rb"))
arr = []
for i in range(len(path)):
    f = open(text[i], 'r')
    url = f.read()
    data = {
    "path": path[i],
    "url": url
    }
    arr.append(data)
    with open("data_file.json", "w") as write_file:
        json.dump(arr, write_file)

# f = open(text[0], 'r')
# data = f.read()
# print(data)
# with open("data_file.json", "r") as read_file:
#     data = json.load(read_file)
# print(data[0]['url'])
# print(type(data[0]['url']))