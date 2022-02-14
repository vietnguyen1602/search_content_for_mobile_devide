# import thư viện
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import pickle
import numpy as np


# Ham tao model
def get_extract_model():
    vgg19_model = VGG19(weights="imagenet")
    extract_model = Model(inputs=vgg19_model.inputs, outputs = vgg19_model.get_layer("fc1").output)
    return extract_model

# Ham tien xu ly, chuyen doi hinh anh thanh tensor
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path):
    print("Xu ly : ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Trich dac trung
    vector = model.predict(img_tensor)[0]
    # Chuan hoa vector = chia chia L2 norm (tu google search)
    vector = vector / np.linalg.norm(vector)
    return vector


# Dinh nghia thu muc data

data_folder = "dataset"

# Khoi tao model
model = get_extract_model()

vectors = []
paths = []

for image_path in os.listdir(data_folder):
    # Noi full path
    image_path_full = os.path.join(data_folder, image_path)
    # Trich dac trung
    image_vector = extract_vector(model,image_path_full)
    # Add dac trung va full path vao list
    vectors.append(image_vector)
    paths.append(image_path_full)

# save vao file
vector_file = "vectors_VGG19.pkl"
path_file = "paths_VGG19.pkl"

pickle.dump(vectors, open(vector_file, "wb"))
pickle.dump(paths, open(path_file, "wb"))