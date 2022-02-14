# search_content_for_mobile_devide
this is a quick project about CBIR
# Requirements
tensorflow 2.x
# How to use
1. Setup
```
pip install -r requirements.txt
```
2. Prepare your datasets
- VGG16
```
python store_vectors.py
```
- VGG19
```
python store_image_vgg19.py
```
you can use either
3. Run
- VGG16
```
python search_image.py
```
- VGG19
```
search_image_vgg19.py
```
4. Result
![](/testimage/results.jpg)
