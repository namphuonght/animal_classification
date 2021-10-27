# Đây là code để sinh ra ảnh tăng cường
# import các thư viện
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
from imutils import paths
import argparse

imageList = []
imageGenList = []
# Xây dựng các tham số thực thi dòng lệnh
# Use: python augmentation_demo.py -i <fiel ảnh> -o <folder lưu ảnh> -p <tiền tố file ảnh tăng cường>

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Đường dẫn input image")
ap.add_argument("-o", "--output", required=True, help="Đường dẫn output directory để lưu trữ ảnh cần tăng cường")
ap.add_argument("-p", "--prefix", type=str, default="image", help="File tiền tố output đã tăng cường")
args = vars(ap.parse_args())

# nạp ảnh đầu vào, convert it sang mảng NumPy array, rồi reshape nó
print("[INFO] Nạp image...")
imagePaths = np.array(list(paths.list_images(args["image"]))) #xác định số file trong dataset
idxs = np.random.randint(0, len(imagePaths), size=(50,)) # Trả về 10 idxs ngẫu nhiên
imagePaths = imagePaths[idxs]
print(imagePaths)
for i in imagePaths:
    image = load_img(i)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    imageList.append(image)


# Tạo bộ sinh ảnh tăng cường và khởi tạo tổng số ảnh được sinh ra
aug = ImageDataGenerator(channel_shift_range=0.4, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest",
                         validation_split=0.3, rescale=True)


# Sinh ảnh tăng cường
for i in imageList:
    print("[INFO] generating images...")
    imageGen = aug.flow(i, batch_size=1, save_to_dir=args["output"], save_prefix=args["prefix"], save_format="jpg")
    imageGenList.append(imageGen)

# Lặp qua các ảnh đã được tăng cường ảnh trong imageGen
for image in imageGenList:
    total = 0
    for i in image:
        # Tăng bộ đếm
        total += 1
        # Lặp 10 lần
        # 10 là Số ảnh được sinh ra, thay đổi giá trị này để có được số ảnh mong muốn
        if total == 10:
            break


