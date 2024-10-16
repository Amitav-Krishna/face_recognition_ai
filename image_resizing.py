import cv2
import os

def resize_images(image_folder, target_size=(160, 160)):
    for person in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person)
        if os.path.isdir(person_folder):
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path)
                resized_img = cv2.resize(img, target_size)
                cv2.imwrite(img_path, resized_img)
resize_images('test/')
