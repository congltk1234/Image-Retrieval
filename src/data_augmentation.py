import numpy as np
import cv2
import imgaug.augmenters as iaa
import albumentations as A
import os
from utils import dict_from_path, show_imgs, inference, check_images

def image_aug (path_image, crop_scale = 0.95):

    image = cv2.imread(path_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    width = int(image.shape[0])
    height = int(image.shape[1])
    dim = (width, height)

    crop_scale = 0.95
    width_crop = width * crop_scale
    height_crop = height * crop_scale

    transform = A.Compose([
        A.RandomCrop(width= int(width_crop), height= int(height_crop)),
        A.Rotate(45),
        A.OneOf([A.HorizontalFlip(),
                 A.VerticalFlip()
                 ],p = 0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2,
                                   always_apply=False,
                                    p=0.2),
    ])

    # resize image
    img_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


    # Augment an image
    transformed = transform(image=img_resized)
    transformed_image = transformed["image"]

    return transformed_image


def save_image(root_img_path, des_path, path_image):
    dict_path = {}
    if os.path.exists(os.path.join(root_img_path, des_path)):
        pass
    else:
        os.mkdir(os.path.join(root_img_path, des_path))

    img_aug_path = os.path.join(root_img_path, des_path)

    # Change the current directory
    # to specified directory
    os.chdir(img_aug_path)

    for idx, values in path_image.items():
        file_name = values.split('/')[-1]

        # get data from augment function
        img = image_aug(values)

        if not os.path.exists(os.path.join(img_aug_path, str(file_name))):
        # save image after augmented
            cv2.imwrite(str(file_name),img)
            dict_path[idx] = os.path.join(img_aug_path, str(file_name))
        else:
            dict_path[idx] = os.path.join(img_aug_path, str(file_name))
    return dict_path




