import os
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def check_images(image_path):
    """
    Check images in path. If error image then remove
    :param image_path:
    :return:
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False


def dict_from_path(root_img_path, categories):
    files = []
    for folder in os.listdir(root_img_path):
        if folder.split("_")[0] in categories:
            path = os.path.join(root_img_path, folder)
            list_dir = [(f"{path}/{name}") for name in os.listdir(path) if name.endswith((".jpg", ".png", ".jpeg"))]
            for file in list_dir:
                if check_images(file):
                    files.append(file)
    return dict(enumerate(files))

def show_imgs(query, f_ids, file_path):
    """
      It takes in a query image and a list of filepaths to images, and displays the query image and the
      top 6 images from the list of filepaths

      :param query: the image we want to find similar images to
      :param f_ids: the list of file ids of the images that are most similar to the query image
      """
    plt.imshow(query)
    fig = plt.figure(figsize=(12, 12))
    columns = 3
    rows = 2
    for i in range(1, columns*rows +1):
        img = mpimg.imread(file_path[f_ids[i-1]])
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

def inference(query_path, retrieval_func, index, k_top = 7):
    """
      Given a query image, we use the retrieval function to extract the feature vector of the query image.
      Then, we use the index to search for the top-k nearest neighbors of the query image

      :param query_path: the path to the query image
      :param retrieval_func: The function that will be used to retrieve the embedding of the query image
      :param index: the index object that we created earlier
      """
    query = retrieval_func(query_path).detach().numpy()
    scores, idx_image = index.search(query, k=k_top)
    return scores[0], idx_image[0]


