import timm
import sys
import torch
import numpy as np
import copy
import random
from PIL import Image
import faiss
from tqdm import tqdm, tqdm_notebook
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json


# check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#get model from timm
class Model():
    def __init__(self, model_name, size = (224,224)):
        self.model = timm.create_model(str(model_name), pretrained=True, num_classes = 0)
        self.size = size

    def preprocessing_image(self, image_path):
        image = Image.open(str(image_path)).resize(self.size)
        image = np.array(image, dtype=np.float32)
        if len (image.shape) == 2:
            image = np.tile(image[:, :, np.newaxis], 3)
        return torch.as_tensor(image).transpose(2,0)[None]

    def eval(self):
        return self.model.eval()


    def __call__(self, image_path):
        image = self.preprocessing_image(image_path)
        return self.model(image)




class My_faiss():
    def __init__(self, model_1, model_2, model_3):
        super().__init__()
        self.model_1 = Model(model_1)
        self.model_2 = Model(model_2)
        self.model_3 = Model(model_3)

    def normal(self, image_path):
        return self.model_1(image_path)

    def mean(self,image_path):
        """
            It takes an image path, runs it through three different models, concatenates the outputs, and then
            takes the mean of the concatenated outputs

            :param image_path: the path to the image you want to classify
            :return: The mean of the three models.
            """
        output1 = self.model_1(image_path)
        output2 = self.model_2(image_path)
        output3 = self.model_3(image_path)

        concat = torch.cat((output1, output2, output3), dim = 0)
        mean = torch.mean(concat, dim=0, keepdim=True)
        return mean

    def concat(self, image_path):
        """
           > The function takes an image path as input, and returns the concatenated output of three different
           models

           :param image_path: the path to the image
           :return: The concatenated output of the three models.
           """
        output1 = self.model_1(image_path)
        output2 = self.model_2(image_path)
        output3 = self.model_3(image_path)

        concat = torch.cat((output1, output2, output3), dim=1)
        return concat

    def __call__(self, retrieval_name, DB, DB_Aug):
        if retrieval_name == 'normal':
            num_index = 2048
            retrieval_func = self.normal
        elif retrieval_name == 'mean':
            num_index = 2048
            retrieval_func = self.mean
        elif retrieval_name == 'concat':
            num_index = 6144
            retrieval_func = self.concat
        else:
            raise Exception("Your way do not supported")

        self.index = faiss.IndexFlatL2(num_index)
        # feature extract with Image organic
        for img_index , img in DB.items():
            embedded = retrieval_func(img).detach().numpy()
            self.index.add(embedded)
        # feature extract with Image Augmentation
        for img_index , img in DB_Aug.items():
            embedded = retrieval_func(img).detach().numpy()
            self.index.add(embedded)
        return self.index