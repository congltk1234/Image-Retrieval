import torch.cuda

from utils import dict_from_path, show_imgs, inference, check_images
from models import  Model, My_faiss
from data_augmentation import  save_image
from PIL import Image
import numpy as np
import os





# the main paths
root_img_path = '/home/quannt/ALL_IN_ONE/Project/AI_Challenge'
img_aug_path = 'data_augmentation'
categories = ['scenery', 'furniture', 'animal', 'plant']
path_dict = dict_from_path(root_img_path=root_img_path, categories= categories)
aug_path_dict = save_image(root_img_path,img_aug_path,path_dict)
query_path = f'{root_img_path}/query.jpg'



# Creating a model object.
xception41_model = 'xception41'
resnest50d_model = 'resnet50'
adv_inception_model = 'adv_inception_v3'

# choose type model: normal, mean, concat

def train(type_model = 'normal',
          model1 = xception41_model,
          model2 = resnest50d_model,
          model3 = adv_inception_model,
          DB = path_dict,
          DB_Aug = aug_path_dict):
    model = My_faiss(model_1=xception41_model, model_2=resnest50d_model, model_3=adv_inception_model)
    if type_model == 'normal':
        normal_idx = model(type_model, DB, DB_Aug)
        normal_scores, normal_ids = inference(query_path, model.normal, normal_idx)
        return normal_scores, normal_ids
    elif type_model == 'mean':
        mean_idx = model(type_model, DB, DB_Aug)
        mean_scores, mean_ids = inference(query_path, model.mean, mean_idx)
        return mean_scores, mean_ids
    elif type_model == 'concat':
        concat_idx = model(type_model, DB, DB_Aug)
        concat_scores, concat_ids = inference(query_path, model.concat, concat_idx)
        return concat_scores, concat_ids
    else:
        raise Exception("Your way do not supported")

scores, ids = train(type_model='normal', DB=path_dict, DB_Aug=aug_path_dict)
query_image = Image.open(query_path)
show_imgs(query_image, ids, path_dict)





