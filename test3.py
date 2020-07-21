# import os
# import shutil 
# imgs_path = "../img"
# segs_path = "../mask1"
# train_path = "../dataset_30000/train"
# val_path = "../dataset_30000/val"
# test_path = "../dataset_30000/test"
# for ia in range(18000):
#     shutil.copyfile(f'{imgs_path}/{ia}.jpg',f"{train_path}/{ia}.jpg")
# for ia in range(18000,21000):
#     shutil.copyfile(f'{imgs_path}/{ia}.jpg',f"{train_path}/{ia}.jpg")
# for ia in range(21000,30000):
#     shutil.copyfile(f'{imgs_path}/{ia}.jpg',f"{test_path}/{ia}.jpg")


from keras_segmentation.data_utils.data_loader import *
from keras_segmentation.data_utils.visualize_dataset import *

img_path = "dataset1/train"
ano_img_path = "dataset1/ano_train"
# img_path = "dataset1/test"
# ano_img_path = "dataset1/ano_test"
img_path = "dataset1/val"
ano_img_path = "dataset1/ano_val"
img_path = "demo/test"
ano_img_path = "demo/ano_test"
# verify_segmentation_dataset(img_path,ano_img_path,12,False,False)

# 
# visuallize
visualize_segmentation_dataset(images_path=img_path,segs_path=ano_img_path,n_classes=12,no_show=False)
# for seg_img, seg_path in visualize_segmentation_dataset(images_path=img_path,segs_path=ano_img_path,n_classes=12,no_show=True):
# 	cv2.imwrite(f"demo/ano_colors/{seg_path.split('/')[-1]}", seg_img)

"""
0:sky
1:building
2:pole #cột đèn
3:road
4:sidewalk
5:vegetation
6:traffic light
7:fence
8:car
9:person
10:rider
11:static
"""