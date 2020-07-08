import os
import shutil 
imgs_path = "../img"
segs_path = "../mask1"
train_path = "../dataset_30000/train"
val_path = "../dataset_30000/val"
test_path = "../dataset_30000/test"
for ia in range(18000):
    shutil.copyfile(f'{imgs_path}/{ia}.jpg',f"{train_path}/{ia}.jpg")
for ia in range(18000,21000):
    shutil.copyfile(f'{imgs_path}/{ia}.jpg',f"{train_path}/{ia}.jpg")
for ia in range(21000,30000):
    shutil.copyfile(f'{imgs_path}/{ia}.jpg',f"{test_path}/{ia}.jpg")
