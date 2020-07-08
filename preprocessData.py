import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import os
import seaborn as sns
import random
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# num = "00000"
# list_num = []
# for i in range(10):
#     temp = "0000" + str(i)
#     list_num.append(temp)

# for i in range(10,100):
#     temp = "000" + str(i)
#     list_num.append(temp)

# for i in range(100,2000):
#     temp = "00" + str(i)
#     list_num.append(temp)

DATA_LOADER_SEED = 2

random.seed(DATA_LOADER_SEED)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(2000)]

def give_color_to_seg_img(seg,n_classes):
    seg_img = np.zeros_like(seg)
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += ((seg[:, :, 0] == c)* (class_colors[c][0])).astype('uint8')
        seg_img[:,:,1] +=((seg[:, :, 0] == c)* (class_colors[c][1])).astype('uint8')
        seg_img[:,:,2] +=((seg[:, :, 0] == c)* (class_colors[c][2])).astype('uint8')
    return(seg_img)

dir_mask = "CelebAMask/mask/5"
dir_mask_tg = "Dataset/mask1"
folder_base = "CelebAMask/mask"
# list_label = ["_hair", "_l_brow", "_r_brow", "_u_lip", "_l_lip", "_l_eye", "_r_eye", "_skin" ]
dict_num_class = {}
dict_num_class["_skin"] = 1
dict_num_class["_l_eye"] = 2
dict_num_class["_r_eye"] = 2
dict_num_class["_l_brow"] = 3
dict_num_class["_r_brow"] = 3
dict_num_class["_u_lip"] = 4
dict_num_class["_l_lip"] = 4
dict_num_class["_hair"] = 5
list_label = ['_skin', '_l_eye', '_r_eye', '_l_brow', '_r_brow', '_u_lip', '_l_lip', '_hair']


for k in range(4000,30000):
    folder_num = k // 2000
    outImg = np.zeros((512,512) , np.uint8)
    for label in list_label:
        img_path = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + label + '.png')
        if os.path.isfile(img_path):
            temp_img = cv2.imread(img_path)
            outImg[temp_img[:,:,0]!=0] = dict_num_class[label]
    out_path = f"{dir_mask_tg}/{k}.png"
    print(out_path)
    cv2.imwrite(out_path,outImg)


for ia in range(20000,20010):
    if os.path.isfile(f"{dir_mask_tg}/{ia}.png"):
        # img = mpimg.imread(f"{dir_mask_tg}/{ia}.png")
        img = cv2.imread(f"{dir_mask_tg}/{ia}.png")
        dd = give_color_to_seg_img(img,6)
        dd = cv2.cvtColor(dd, cv2.COLOR_BGR2RGB)
        # print(dd)
        # cv2.imshow("cc",dd)
        # cv2.waitKey(0)
        plt.title(f"{dir_mask_tg}/{ia}.png")
        plt.imshow(dd)
        plt.pause(0.001)
        break
plt.show()
# dd = cv2.imread("cc.png")

# dd = give_color_to_seg_img(dd,9)
# cv2.imshow("cc",dd)
# # sleep(10)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# path_img = "CelebAMask/image/"
# for ia in range(2000):
#     # print(f"{path_img}{ia}.jpg")
#     img = cv2.imread(f"{path_img}{ia}.jpg")
#     # print(img.shape)
#     img_resize = cv2.resize(img, (512,512))
#     cv2.imwrite(f'Dataset/img/{ia}.jpg',img_resize)


    
exit(0)