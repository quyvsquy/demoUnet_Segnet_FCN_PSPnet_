
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
listName = ['0016E5_07530','0016E5_08159']
listNameNet = ['vgg_unet','vgg_pspnet','vgg_segnet','fcn_32_vgg']
ground_truth = "demo/ground_truth"
pre_dict = "demo/output"
in_img = "demo/images_preped_test_small"

# fig=plt.figure(figsize=(16,5))
fig=plt.figure()
fig.subplots_adjust(hspace=0.01, wspace=0.01, left=0.02, bottom=0.02, right=0.98, top=0.98)
axes = fig.subplots(3, 5)
for ia in range(3):
    for ib in range(5):
        axes[ia,ib].set_xticks([])
        axes[ia,ib].set_yticks([])

axes[0,0].imshow(Image.open(f"{in_img}/{listName[0]}.png"))
axes[0,0].set_xlabel("input")
axes[0,1].imshow(Image.open(f"{in_img}/{listName[1]}.png"))
axes[0,1].set_xlabel("input")
for ia in range(2,5):
    axes[0,ia].axis('off')

axes[1,0].imshow(Image.open(f"{ground_truth}/{listName[0]}.png"))
axes[1,0].set_xlabel("ground_truth")
axes[2,0].imshow(Image.open(f"{ground_truth}/{listName[1]}.png"))
axes[2,0].set_xlabel("ground_truth")

for ia in listdir(pre_dict):
    if listName[0] in ia:
        for ib in range(1,5):
            axes[1,ib].imshow(Image.open(f"{pre_dict}/{listName[0]}_{listNameNet[ib-1]}_1.png"))
            axes[1,ib].set_xlabel(listNameNet[ib-1])
    elif listName[1] in ia:
        for ib in range(1,5):
            axes[2,ib].imshow(Image.open(f"{pre_dict}/{listName[1]}_{listNameNet[ib-1]}_1.png"))
            axes[2,ib].set_xlabel(listNameNet[ib-1])
    
plt.show()