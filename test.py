import pickle
import matplotlib.pyplot as plt
import numpy as np
dictModels = {}
dictModels[0] = "vgg_unet"
dictModels[1] = "vgg_pspnet"
dictModels[2] = "vgg_segnet"
dictModels[3] = "fcn_32_vgg"
# H = pickle.load(open("./saveModel/His/vgg_unet","rb"))
# H = pickle.load(open("./saveModel/His/vgg_pspnet","rb"))
# H = pickle.load(open("./saveModel/His/vgg_segnet","rb"))
# print(list(H))
N = 50
plt.style.use("ggplot")
fig=plt.figure()
fig.subplots_adjust(hspace=0.117, wspace=0.03, left=0.017, bottom=0.055, right=0.98, top=1)
axes = fig.subplots(2, 2)
# axes[0,1].set_yticks(np.arange(0,4, 0.5))
# for ia in range(2):
#     for ib in range(2):
#         axes[ia,ib].set_xticks([])
#         axes[ia,ib].set_yticks([])
# plt.figure()

dem = 0
for ia in range(2):
    for ib in range(2):
        try:
            H = pickle.load(open(f"./saveModel/His/{dictModels[dem]}","rb"))
            axes[ia,ib].plot(np.arange(0, N), H["loss"], label="train_loss")
            axes[ia,ib].plot(np.arange(0, N), H["val_loss"], label="val_loss")
            axes[ia,ib].set_yticks(np.arange(0,4, 0.5))
            # print(H["loss"][-1], H["val_loss"][-1])
            print(H["val_loss"][18:21])
            axes[ia,ib].legend(loc="upper right", title=dictModels[dem])
            # axes[ia,ib].plot(np.arange(0, N), H["accuracy"], label="train_acc")
            # axes[ia,ib].plot(np.arange(0, N), H["val_accuracy"], label="val_acc")
            # axes[ia,ib].set_yticks(np.arange(0,1.1, 0.1))
            # axes[ia,ib].legend(loc="lower right", title=dictModels[dem])
            axes[ia,ib].set_xlabel("Epoch #")
            dem += 1
        except:
            pass
# axes[0,0].plot(np.arange(0, N), H["loss"], label="train_loss")
# axes[0,0].plot(np.arange(0, N), H["val_loss"], label="val_loss")
# axes[0,0].plot(np.arange(0, N), H["accuracy"], label="train_acc")
# axes[0,0].plot(np.arange(0, N), H["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# axes[0,0].legend(loc="upper right")
plt.show()