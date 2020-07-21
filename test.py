import pickle
import matplotlib.pyplot as plt
import numpy as np
H = pickle.load(open("./saveModel/His/pspnet","rb"))
# print(list(H))
# print(H["loss"][19:40])
# print(H["val_loss"][19:40])
N = 50
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H["loss"], label="train_loss")
plt.plot(np.arange(0, N), H["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()