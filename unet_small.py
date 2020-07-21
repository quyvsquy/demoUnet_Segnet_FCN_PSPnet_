from keras_segmentation.models.all_models import model_from_name
import matplotlib.pyplot as plt
import pickle
import numpy as np
model_name = []
model = []
model_name.append("vgg_unet")
checkpoints_path = []
checkpoints_path.append("./saveModel/vgg_unet_1")

for ia in model_name:
    model.append(model_from_name[ia](n_classes=12,  input_height=384, input_width=576))
H = None
for ib,ia in enumerate(model):
    H = ia.train(
        verify_dataset = False,
        batch_size = 3,
        epochs= 10,
        steps_per_epoch = 337 // 3,
        train_images =  "dataset1/train",
        train_annotations = "dataset1/ano_train/",
        validate = True,
        val_images = "dataset1/val",
        val_annotations = "dataset1/ano_val",
        val_batch_size = 1,
        val_steps_per_epoch = 30 // 1,
        # gen_use_multiprocessing = True,   
        checkpoints_path = checkpoints_path[ib] 
    )
    pickle.dump(H.history,open(f"./saveModel/His/unet_{ia}","wb"))
