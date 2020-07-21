from keras_segmentation.models.all_models import model_from_name

model_name = []
model = []
# model_name.append("vgg_unet")
# model_name.append("vgg_pspnet")
# model_name.append("vgg_segnet")
model_name.append("fcn_32_vgg")
checkpoints_path = []
# checkpoints_path.append("./saveModel/vgg_unet_1")
# checkpoints_path.append("./saveModel/new1/vgg_pspnet_1")
# checkpoints_path.append("./saveModel/vgg_segnet_1")
checkpoints_path.append("./saveModel/fcn_32_vgg_1")

for ia in model_name:
    model.append(model_from_name[ia](n_classes=6 ,  input_height=512, input_width=512))

for ib,ia in enumerate(model):
    ia.train(
        verify_dataset = False,
        batch_size = 5,
        epochs= 10,
        steps_per_epoch = 200,
        train_images =  "../datasmall/train",
        train_annotations = "../datasmall/mask_train",
        validate = True,
        val_images = "../datasmall/val",
        val_annotations = "../datasmall/mask_val",
        val_batch_size = 2,
        val_steps_per_epoch = 50,
        # gen_use_multiprocessing = True,
        checkpoints_path = checkpoints_path[ib] 
    )


