from keras_segmentation.models.all_models import model_from_name

#python3 -m keras_segmentation visualize_dataset --images_path="./demo/images_preped_test_small/" --segs_path="./demo/annotations_preped_test_small/" --n_classes=30
model_name = []
model = []
model_name.append("vgg_unet")
model_name.append("vgg_pspnet")
model_name.append("vgg_segnet")
model_name.append("fcn_32_vgg")
checkpoints_path = []
checkpoints_path.append("./saveModel/vgg_unet_1")
checkpoints_path.append("./saveModel/new1/vgg_pspnet_1")
checkpoints_path.append("./saveModel/vgg_segnet_1")
checkpoints_path.append("./saveModel/fcn_32_vgg_1")

for ia in model_name:
    model.append(model_from_name[ia](n_classes=6 ,  input_height=384, input_width=576))

for ib,ia in enumerate(model):
    ia.train(
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


