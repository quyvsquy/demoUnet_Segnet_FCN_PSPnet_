from keras_segmentation.models.all_models import model_from_name

#python3 -m keras_segmentation visualize_dataset --images_path="./demo/images_preped_test_small/" --segs_path="./demo/annotations_preped_test_small/" --n_classes=30
model_name = []
model = []
model_name.append("vgg_unet")
# model_name.append("vgg_pspnet")
# model_name.append("vgg_segnet")
# model_name.append("fcn_32_vgg")
checkpoints_path = []
checkpoints_path.append("./saveModel/new1/vgg_unet_1")
# checkpoints_path.append("./saveModel/new1/vgg_pspnet_1")
# checkpoints_path.append("./saveModel/vgg_segnet_1")
# checkpoints_path.append("./saveModel/fcn_32_vgg_1")

for ia in model_name:
    model.append(model_from_name[ia](n_classes=6 ,  input_height=512, input_width=512))

for ib,ia in enumerate(model):
    ia.train(
        batch_size = 2,
        epochs= 5,
        # steps_per_epoch = 367 // 5,
        # train_images =  "dataset1/images_prepped_train/",
        # train_annotations = "dataset1/annotations_prepped_train/",
        train_images =  "../test/example_dataset/train",
        train_annotations = "../test/example_dataset/mask_train",
        validate = True,
        val_images = "../val",
        val_annotations = "../mask_val",
        val_batch_size = 5,
        # val_steps_per_epoch = 512,
        gen_use_multiprocessing = True,
        checkpoints_path = checkpoints_path[ib] 
    )


