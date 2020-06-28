from keras_segmentation.models.all_models import model_from_name
# from keras_segmentation.predict import predict_multiple

#python3 -m keras_segmentation visualize_dataset --images_path="./demo/images_preped_test_small/" --segs_path="./demo/annotations_preped_test_small/" --n_classes=30
model_name = []
model = []
model_name.append("vgg_unet")
model_name.append("vgg_pspnet")
model_name.append("vgg_segnet")
model_name.append("fcn_32_vgg")
checkpoints_path = []
checkpoints_path.append("./saveModel/vgg_unet_1")
checkpoints_path.append("./saveModel/vgg_pspnet_1")
checkpoints_path.append("./saveModel/vgg_segnet_1")
checkpoints_path.append("./saveModel/fcn_32_vgg_1")

for ia in model_name:
    model.append(model_from_name[ia](n_classes=30 ,  input_height=384, input_width=576))

for ib,ia in enumerate(model):
    ia.train(
        train_images =  "dataset1/images_prepped_train/",
        train_annotations = "dataset1/annotations_prepped_train/",
        checkpoints_path = checkpoints_path[ib] , epochs=5
    )

# predict_multiple(
#     checkpoints_path=checkpoints_path[0],
#     inp_dir="demo/images_preped_test_small/",
#     out_dir="demo/output"
# )

