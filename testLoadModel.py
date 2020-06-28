from keras_segmentation.predict import predict_multiple
checkpoints_path = []
checkpoints_path.append("./saveModel/vgg_unet_1")
checkpoints_path.append("./saveModel/vgg_pspnet_1")
checkpoints_path.append("./saveModel/vgg_segnet_1")
checkpoints_path.append("./saveModel/fcn_32_vgg_1")

for ia in checkpoints_path:
	predict_multiple(
		checkpoints_path=ia,
		inp_dir="demo/images_preped_test_small/",
		out_dir="demo/output"
)
