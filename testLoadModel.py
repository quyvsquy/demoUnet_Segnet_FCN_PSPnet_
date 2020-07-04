from keras_segmentation.predict import predict_multiple
from keras_segmentation.predict import evaluate
checkpoints_path = []
checkpoints_path.append("./saveModel/vgg_unet_1")
checkpoints_path.append("./saveModel/vgg_pspnet_1")
checkpoints_path.append("./saveModel/vgg_segnet_1")
checkpoints_path.append("./saveModel/fcn_32_vgg_1")

# for ia in checkpoints_path:
# 	predict_multiple(
# 		checkpoints_path=ia,
# 		inp_dir="demo/images_preped_test_small",
# 		out_dir="demo/output"
# 	)




for ia in checkpoints_path:
	output = evaluate(
		checkpoints_path=ia,
		inp_images_dir="dataset1/images_prepped_test/",
		annotations_dir = "dataset1/annotations_prepped_test/",
	)
	print(output)