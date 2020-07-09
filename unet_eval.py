from keras_segmentation.predict import predict_multiple, evaluate, model_from_checkpoint_path
from keras_segmentation.data_utils.visualize_dataset import visualize_segmentation_dataset
import cv2

checkpoints_path = []
checkpoints_path.append("./saveModel/vgg_unet_1")

for ia in checkpoints_path:
	output = evaluate(
		checkpoints_path=ia,
		inp_images_dir="../datasmall/test",
		annotations_dir = "../datasmall/mask_test",
	)
	print(output)