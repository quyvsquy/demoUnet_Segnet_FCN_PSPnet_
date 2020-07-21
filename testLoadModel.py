from keras_segmentation.predict import predict_multiple, evaluate, model_from_checkpoint_path
from keras_segmentation.data_utils.visualize_dataset import visualize_segmentation_dataset
import cv2

checkpoints_path = []
checkpoints_path.append("./saveModel/vgg_unet_1")
checkpoints_path.append("./saveModel/vgg_pspnet_1")
checkpoints_path.append("./saveModel/vgg_segnet_1")
checkpoints_path.append("./saveModel/fcn_32_vgg_1")

for ia in checkpoints_path:
	predict_multiple(
		checkpoints_path=ia,
		inp_dir="./demo/test",
		out_dir="./demo/predict"
	)

# view summary
# for ia in checkpoints_path:
# 	model = model_from_checkpoint_path(ia)
# 	print(model.summary())

# visuallize
# visualize_segmentation_dataset(images_path="./test/example_dataset/images_prepped_train",segs_path="./test/example_dataset/annotations_prepped_train",n_classes=6,no_show=False)
# visualize_segmentation_dataset(images_path="demo/images_preped_test_small/",segs_path="demo/annotations_preped_test_small/",n_classes=6,no_show=True)
# for seg_img, seg_path in visualize_segmentation_dataset(images_path="demo/images_preped_test_small/",segs_path="demo/annotations_preped_test_small/",n_classes=6,no_show=True):
# 	cv2.imwrite(f"demo/ground_truth/{seg_path.split('/')[-1]}", seg_img)


# for ia in checkpoints_path:
# 	output = evaluate(
# 		checkpoints_path=ia,
# 		inp_images_dir="../datasmall/test",
# 		annotations_dir = "../datasmall/mask_test",
# 	)
# 	print(output)