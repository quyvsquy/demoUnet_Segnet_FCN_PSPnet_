from keras_segmentation.predict import predict_multiple, evaluate, model_from_checkpoint_path
from keras_segmentation.data_utils.visualize_dataset import visualize_segmentation_dataset
import cv2

checkpoints_path = []
# checkpoints_path.append("./saveModel/new/vgg_unet_1")
# checkpoints_path.append("./saveModel/new/vgg_pspnet_1")
# checkpoints_path.append("./saveModel/new/vgg_segnet_1")
checkpoints_path.append("./saveModel/fcn_32_vgg_1")

# for ia in checkpoints_path:
# 	predict_multiple(
# 		checkpoints_path=ia,
# 		inp_dir="demo/images_preped_test_small",
# 		out_dir="demo/output/New"
# 	)

# view summary
# for ia in checkpoints_path:
# 	model = model_from_checkpoint_path(ia)
# 	print(model.summary())

# visuallize
# for seg_img, seg_path in visualize_segmentation_dataset(images_path="demo/images_preped_test_small/",segs_path="demo/annotations_preped_test_small/",n_classes=30,no_show=True):
# 	cv2.imwrite(f"demo/ground_truth/{seg_path.split('/')[-1]}", seg_img)


# for ia in checkpoints_path:
# 	output = evaluate(
# 		checkpoints_path=ia,
# 		inp_images_dir="dataset1/images_prepped_test/",
# 		annotations_dir = "dataset1/annotations_prepped_test/",
# 	)
# 	print(output)