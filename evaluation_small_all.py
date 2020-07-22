from keras_segmentation.predict import predict_multiple, evaluate, model_from_checkpoint_path
from keras_segmentation.data_utils.visualize_dataset import visualize_segmentation_dataset

checkpoints_path = []
checkpoints_path.append("./saveModel/vgg_unet_1")
# checkpoints_path.append("./saveModel/vgg_pspnet_1")
# checkpoints_path.append("./saveModel/vgg_segnet_1")
# checkpoints_path.append("./saveModel/fcn_32_vgg_1")
dictModels = {}
dictModels[0] = "vgg_unet"
dictModels[1] = "vgg_pspnet"
dictModels[2] = "vgg_segnet"
dictModels[3] = "fcn_32_vgg"

for ia in dictModels:
    output = evaluate(
        checkpoints_path=f"./saveModel/{dictModels[ia]}_1",
        inp_images_dir="./dataset1/test",
        annotations_dir = "./dataset1/ano_test",
    )
    print(dictModels[ia], end=" | ")
    print(output)