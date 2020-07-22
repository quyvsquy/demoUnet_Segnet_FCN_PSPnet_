from keras_segmentation.models.all_models import model_from_name
import pickle

#python3 -m keras_segmentation visualize_dataset --images_path="./demo/images_preped_test_small/" --segs_path="./demo/annotations_preped_test_small/" --n_classes=30
dictModels = {}
dictModels[0] = "vgg_unet"
dictModels[1] = "vgg_pspnet"
dictModels[2] = "vgg_segnet"
dictModels[3] = "fcn_32_vgg"

for ia in range(len(dictModels)):
    t = model_from_name[dictModels[ia]](n_classes=12 ,  input_height=384, input_width=576)
    H = t.train(
        verify_dataset = False,
        epochs= 50,
        batch_size = 3,
        steps_per_epoch = 337 // 3,
        train_images =  "dataset1/train",
        train_annotations = "dataset1/ano_train/",
        validate = True,
        val_images = "dataset1/val",
        val_annotations = "dataset1/ano_val",
        val_batch_size = 1,
        val_steps_per_epoch = 30 // 1,
        # gen_use_multiprocessing = True,
        checkpoints_path = f"./saveModel/{dictModels[ia]}_1" 
    )
    pickle.dump(H.history,open(f"./saveModel/His/{dictModels[ia]}","wb"))

