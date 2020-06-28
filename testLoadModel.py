from keras_segmentation.predict import predict


path = "./saveModel/vgg_unet_1"
predict( 
	checkpoints_path=path, 
	inp="dataset1/images_prepped_test/0016E5_07965.png", 
	out_fname="output.png" 
)