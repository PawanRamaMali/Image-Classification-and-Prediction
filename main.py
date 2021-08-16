# Required libraries
import os
import sys
import tensorflow as tf


# Import mrcnn libraries
import skimage
import numpy as np
from functions import readJson, validate_result_directory
from mrcnn.config import Config
from mrcnn import visualize
import mrcnn.model as modellib


ROOT_DIR = os.path.abspath('')
sys.path.append(ROOT_DIR)
print(f'ROOT_DIR: {ROOT_DIR}')

# Configuration
model_path = "./coco/mask_rcnn_coco.h5"
json_path = "./coco/COCO_defualt_config.json"
args_IMAGE_MIN_DIM = 1024
args_IMAGE_MAX_DIM = 2048
args_DETECTION_MIN_CONFIDENCE = 0.7
Test_dir = "./images"
output_path = "./results"

def predict_image(name):

    config_data = readJson(model_path, json_path)
    class_names = config_data['class_names']

    class InferenceConfig(Config):
        # Give the configuration a recognizable name
        NAME = "Inference_config"

        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_MIN_DIM = args_IMAGE_MIN_DIM
        IMAGE_MAX_DIM = args_IMAGE_MAX_DIM
        DETECTION_MIN_CONFIDENCE = args_DETECTION_MIN_CONFIDENCE

        # Number of classes (including background)
        NUM_CLASSES = len(class_names)  # background + classes

        # Matterport originally used resnet101
        BACKBONE = 'resnet101'

        # more advance configuration
        RPN_ANCHOR_SCALES = config_data['RPN_ANCHOR_SCALES']
        MAX_GT_INSTANCES = 100
        POST_NMS_ROIS_INFERENCE = 500

        # new added
        COMPUTE_BACKBONE_SHAPE = config_data['COMPUTE_BACKBONE_SHAPE']
        BACKBONE_STRIDES = config_data['BACKBONE_STRIDES']
        FPN_CLASSIF_FC_LAYERS_SIZE = config_data['FPN_CLASSIF_FC_LAYERS_SIZE']
        TOP_DOWN_PYRAMID_SIZE = config_data['TOP_DOWN_PYRAMID_SIZE']
        RPN_ANCHOR_RATIOS = config_data['RPN_ANCHOR_RATIOS']
        RPN_ANCHOR_STRIDE = config_data['RPN_ANCHOR_STRIDE']
        RPN_NMS_THRESHOLD = config_data['RPN_NMS_THRESHOLD']
        RPN_TRAIN_ANCHORS_PER_IMAGE = config_data['RPN_TRAIN_ANCHORS_PER_IMAGE']
        PRE_NMS_LIMIT = config_data['PRE_NMS_LIMIT']
        POST_NMS_ROIS_TRAINING = config_data['POST_NMS_ROIS_TRAINING']
        POST_NMS_ROIS_INFERENCE = config_data['POST_NMS_ROIS_INFERENCE']
        USE_MINI_MASK = config_data['USE_MINI_MASK']
        MINI_MASK_SHAPE = config_data['MINI_MASK_SHAPE']
        IMAGE_RESIZE_MODE = config_data['IMAGE_RESIZE_MODE']
        IMAGE_MIN_SCALE = config_data['IMAGE_MIN_SCALE']
        IMAGE_CHANNEL_COUNT = config_data['IMAGE_CHANNEL_COUNT']
        MEAN_PIXEL = config_data['MEAN_PIXEL']
        TRAIN_ROIS_PER_IMAGE = config_data['TRAIN_ROIS_PER_IMAGE']
        ROI_POSITIVE_RATIO = config_data['ROI_POSITIVE_RATIO']
        POOL_SIZE = config_data['POOL_SIZE']
        MASK_POOL_SIZE = config_data['MASK_POOL_SIZE']
        MASK_SHAPE = config_data['MASK_SHAPE']
        RPN_BBOX_STD_DEV = config_data['RPN_BBOX_STD_DEV']
        BBOX_STD_DEV = config_data['BBOX_STD_DEV']
        DETECTION_MAX_INSTANCES = config_data['DETECTION_MAX_INSTANCES']
        DETECTION_NMS_THRESHOLD = config_data['DETECTION_NMS_THRESHOLD']
        LEARNING_RATE = config_data['LEARNING_RATE']
        LEARNING_MOMENTUM = config_data['LEARNING_MOMENTUM']
        WEIGHT_DECAY = config_data['WEIGHT_DECAY']
        LOSS_WEIGHTS = config_data['LOSS_WEIGHTS']
        USE_RPN_ROIS = config_data['USE_RPN_ROIS']
        TRAIN_BN = config_data['TRAIN_BN']
        GRADIENT_CLIP_NORM = config_data['GRADIENT_CLIP_NORM']

    inference_config = InferenceConfig()

    image_path = "./images/2516944023_d00345997d_z.jpg"
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)

    DEVICE = "/device:CPU:0"

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir='')

        # Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

    # create results folder
    save_folder, folder_name = validate_result_directory(model_path, output_path)

    if np.shape(img_arr)[2] > 3:
        new_img = Image.fromarray(img_arr)
        new_img = new_img.convert('RGB')
        img_arr = np.array(new_img)

    with tf.device(DEVICE):
        results = model.detect([img_arr])

    # save prediction
    r = results[0]

    class_names = {int(key): class_names[key] for key in class_names.keys()}

    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], figsize=(16, 16))

    # save image
    visualize.save_image(np.array(img), image_name="output", boxes=r['rois'], masks=r['masks'],
                         class_ids=r['class_ids'], scores=r['scores'],
                         class_names=class_names, save_dir=save_folder)

    print(f'This is The, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    predict_image('End')

