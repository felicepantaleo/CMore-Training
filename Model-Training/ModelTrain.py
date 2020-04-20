import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = 'Mask_RCNN-master'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist'

# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import MaskRCNN



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class CocoLikeDataset(utils.Dataset):

    def load_data(self, annotation_json, images_dir):

        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids


dataset_train = CocoLikeDataset()
dataset_train.load_data('train.json', '/coco/coco')
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data('val.json', '/coco/coco_val')
dataset_val.prepare()


# define a configuration for the model

class TrainConfig(Config):
	NAME = "Train_cfg"
	NUM_CLASSES = 1 + 1 + 1
	STEPS_PER_EPOCH = 10


config = TrainConfig()

model = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)
 



#Training
model = MaskRCNN(mode='training', model_dir='./', config=config)
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
model.save("ObjRec.h5")