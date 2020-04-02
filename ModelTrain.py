import os
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from sys import platform



class TrainingDataset(Dataset):
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "object")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# fProcessing the images
		for filename in listdir(images_dir):
			image_id = filename[:-4]
			if image_id in ['00090']:
				continue
			if is_train and int(image_id) >= 150:
				continue
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 
	
	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		tree = ElementTree.parse(filename)
		root = tree.getroot()
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# getting image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
 
	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('object'))
		return masks, asarray(class_ids, dtype='int32')
 
	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
 
# define a configuration for the model
class TrainConfig(Config):
	NAME = "Train_cfg"
	# classes (background + ob1)
	NUM_CLASSES = 1 + 1
	STEPS_PER_EPOCH = 131
 
# prepare train set
train_set = TrainingDataset()
train_set.load_dataset("Mask_RCNN/Dataset",is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
test_set = TrainingDataset()
test_set.load_dataset("Mask_RCNN/Dataset",is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
config = TrainConfig()
config.display()

#Training
model = MaskRCNN(mode='training', model_dir='./', config=config)
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
model.save("ObjRec.h5")
