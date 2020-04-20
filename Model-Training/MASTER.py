import os
os.system("pip install -r Mask_RCNN-master/requirements.txt")
os.system("python unzip_dataset_creation.py")
os.system("python ModelTrain.py")