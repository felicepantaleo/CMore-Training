import os
import urllib.request
import zipfile

orig=os.getcwd()
current=orig+"/coco.zip"
url = 'https://drive.google.com/uc?export=download&id=1Q7cMKvowHKI2Nxnl5k1ZqkM7ddsAA2-j'
urllib.request.urlretrieve(url, current)
dest=orig+"/coco"
with zipfile.ZipFile("coco.zip", 'r') as zip_ref:
    zip_ref.extractall(dest)

current=orig+"/Mask_RCNN-master/mask_rcnn_coco.h5"
url = 'https://drive.google.com/uc?export=download&id=1DgosDoxddKr0mq9sRxjALez8v9HEmVm1'
urllib.request.urlretrieve(url, current)

os.system("python convert_labelme_to_coco.py coco/coco")
os.rename(orig+"/trainval.json",orig+"/train.json")
os.system("python convert_labelme_to_coco.py coco/coco_val")
os.rename(orig+"/trainval.json",orig+"/val.json")