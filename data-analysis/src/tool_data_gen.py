"""
This main would predict covid
"""
# %% Import
###############
##### LIB #####
###############
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import datetime

from PIL import Image

# ML:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchvision.models as models

import cv2
from skimage.measure import label, regionprops

# custom:
from icecream import ic

from enum import Enum, Flag, auto

# %%
#######################
##### LOCAL LIB #######
#######################
## USER DEFINED:
ABS_PATH = "/home/jx/JX_Project/affordable-DR-monitoring/data-analysis/" # Define ur absolute path here
ABS_DATA_DIRECTORY = "/home/jx/JX_Project/data/dr-dataset"
# FOLDER_TAG = "sample"
# FOLDER_TAG = "train"
FOLDER_TAG = "test"

## Custom Files:
def abspath(relative_path):
    return os.path.join(ABS_PATH, relative_path)

def abs_data_path(relative_path):
    return os.path.join(ABS_DATA_DIRECTORY, relative_path)


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(abspath("src"))

import jx_lib

# %%
# convert filename to absolute path:
def filename_to_abspath(filenames, tag):
    return [abs_data_path("{}/{}".format(tag, filename)) for filename in filenames]

TRAIN_DATA_LUT = {
    "[filename]": []
}
### Temporary injection:
# for i in [10,13,15,16,17]:
#     TRAIN_DATA_LUT["[filename]"].append("{0}_left.jpeg".format(i))
#     TRAIN_DATA_LUT["[filename]"].append("{0}_right.jpeg".format(i))

TRAIN_DATA_LUT["[filename]"] = jx_lib.get_file_names(DIR=abs_data_path(FOLDER_TAG), file_end=".jpeg")
TRAIN_DATA_LUT["img_abs_path"] = filename_to_abspath(filenames=TRAIN_DATA_LUT["[filename]"], tag=FOLDER_TAG)

print("> Found Total: {} jpeg images!".format(len(TRAIN_DATA_LUT["[filename]"])))
# ic(TRAIN_DATA_LUT)
# %% USER DEFINE ----- ----- ----- ----- ----- -----
#######################
##### PREFERENCE ######
#######################
TRAIN_NEW_IMG_SIZE = (320,320)
TEST_NEW_IMG_SIZE = TRAIN_NEW_IMG_SIZE # None for original size
TRAIN_TEST_SPLIT = 0.8 # None

class ENUM_FEATURE_FLAGS(Enum):
    # pre-process
    HISTOGRAM_EQUALIZATION  = "EQ" 
    GAUSSIAN_INVERSION      = "GI"
    # Augmentation
    RANDOM_ROTATION         = "RR"
    RANDOM_ZOOM             = "RZ"


# %% image conversion function: ----- ----- ----- ----- ----- -----
######################
##### FUNCTIONS ######
######################
def img_batch_conversion(
    PATH_LUT:Dict,
    flags=[],
    EXPECT_DIM = [320,320], #Eye , Output-after-padding
    THRESHOLD_CROP_RATE  = 3,
):
    OUT_FILENAME_FULL = "{}-(crop_{}_{})".format(FOLDER_TAG, "_".join([x.value for x in flags]), EXPECT_DIM)
    ic(OUT_FILENAME_FULL)
    OUT_DIR = abs_data_path(OUT_FILENAME_FULL)        

    jx_lib.create_folder(DIR=OUT_DIR)
    FAILED_LOG_PATH = os.path.join(OUT_DIR,"failed.txt")
    SUCCESS_LOG_PATH = os.path.join(OUT_DIR,"success.txt")

    def log_failed(filename):
        with open(FAILED_LOG_PATH, "w") as f:
            f.write("{}\n".format(filename))

    def log_success(filename):
        with open(SUCCESS_LOG_PATH, "w") as f:
            f.write("{}\n".format(filename))

    counter = 0
    for img_path, file_name in zip(PATH_LUT["img_abs_path"], PATH_LUT["[filename]"]):
        counter += 1
        out_path = "{}/{}".format(OUT_DIR, file_name)
        img = cv2.imread(img_path)

        if img is None:
            continue # SKIP

        ### Pre-augmentation:
        if ENUM_FEATURE_FLAGS.RANDOM_ROTATION in flags:
            def random_rotation(img, angle):
                angle = int(random.uniform(-angle, angle))
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
                img = cv2.warpAffine(img, M, (w, h))
                return img
            img = random_rotation(img, 90)

        if ENUM_FEATURE_FLAGS.RANDOM_ZOOM in flags:
            # assert("ERROR: to be implemented!")
            # define operator:
            def fill(img, h, w):
                img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
                return img
            def random_zoom_crop(img, value):
                if value > 1 or value < 0:
                    print('Value for zoom should be less than 1 and greater than 0')
                    return img
                value = random.uniform(value, 1)
                h, w = img.shape[:2]
                h_taken = int(value*h)
                w_taken = int(value*w)
                h_start = random.randint(0, h-h_taken)
                w_start = random.randint(0, w-w_taken)
                img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
                img = fill(img, h, w)
                return img            
            img = random_zoom_crop(img, 0.8)

        ### Find Bounding Box
        img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 =  cv2.GaussianBlur(img_gray, (11, 11), 0)
        img_sample_color = np.mean(img2)/THRESHOLD_CROP_RATE

        img_0 = img
        mask_0 = img2 > img_sample_color
        print("\r   >[{}/{}] sampled_threshold:{}".format(counter,len(PATH_LUT["img_abs_path"]),img_sample_color),  end='')

        h,w,c = np.shape(img_0)
        lbl_0 = label(mask_0) 
        props = regionprops(lbl_0)

        if len(props) == 0:
            assert("Error Finding A Bounding Box") 
            log_failed(file_name)
            continue # SKIP

        prop_first = props[0]
        # ic(prop_first.bbox)

        (y0, x0, y1, x1) = prop_first.bbox

        height = y1 - y0
        width = x1 - x0
        cy = int(h/2) if height == h else int(y0+(height)/2) 
        cx = int(w/2) if width == w else int(x0+(width)/2)
        R_min = int(min(height, width)/2) - 10

        if R_min <= 200:
            print("{}: R_min:{} Error cropping A Circle: too small!".format(file_name, R_min))
            log_failed(file_name)
            continue # SKIP

        # ic(h,w,c, height, width, cx, cy, R_min)        
        ### Circular Cropping
        # Circular Cropping
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask,(cx, cy),R_min,1,-1)
        masked_img = cv2.bitwise_and(img_0,img_0,mask = mask)

        # Remap
        Cropped_image = masked_img[cy-R_min:cy+R_min, cx-R_min:cx+R_min, :]
        # ic(np.shape(Cropped_image))

        ## Equalization??
        if ENUM_FEATURE_FLAGS.HISTOGRAM_EQUALIZATION in flags:
            img_yuv = cv2.cvtColor(Cropped_image, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            Cropped_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        ### Apply Filters On Original Dataset: (Trick on competition)
        if ENUM_FEATURE_FLAGS.GAUSSIAN_INVERSION in flags:
            Cropped_image = cv2.addWeighted(Cropped_image,4, cv2.GaussianBlur(Cropped_image , (0,0) , R_min/10) ,-4 ,128)

        ### crop again:
        mask = np.zeros((R_min+R_min, R_min+R_min), dtype=np.uint8)
        cv2.circle(mask,(R_min, R_min),R_min,1,-1)
        Cropped_image = cv2.bitwise_and(Cropped_image,Cropped_image,mask = mask)

        ### Resizing:
        es, fs = EXPECT_DIM
        delta = int((fs-es)/2)
        EYE_SIZE = (es, es)
        FINAL_SIZE = (fs,fs,3) 

        resize_image = cv2.resize(Cropped_image, EYE_SIZE, interpolation=cv2.INTER_LINEAR)

        # add padding
        output_image = np.zeros(FINAL_SIZE, dtype=np.uint8)
        if delta <= 0:
            output_image = resize_image
        else:
            output_image[delta:es+delta,delta:es+delta,:] = resize_image


        # print(out_path)
        cv2.imwrite(out_path, output_image)
        # break
        log_success(out_path)

    return

# %%
#%% MAIN
def main():
    ## MAIN:
    img_batch_conversion(
        PATH_LUT=TRAIN_DATA_LUT, 
        flags=[
            # ENUM_FEATURE_FLAGS.GAUSSIAN_INVERSION, 
            # ENUM_FEATURE_FLAGS.HISTOGRAM_EQUALIZATION, # somehow is garbage , so do not use
        ],
        EXPECT_DIM = [320,320], #Eye , Output-after-padding        
        THRESHOLD_CROP_RATE  = 3,
    )

main()