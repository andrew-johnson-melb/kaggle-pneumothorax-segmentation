import pydicom
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

from mask_functions import rle2mask, mask2rle


def read_dicom_file(path, metadata_cols=()):
    """Read dicom file and return img, the base file name,
    and the relevant meta data"""
    data = pydicom.read_file(path)
    meta_data = {}
    for field in metadata_cols:
        meta_data[field] = getattr(data, field)
    name = Path(path).stem
    return data.pixel_array, name, meta_data


class PreprocessorImages:
    
    def __init__(self, resize_dim=512, meta_data_cols=None,
                 train_in_path='../data/dicom-images/train/', 
                 test_in_path='../data/dicom-images/test/', 
                 mask_in_path='../data/train-rle.csv', 
                 train_out_path='../data/preprocessed_data/tmp/train/', 
                 test_out_path='../data/preprocessed_data/tmp/test/', 
                 mask_out_path='../data/preprocessed_data/tmp/train-masks/'):
        
        if meta_data_cols is None:
            self.meta_data_cols = ('PatientAge', 'PatientID', 'PatientSex', 'PatientOrientation', 'ViewPosition')
        
        self.train_in_path = train_in_path 
        self.test_in_path = test_in_path 
        self.train_out_path = train_out_path + f'size-{resize_dim}/'
        self.test_out_path = test_out_path + f'size-{resize_dim}/'
        self.mask_out_path = mask_out_path + f'size-{resize_dim}/'
        self.resize_dim = resize_dim
        
        # Make the dir's to save the data in
        os.makedirs(self.train_out_path, exist_ok=True)
        os.makedirs(self.test_out_path, exist_ok=True)
        os.makedirs(self.mask_out_path, exist_ok=True)
        
        # Load in the mask data.
        self.mask_info = pd.read_csv(mask_in_path)
        self.meta_data_collection = []
        
    def run(self, train_files, test_files):
        
        print("Running on the training set..")
        for path in tqdm(train_files):
            self.preprocess_dicom_files(path, self.train_out_path, self.mask_out_path, train_set=True)
            
        print("Running on the test set...")
        for path in tqdm(test_files):
            self.preprocess_dicom_files(path, self.test_out_path)
    
    def get_meta_data(self):
        return pd.DataFrame(self.meta_data_collection)
    
    def preprocess_dicom_files(self, file, out_path, out_path_mask=None, train_set=False):
        """Read in dicom file, resize it, and then save as png
        if the data is part of the train set, find the associated mask"""
        img, name, meta_data = read_dicom_file(file, self.meta_data_cols)
        img_resized = cv2.resize(img, (self.resize_dim, self.resize_dim))
        cv2.imwrite(f'{out_path}{name}.png', img_resized)
        flag = None
        if train_set:
            mask, flag = self.get_mask(file, width=img.shape[0], height=img.shape[1])
            mask = cv2.resize(mask, (self.resize_dim, self.resize_dim))
            # Add information on presence of issue to the meta data
            cv2.imwrite(f'{out_path_mask}{name}.png', mask)
        # Collect and store metadata for each dicom files.
        meta_data['label'] = flag
        meta_data['train_set'] = train_set
        meta_data['file_name'] = name
        self.meta_data_collection.append(meta_data)
        
        
    def get_mask(self, path, width, height, pixel_col=' EncodedPixels', missing_flag=['-1']):
        # This flag will indicate the type of mask
        flag = None
        name = Path(path).stem
        mask_values = self.mask_info[self.mask_info.ImageId == name][pixel_col].values
        
        if np.any((mask_values == missing_flag)) or (len(mask_values) == 0):
            # All zeros no detection
            flag = 'no-region'
            mask = np.zeros((width, height))
        elif len(mask_values) == 1:
            flag = 'single-region'
            # Single region in image
            mask = rle2mask(mask_values[0], width, height)
        elif len(mask_values) > 1:
            flag = 'multiple-regions'
            # Multiple masks in image.
            mask = np.zeros((width, height))
            for region in mask_values:
                mask += rle2mask(region, width,height)    
        else:
            print(f'error loading mask name={name} mask={mask}')
        return mask.T, flag
