import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import random
import tensorflow_probability as tfp
import math


def augmentation(img,aug_par):
  
    """ This function augments the input images.
          Args:
              img (np.ndarray): input image.
              aug_par (list): includes two values for the augmentation parameters for vertical and horizontal flip, respectively.

          Returns:
              np.ndarray (): Returns the augmentated version of the image.
    """
  
    if aug_par[0] == 1:
        img = cv2.flip(img,0) #Vertical flip
    if aug_par[1] == 1:
        img = cv2.flip(img,1) #Horizontal flip
    
    return img

def ev_alignment(img, expo, gamma):
    return ((img ** gamma) * 2.0**(-1*expo))**(1/gamma)

def image_read(img_path,expo,gamma,train,img_size,aug_par):
  
    """ This function read and input image and returns the input image with its corresponding mask.
          Args:
              img_path (str): file address of the image.
              expo (int): exposure value for gamma correction.
              gamma (float): gamma value for gamma correction. Default value is 2.24.
              train (bool): a parameter for specifying the type of preprocess.
              img_size (tuple): the target size of the input image.
              aug_par (list): parameters for augmentation
          
          Returns:
              final_image (np.ndarray): a six channel image consisting of the image and its corrected version together.
              final_mask (np.ndarray): a two channel segmentation map produced from the input image.
    """

    img = cv2.cvtColor(cv2.imread(img_path,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
    
    if train:
        img = cv2.resize(img,img_size)
    
    if aug_par!=None:
        img = augmentation(img,aug_par)
    mask = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)[:,:,0]
    
    if 'short' in img_path:
        _,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif 'long' in img_path:
        _,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
    mask[mask>0] = 1
    mask = np.expand_dims(mask,-1)
    img_new = ev_alignment(img/255.0,expo,gamma)

    final_img = np.concatenate([img/255.0,img_new],2).astype(np.float32)
    final_mask = np.concatenate([mask,mask],2)
    
    return (final_img,final_mask)

def path_split(path,train,img_size,aug_par):
    
    """ This function reads input images in different exposures and returns the input images with their corresponding masks.
          Args:
              path (str): the address of the short exposure image.
              train (bool): a parameter for specifying the type of preprocess.
              img_size (tuple): the target size of the input image.
              aug_par (list): parameters for augmentation
          
          Returns:
              inputs (list): returns a list consists of the input images in the order of short, medium, and long exposure images.
              segmentations (list): returns a list consists of the masks corresponding to short and long exposure images, respectively.
    """
    
    sh_path = path
    me_path = path.replace('short','medium')
    lo_path = path.replace('short','long')
    exposures_path = path.replace('_short.png','_exposures.npy')

    exposure = np.load(exposures_path).astype(np.float32)
    floating_exposures = exposure - exposure[1]
    
    sh_img,short_mask = image_read(sh_path,floating_exposures[0],2.24,train,img_size,aug_par)
    lo_img,long_mask = image_read(lo_path,floating_exposures[-1],2.24,train,img_size,aug_par)
    
    me_img = cv2.cvtColor(cv2.imread(me_path,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)/255.0
    
    if train:
        me_img = cv2.resize(me_img,img_size)
    
    if aug_par!=None:
        me_img = augmentation(me_img,aug_par)
    
    me_img = np.concatenate([me_img,me_img],2).astype(np.float32)
    
    inputs = [sh_img,me_img,lo_img]
    segmentations = [short_mask,long_mask]

    return(inputs,segmentations)

def convert_path(path):
    
    """ This function converts the input path into the Ground Truth path.
          Args:
              path (str): the address of the short exposure image.
          
          Returns:
              HDR_path (str): returns the address of the Ground Truth.
              align_path (str): returns the address of the align ratio path.
    """
    
    HDR_path = path.replace('LDR','HDR')
    HDR_path = HDR_path.replace('_short','_gt')
    HDR_path = HDR_path.replace('short','')
    align_path = HDR_path.replace('gt.png','alignratio.npy')

    return (HDR_path,align_path)

def imread_uint16_png(image_path, alignratio_path,aug_par,train,img_size):
    # Load the align_ratio variable and ensure is in np.float32 precision
    align_ratio = np.load(alignratio_path).astype(np.float32)
    # Load image without changing bit depth and normalize by align ratio
    out = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio
    if train:
      out = cv2.resize(out,img_size)
    if aug_par!=None:
      out = augmentation(out,aug_par)
    return out

def imwrite_uint16_png(image_path, image, alignratio_path):
    image = image[14:1074,10:1910]
    image = tf.math.log(tf.cast(image,tf.float64)/(1.000000000000001-tf.cast(image,tf.float64)))
    image = np.array(tf.clip_by_value(image,0,tf.math.reduce_max(image),np.float64))
    align_ratio = (2 ** 16 - 1) / image.max()
    np.save(alignratio_path, align_ratio)
    uint16_image_gt = np.round(image * align_ratio).astype(np.uint16)
    cv2.imwrite(image_path, cv2.cvtColor(uint16_image_gt, cv2.COLOR_RGB2BGR))
    return None

class Create_Dataset(Sequence):
    
    def __init__(self, batch_size, input_img_paths,image_size=(256,256),stage = 'train',augment = True):
        self.batch_size = batch_size
        self.stages = {'train':False,'valid':False,'test':False}
        if stage =='train':
            self.stages['train'] = True
            self.img_size = image_size
        elif stage == 'valid':
            self.stages['valid'] = True
            self.img_size = (1088,1920)
        else:
            self.stages['test'] = True
            self.img_size = (1088,1920)
        self.augment = augment

        self.input_img_paths = input_img_paths
        
    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        if self.augment:
          aug_par=[np.random.randint(0,2) for i in range(2)]
        else:
          aug_par = None
        random.shuffle(self.input_img_paths)
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        if self.stages['train']:
          short = np.zeros((self.batch_size,) + self.img_size + (6,), dtype="float32")
          medium = np.zeros((self.batch_size,) + self.img_size + (6,), dtype="float32")
          long = np.zeros((self.batch_size,) + self.img_size + (6,), dtype="float32")
          short_mask = np.zeros((self.batch_size,) + self.img_size + (2,), dtype="uint8")
          long_mask = np.zeros((self.batch_size,) + self.img_size + (2,), dtype="uint8")
          
          out = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        else:
          short = np.zeros((self.batch_size,) + self.img_size + (6,), dtype="float32")
          medium = np.zeros((self.batch_size,) + self.img_size + (6,), dtype="float32")
          long = np.zeros((self.batch_size,) + self.img_size + (6,), dtype="float32")
          short_mask = np.zeros((self.batch_size,) + self.img_size + (2,), dtype="uint8")
          long_mask = np.zeros((self.batch_size,) + self.img_size + (2,), dtype="uint8")
          
          out = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            
            input_images,segmentations = path_split(path,self.stages['train'],self.img_size,aug_par)
            if self.stages['train']:
                hdr_path,aligh_path = convert_path (path)
                final_out = imread_uint16_png(hdr_path,aligh_path,aug_par,self.stages['train'],self.img_size)
                final_out_sigmoid = np.array(tf.math.sigmoid(final_out))
                
                short[j] = input_images[0]
                medium[j] = input_images[1]
                long[j] = input_images[2]
                short_mask[j] = segmentations[0]
                long_mask[j] = segmentations[1]
              
                out[j] = final_out_sigmoid
            elif self.stages['valid']:
                hdr_path,aligh_path = convert_path (path)
                final_out = imread_uint16_png(hdr_path,aligh_path,aug_par,self.stages['train'],self.img_size)
                final_out_sigmoid = np.array(tf.math.sigmoid(final_out))
                
                short[j] = cv2.copyMakeBorder(input_images[0],14,14,10,10,cv2.BORDER_CONSTANT)
                medium[j] = cv2.copyMakeBorder(input_images[1],14,14,10,10,cv2.BORDER_CONSTANT)
                long[j] = cv2.copyMakeBorder(input_images[2],14,14,10,10,cv2.BORDER_CONSTANT)
                short_mask[j] = cv2.copyMakeBorder(segmentations[0],14,14,10,10,cv2.BORDER_CONSTANT)
                long_mask[j] = cv2.copyMakeBorder(segmentations[1],14,14,10,10,cv2.BORDER_CONSTANT)
                
                out[j] = cv2.copyMakeBorder(final_out_sigmoid,14,14,10,10,cv2.BORDER_CONSTANT)
            else:
                '''to unpad the outpt I need to use the following code:
                unpad_image = padded_image[14:1074,10:1910]
                '''
                short[j] = cv2.copyMakeBorder(input_images[0],14,14,10,10,cv2.BORDER_CONSTANT)
                medium[j] = cv2.copyMakeBorder(input_images[1],14,14,10,10,cv2.BORDER_CONSTANT)
                long[j] = cv2.copyMakeBorder(input_images[2],14,14,10,10,cv2.BORDER_CONSTANT)
                short_mask[j] = cv2.copyMakeBorder(segmentations[0],14,14,10,10,cv2.BORDER_CONSTANT)
                long_mask[j] = cv2.copyMakeBorder(segmentations[1],14,14,10,10,cv2.BORDER_CONSTANT)
        if self.stages['train'] or self.stages['valid']:
            X = [short,medium,long,short_mask,long_mask]
            Y = out
            return X, Y
        else:
            X = [short,medium,long,short_mask,long_mask]
            return X

def name_sorting(path):
    
    """ This function loads and sorts the input images' paths."""
    
    input_img_paths =     sorted([
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".png")
        ])
    return (input_img_paths)

def psnr_tanh_norm_mu_tonemap(hdr_nonlinear_ref, hdr_nonlinear_res, percentile=99, gamma=2.24):
    
    def psnr(ref, res):
      """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images whose ranges are [0-1].
          the mu-law tonemapped image.
          Args:
              im0 (np.ndarray): Image 0, should be of same shape and type as im1
              im1 (np.ndarray: Image 1,  should be of same shape and type as im0

          Returns:
              np.ndarray (): Returns the mean PSNR value for the complete image.

          """
      # return -10*np.log10(np.mean(np.power(im0-im1, 2)))
      return tf.image.psnr(ref,res,max_val=tf.math.reduce_max(ref))
    
    """ This function computes Peak Signal to Noise Ratio (PSNR) between the mu-law computed images from two non-linear
    HDR images.

            Args:
                hdr_nonlinear_ref (np.ndarray): HDR Reference Image after gamma correction, used for the percentile norm
                hdr_nonlinear_res (np.ndarray: HDR Estimated Image after gamma correction
                percentile (float): Percentile to to use for normalization
                gamma (float): Value used to linearized the non-linear images

            Returns:
                np.ndarray (): Returns the mean mu-law PSNR value for the complete image.

            """
    
    # Added 000000000000001 to the denominator to avoid any zero division problem.
    hdr_nonlinear_ref = tf.math.log(tf.cast(hdr_nonlinear_ref,tf.float64)/(1-tf.cast(hdr_nonlinear_ref,tf.float64)+.000000000000001))
    hdr_nonlinear_ref = tf.clip_by_value(hdr_nonlinear_ref,0,tf.math.reduce_max(hdr_nonlinear_ref))
    hdr_nonlinear_res = tf.math.log(tf.cast(hdr_nonlinear_res,tf.float64)/(1-tf.cast(hdr_nonlinear_res,tf.float64+.000000000000001)))
    hdr_nonlinear_res = tf.clip_by_value(hdr_nonlinear_res,0,tf.math.reduce_max(hdr_nonlinear_res))
    
    hdr_linear_ref = hdr_nonlinear_ref**gamma
    hdr_linear_res = hdr_nonlinear_res**gamma
    norm_perc = tfp.stats.percentile(hdr_linear_ref,percentile)
    return psnr(tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc), tanh_norm_mu_tonemap(hdr_linear_res, norm_perc))

def peak_signal_noise_ratio(y_true, y_pred):
    # Added 000000000000001 to the denominator to avoid any zero division problem.
    y_true_new = tf.math.log(tf.cast(y_true,tf.float64)/(1-tf.cast(y_true,tf.float64)+.000000000000001))
    y_true_new = tf.clip_by_value(y_true_new,0,tf.math.reduce_max(y_true_new))
    
    y_pred_new = tf.math.log(tf.cast(y_pred,tf.float64)/(1-tf.cast(y_pred,tf.float64)+.000000000000001))
    y_pred_new = tf.clip_by_value(y_pred_new,0,tf.math.reduce_max(y_pred_new))
    
    return tf.image.psnr(y_pred_new, y_true_new, max_val=tf.math.reduce_max(y_true_new))

def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    bounded_hdr = tf.tanh(hdr_image/norm_value)
    return mu_tonemap(bounded_hdr, mu)

def mu_tonemap(hdr_image, mu=5000):
    numerator = tf.math.log(1 + mu * hdr_image)
    denominator = math.log(1 + mu)
    return  numerator / denominator

if __name__ == '__main__':
    a = name_sorting('/dataset/Train/val/LDR/short/')
    f = Create_Dataset(16,a,(512,512),stage='valid',augment=False)
    f.__getitem__(0)
    