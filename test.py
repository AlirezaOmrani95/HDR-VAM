import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils import peak_signal_noise_ratio, name_sorting,path_split,tanh_norm_mu_tonemap,Create_Dataset
from train import gpu_checker
from model import model

def save_raw(tensor, root_path,c):
    for i in range(tensor.shape[0]):
        path = os.path.join(root_path,f'{c}_output.png')
        image = tensor[i,14:1074,10:1910,:]
        image = tf.math.log(tf.cast(image,tf.float64)/(1-tf.cast(image,tf.float64)+.000000000000001))
        image = np.array(tf.clip_by_value(image,0,tf.math.reduce_max(image),np.float64))
        
        linear = image ** 2.24
        norm_perc = np.quantile(linear,.99)
        mu_pred = tanh_norm_mu_tonemap(linear, norm_perc)
        tf.keras.utils.save_img(path,mu_pred)
        c+=1
    return c

if __name__ == '__main__':
    print('---------Checking GPU---------\n')
    gpu_checker()
    
    batch_size = 2
    print('---------Reading Model---------\n')
    model_ = model()
    model_.compile(optimizer='adam', 
                    loss=[
                    'mae'
                    ],
                    metrics=[peak_signal_noise_ratio])

    if 'weights.h5' in os.listdir('./weight'):
        print('---------Loading Weights---------\n')
        model_.load_weights('./weight/weights.h5')


    test_dir = '/dataset/test/LDR/short/'
    out_path = './dataset/test/output/'
    test_input_img_paths = name_sorting(test_dir)

    print('---------Making Dataset Generator---------\n')
    test_gen = Create_Dataset(batch_size,test_input_img_paths,stage='test',augment=False)
    
    counter = 0
    print('---------Testing---------\n')
    for batch in range(test_gen.__len__()):
        inputs = test_gen.__getitem__(batch)
        pred = model_.predict(inputs)
        counter = save_raw(pred,out_path,counter)
