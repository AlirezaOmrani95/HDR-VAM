import tensorflow as tf

def feature_extractor(input_img):
  features = tf.keras.layers.SeparableConv2D(32,3,padding='same')(input_img)
  features_max = tf.keras.layers.MaxPool2D(2)(features)
  features_avg = tf.keras.layers.AvgPool2D(2)(features)
  
  concat = tf.keras.layers.concatenate([features_max,features_avg])
  features = tf.keras.layers.SeparableConv2D(32,3,activation = 'relu',padding='same')(concat)
  features = tf.keras.layers.UpSampling2D(2)(features)
  
  return features
  
def alingment(input_i,input_r):
  input_reference = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(input_r)
  input_reference1 = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(input_reference)
  input_reference2 = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(input_reference)
  
  multiplication = tf.keras.layers.multiply([input_i,input_reference1])
  out = tf.keras.layers.add([multiplication,input_reference2])
  
  return out
  
def attention(input_i,input_r):
  concat = tf.keras.layers.concatenate([input_i,input_r],axis=-1)
  x = tf.keras.layers.SeparableConv2D(64,3,padding='same',activation='relu')(concat)
  sigmoid = tf.keras.layers.SeparableConv2D(32,3,padding='same',activation='sigmoid')(x)
  return sigmoid

def alignment_attention(input_i,input_r_alingment,input_r_attention):
  
  input_i_alignment = feature_extractor(input_i[:,:,:,0:3])
  input_i_attention = feature_extractor(input_i[:,:,:,3:])
  
  align_i = alingment(input_i_alignment,input_r_alingment)
  attention_i = attention(input_i_attention,input_r_attention)
  

  return(align_i,attention_i)
  
def visual_attention(inps,masks):
  low_multiplication = tf.keras.layers.multiply([inps[0],masks[0]] ,name ='low_multiplication')
  high_multiplication = tf.keras.layers.multiply([inps[1],masks[1]] ,name ='high_multiplication')
  
  low_features = feature_extractor(low_multiplication)
  high_features = feature_extractor(high_multiplication)
  
  
  addition = tf.keras.layers.add([low_features,high_features])
  
  return addition
   
def reconstruction(inps,addition):
    reference = inps[:,:,:,-64:]
    x = tf.keras.layers.SeparableConv2D(64,(3,3), padding = 'same')(inps) #(256,256)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    avg_pool = tf.keras.layers.AveragePooling2D((2,2))(x) #(128,128)
    max_pool = tf.keras.layers.MaxPool2D((2,2))(x)        #(128,128)
    concat = tf.keras.layers.concatenate([avg_pool,max_pool])

    x = tf.keras.layers.SeparableConv2D(128,(3,3), padding = 'same')(concat)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    avg_pool = tf.keras.layers.AveragePooling2D((2,2))(x) #(64,64)
    max_pool = tf.keras.layers.MaxPool2D((2,2))(x)        #(64,64)
    concat = tf.keras.layers.concatenate([avg_pool,max_pool])
    

    x = tf.keras.layers.SeparableConv2D(256,(3,3), padding = 'same')(concat)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    avg_pool = tf.keras.layers.AveragePooling2D((2,2))(x) #(32,32)
    max_pool = tf.keras.layers.MaxPool2D((2,2))(x)        #(32,32)
    concat = tf.keras.layers.concatenate([avg_pool,max_pool])
    
    
    x = tf.keras.layers.SeparableConv2D(256,(3,3), padding = 'same')(concat)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    avg_pool = tf.keras.layers.AveragePooling2D((2,2))(x) #(16,16)
    max_pool = tf.keras.layers.MaxPool2D((2,2))(x)        #(16,16)
    concat = tf.keras.layers.concatenate([avg_pool,max_pool])
    
    
    # Making the ref image and add layer the same size of (16,16).
    add_avg_pool = tf.keras.layers.AveragePooling2D((16,16))(addition)
    
    med_avg_pool = tf.keras.layers.AveragePooling2D((16,16))(reference)
    
    cat = tf.keras.layers.concatenate([concat,add_avg_pool,med_avg_pool])
    x = tf.keras.layers.SeparableConv2D(256,(3,3), padding = 'same', activation = 'relu')(cat)
    x = tf.keras.layers.UpSampling2D((2,2))(x) #(32,32)
    
    # Making the ref image and add layer the same size of (32,32).
    add_avg_pool = tf.keras.layers.AveragePooling2D((8,8))(addition)
    
    med_avg_pool = tf.keras.layers.AveragePooling2D((8,8))(reference)
    
    cat = tf.keras.layers.concatenate([x,add_avg_pool,med_avg_pool])
    x = tf.keras.layers.SeparableConv2D(128,(3,3), padding = 'same', activation = 'relu')(cat)
    x = tf.keras.layers.UpSampling2D((2,2))(x) #(64,64)

    # Making the ref image and add layer the same size of (64,64).
    add_avg_pool = tf.keras.layers.AveragePooling2D((4,4))(addition)
    
    med_avg_pool = tf.keras.layers.AveragePooling2D((4,4))(reference)
    
    cat = tf.keras.layers.concatenate([x,add_avg_pool,med_avg_pool])
    x = tf.keras.layers.SeparableConv2D(64,(3,3), padding = 'same', activation = 'relu')(cat)
    x = tf.keras.layers.UpSampling2D((2,2))(x) #(128,128)

    # Making the ref image and add layer the same size of (128,128).
    add_avg_pool = tf.keras.layers.AveragePooling2D((2,2))(addition)
    
    med_avg_pool = tf.keras.layers.AveragePooling2D((2,2))(reference)
    
    cat = tf.keras.layers.concatenate([x,add_avg_pool,med_avg_pool])
    x = tf.keras.layers.SeparableConv2D(32,(3,3), padding = 'same', activation = 'relu')(cat)
    x = tf.keras.layers.UpSampling2D((2,2))(x) #(256,256)

    out = tf.keras.layers.SeparableConv2D(3,(1,1), padding='same',activation='relu')(x)
    return (out)

def refinement(input,reconstructed):
    input = tf.keras.layers.SeparableConv2D(3,3,padding='same',activation='relu')(input)
    
    concat = tf.keras.layers.concatenate([reconstructed,input])
    x = tf.keras.layers.SeparableConv2D(16,(3,3),padding='same')(concat)
    x = tf.keras.layers.SeparableConv2D(16,(3,3),padding='same',activation='relu')(x)

    concat = tf.keras.layers.concatenate([x,input])
    x = tf.keras.layers.SeparableConv2D(16,(3,3),padding='same')(concat)
    x = tf.keras.layers.SeparableConv2D(16,(3,3),padding='same',activation='relu')(x)

    concat = tf.keras.layers.concatenate([x,input])
    x = tf.keras.layers.SeparableConv2D(16,(3,3),padding='same')(concat)
    x = tf.keras.layers.SeparableConv2D(16,(3,3),padding='same',activation='relu')(x)

    final = tf.keras.layers.Conv2D(3,(1,1),padding='same',activation='sigmoid',name='Final_output')(x)

    return final

def visual_mask_expansion(mask):
    
    #This function expands the number of channels of the masks into 6.
    
    exp_mask = tf.keras.layers.concatenate([mask,mask],axis=-1)
    exp_mask = tf.keras.layers.concatenate([exp_mask,mask],axis=-1)
    return exp_mask

def model():
    low_input = tf.keras.layers.Input((None,None) + (6,),name='input_low')
    medium_input = tf.keras.layers.Input((None,None) + (6,),name='input_medium')
    high_input = tf.keras.layers.Input((None,None) + (6,),name='input_high')
    low_mask = tf.keras.layers.Input((None,None) + (2,),name='input_low_mask')
    high_mask = tf.keras.layers.Input((None,None) + (2,),name='input_high_mask')
    
    #expand visual segmentations to 6 channels
    exp_low_mask = visual_mask_expansion(low_mask)
    exp_high_mask = visual_mask_expansion(high_mask)
    
    # visual alignment
    add = visual_attention([low_input,high_input],[exp_low_mask,exp_high_mask])
    
    reference_features_alignment = feature_extractor(medium_input)
    reference_features_attention = feature_extractor(medium_input)
    reference_features = tf.keras.layers.concatenate([reference_features_alignment,reference_features_attention],axis=-1)
    
    #image alignment and attention
    align_low,attention_low = alignment_attention(low_input,reference_features_alignment,reference_features_attention)
    align_high, attention_high = alignment_attention(high_input,reference_features_alignment,reference_features_attention) 

    inps = tf.keras.layers.concatenate([align_low,attention_low,align_high,attention_high,add,reference_features],axis=-1)

    #reconstruction stage
    reconstructed = reconstruction(inps,add)

    #refinement stage
    final_out = refinement(reference_features,reconstructed)

    model = tf.keras.Model(inputs=[low_input,medium_input,high_input,low_mask,high_mask],outputs=final_out)
    return(model)
model()

def test_model():
    from model_profiler import model_profiler
    model_ = model()
    profile = model_profiler(model_,1)
    print(profile)
if __name__ == '__main__':
    test_model()