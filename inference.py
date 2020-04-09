import numpy as np
import json
# from keras.applications.inception_v3 import InceptionV3,preprocess_input
# from keras.applications.vgg16 import VGG16,preprocess_input
# from keras.applications.vgg19 import VGG19,preprocess_input
# from keras.applications.xception import Xception,preprocess_input
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.models import Model, load_model
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

def single_feature_extract(filename) :
        # model = 
        model = ResNet50(weights='imagenet')
        model.layers.pop()
        model = Model(input=model.inputs, outputs=model.layers[-1].output)
        image = load_img(filename,target_size=(model.input_shape[1],model.input_shape[2]))
        image = img_to_array(image)
        image = np.expand_dims(image,axis=0)
        image = preprocess_input(image)
        feature = model.predict(image)
        return feature

def greedy_search(model,photo_file,max_len):
        in_text = 'startseq'
        photo = single_feature_extract(photo_file)
        # photo = np.reshape(photo,photo.shape[1])
        with open('word_to_ix.json', 'r') as f:
            word_to_ix = json.load(f)
        with open('ix_to_word.json', 'r') as f:
            ix_to_word = json.load(f)  

        ix_to_word = { int(i) : w for i,w in ix_to_word.items()}    
        word_to_ix = { w : int(i) for w,i in word_to_ix.items()}   

        print(photo.shape,type(photo))
        print(word_to_ix['endseq'])
        for i in range(max_len) :
            seq = [word_to_ix[w] for w in in_text.split() if w in word_to_ix] 
            seq = pad_sequences([seq],maxlen=max_len)
            yhat = model.predict([photo, seq])
            yhat = np.argmax(yhat)
            word = ix_to_word[yhat]
            in_text += ' ' + word
            if word=='endseq':
                break
        print(max_len)    
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final

if __name__ == "__main__":
    loaded_model = load_model('D:\Coding_wo_cp\Image_Caption_Generator\model-10_resnet\model-ep001-loss3.341-val_loss3.789.h5')
    description = greedy_search(loaded_model,'D:/Coding_wo_cp/Image_Caption_Generator/Flicker8k_Dataset/494792770_2c5f767ac0.jpg',34)
    im = plt.imread('D:/Coding_wo_cp/Image_Caption_Generator/Flicker8k_Dataset/494792770_2c5f767ac0.jpg')
    plt.imshow(im) 
    plt.xlabel(description)       