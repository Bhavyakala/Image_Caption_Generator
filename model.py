import numpy as np
import pandas as pd
from keras.layers import Input,Embedding,LSTM,Dropout,Dense,Activation,add
from keras.models import  Model,load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
from PIL import Image
from DataPreproccessor import Dp                
import os
import copy
from keras.callbacks.callbacks import ModelCheckpoint
from copy import deepcopy
import keras
import tensorflow as tf
import datetime
import json
class model_build():
    glove_file_address = ''
    embeddings_index = {}
    embedding_dim = 0
    embedding_matrix = 0
    dp = 0

    def __init__(self,address,glove_file_address,embedding_dim):
        super().__init__()
        self.glove_file_address = glove_file_address
        self.embedding_dim = embedding_dim
        self.dp = Dp(address)
        # d = self.dp.to_dictionary()
        df = self.dp.cleaning()
        vocab = self.dp.to_vocabulary()
        word_to_ix, ix_to_word = self.dp.word_index()
        m = self.dp.maxLength()
        features = self.dp.feature_extract('D:\Coding_wo_cp\Image_Caption_Generator\Flicker8k_Dataset')   
        photo,dataset,description_dataset = self.dp.load_dataset('D:\Coding_wo_cp\Image_Caption_Generator\Flickr_8k.trainImages.txt')

    def embedding_glove(self) :
        f = open(self.glove_file_address,encoding='utf-8')
        for line in f :
            word, coefs = line.split(maxsplit = 1)
            coefs = np.fromstring(coefs,'f',sep=' ')
            self.embeddings_index[word] = coefs
        f.close()
        self.embedding_matrix = np.zeros((self.dp.vocab_len,self.embedding_dim))
        for word , i in self.dp.word_to_ix.items() :
            embedding_vector  = self.embeddings_index.get(word)
            if embedding_vector is not None :
                self.embedding_matrix[i] = embedding_vector
        print('glove embeddings prepared...')        
        return self.embeddings_index, self.embedding_matrix,self.dp

    def data_generator(self,num_photos_per_batch) :
        X1, X2, y = list(), list(), list() 
        n = 0
        while 1:
            for key, desc_list in self.dp.description_dataset.items() :
                n+=1
                for d in desc_list :
                    seq = [self.dp.word_to_ix[word] for word in d.split(' ') if word in self.dp.word_to_ix]
                    for i in range(1,len(seq)) :
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences([in_seq],maxlen=self.dp.max_len)[0]
                        out_seq = to_categorical([out_seq],num_classes=self.dp.vocab_len)[0]
                        X1.append(self.dp.photo[key])
                        X2.append(in_seq)
                        y.append(out_seq)
                if n==num_photos_per_batch :
                    yield [[np.array(X1),np.array(X2)],np.array(y)]   
                    X1, X2, y = list(), list(), list()     
                    n=0

    def model_architecture(self) :
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        inputs2 = Input(shape=(self.dp.max_len,))
        se1 = Embedding(self.dp.vocab_len,
                        self.embedding_dim,
                        mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        decoder1 = add([fe2,se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.dp.vocab_len,activation='softmax')(decoder2)

        model = Model(inputs=[inputs1,inputs2], outputs= outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        model.layers[2].set_weights([self.embedding_matrix])
        # model.layers[2].trainable = False
        
        model.summary()
        return model  

    def greedy_search(self,model,photo_file):
        in_text = 'startseq'
        photo = self.dp.single_feature_extract(photo_file)
        # photo = np.reshape(photo,photo.shape[1])
        with open('word_to_ix.json', 'r') as f:
            self.dp.word_to_ix = json.load(f)
        with open('ix_to_word.json', 'r') as f:
            self.dp.ix_to_word = json.load(f)  

        self.dp.ix_to_word = { int(i) : w for i,w in self.dp.ix_to_word.items()}    
        self.dp.word_to_ix = { w : int(i) for w,i in self.dp.word_to_ix.items()}   

        print(photo.shape,type(photo))
        print(self.dp.word_to_ix['endseq'])
        for i in range(self.dp.max_len) :
            seq = [self.dp.word_to_ix[w] for w in in_text.split() if w in self.dp.word_to_ix] 
            seq = pad_sequences([seq],maxlen=self.dp.max_len)
            yhat = model.predict([photo, seq])
            yhat = np.argmax(yhat)
            word = self.dp.ix_to_word[yhat]
            in_text += ' ' + word
            if word=='endseq':
                break
        print(self.dp.max_len)    
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final                                                 

if __name__ == "__main__": 

    mb = model_build('D:\Coding_wo_cp\Image_Caption_Generator\Flickr8k.token.txt',
                    'D:\Coding_wo_cp\Image_Caption_Generator\glove.6B.200d.txt',
                    200)   
    embedding_index, embedding_matrix, dp = mb.embedding_glove()
    
    # dev set loading
    ob = deepcopy(dp)  
    photo,dataset,description_dataset = ob.load_dataset('D:\Coding_wo_cp\Image_Caption_Generator\Flickr_8k.devImages.txt')
    X1_dev,X2_dev,y_dev = ob.create_sequences()

    # creating directory for saving model and corresponding mapping of word to ix
    model_no = '8'
    directory = 'model-'+ model_no
    os.mkdir(directory) 

    epochs = 7
    num_photos_per_batch = 3
    steps = len(dp.description_dataset)//num_photos_per_batch
    filepath = directory + '\model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')                
    
    model = mb.model_architecture()

    for i in range(epochs) :
        print('epoch : {}/{}'.format(i+1,epochs))
        generator = mb.data_generator(num_photos_per_batch)
        model.fit_generator(generator,
                            epochs=1,
                            steps_per_epoch=steps,
                            callbacks=[checkpoint],
                            validation_data=([X1_dev,X2_dev],y_dev))                   
    model.save(directory + '\m' + model_no + '.h5')  

    loaded_model = load_model('D:\Coding_wo_cp\Image_Caption_Generator\model-8\model-ep001-loss3.420-val_loss3.866.h5')
    description = mb.greedy_search(loaded_model,'D:/Coding_wo_cp/Image_Caption_Generator/Flicker8k_Dataset/2748729903_3c7c920c4d.jpg')
    im = plt.imread('D:/Coding_wo_cp/Image_Caption_Generator/Flicker8k_Dataset/2748729903_3c7c920c4d.jpg')
    plt.imshow(im)
    plt.xlabel(description)

    # TODO : save mapping of words along with models                       
