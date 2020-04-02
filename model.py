import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input,Embedding,LSTM,Dropout,Dense,Activation,add
from keras.models import  Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
from DataPreproccessor import Dp 
import os
from keras.callbacks.callbacks import ModelCheckpoint
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
        d = self.dp.to_dictionary()
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
                        out_seq = to_categorical([out_seq],num_classes=len(self.dp.vocab)+1)[0]
                        X1.append(self.dp.photo[key])
                        X2.append(in_seq)
                        y.append(out_seq)
                if n==num_photos_per_batch :
                    X1 = np.squeeze(np.array(X1))  
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

        model.summary()
        return model                                        

if __name__ == "__main__":
    mb = model_build('D:\Coding_wo_cp\Image_Caption_Generator\Flickr8k.token.txt',
                    'D:\Coding_wo_cp\Image_Caption_Generator\glove.6B.200d.txt',
                    200)   
    epochs = 20
    num_photos_per_batch = 6
    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')                
    model = mb.model_architecture()
    _, embedding_matrix,dp = mb.embedding_glove()
    x = mb.data_generator(num_photos_per_batch)

    steps = len(dp.description_dataset)//num_photos_per_batch
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False
    for i in range(epochs) :
        generator = mb.data_generator(num_photos_per_batch)
        model.fit_generator(generator,epochs=1,steps_per_epoch=steps, callbacks=[checkpoint])



    