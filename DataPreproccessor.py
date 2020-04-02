import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import csv,string
import json
from keras.applications import  VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from PIL import Image
from os import listdir
class Dp:
    address = ''
    desc = ''
    word_count = {}
    vocab = set()
    word_to_ix = {}
    ix_to_word = {}
    max_len = 0
    vocab_len = 0 
    features = {}
    df = pd.DataFrame()
    photo = {}
    dataset = []
    description_dataset = {}
    X1,X2,y = list(), list(), list()
    def __init__(self,address):
        super().__init__()
        self.address = address

    def to_dictionary(self) :
        mapping = dict()
        fr = open(self.address,"r")
        body = fr.read()
        l = list()
        for line in body.split("\n") :
            tokens = line.split()
            if(len(line)<2) :
                continue
            image_id, image_desc = tokens[0], tokens[1:]
            image_id = image_id.split('.')[0]
            image_desc = ' '.join(image_desc)
            if image_id not in mapping :
                mapping[image_id] = list(list())
            mapping[image_id].append(image_desc)    
        self.desc='desc.csv'
        fw = open(self.desc,"w")
        writer = csv.writer(fw)
        writer.writerow(['image_id','desc'])
        for i,ds in mapping.items():
            for d in ds :
                writer.writerow([i,d])
        fr.close()
        fw.close()        
        return mapping   

    def cleaning(self):
        self.df = pd.read_csv(self.desc,header=0)
        # print(df['desc'])
        self.df['desc'] = self.df['desc'].str.lower()
        self.df['desc'] = self.df['desc'].str.translate(str.maketrans('', '', string.punctuation))
        self.df['desc'] = self.df['desc'].str.replace(r'\b\w\b',' ')
        d = []
        for i in self.df.index:
            line = self.df['desc'][i]
            for word in line.split():
                if word.isalpha():
                    d.append(word)
            if(len(d)>1) :
                self.df['desc'][i] = 'startseq ' + " ".join(d) + ' endseq' 
            else:
                self.df.drop(self.df.index[i])    
            d.clear()                   
        self.df.to_csv('desc.csv')
        return self.df          

    def to_vocabulary(self) :
        df = pd.read_csv(self.desc,header = 0)
        l = []
        for i in df.index :
            line = df['desc'][i] 
            [self.vocab.update(line.split())]
        self.vocab_len = len(self.vocab)    
        return self.vocab            

    def word_index(self) :
        i = 1
        for w in self.vocab :
            self.word_to_ix[w] = i
            self.ix_to_word[i] = w
            i+=1
        return self.word_to_ix,self.ix_to_word  

    def maxLength(self) :
        df = pd.read_csv(self.desc,header = 0)
        l = len(df['desc'][0].split())
        for i in df.index :
            line = df['desc'][i]
            l = len(line.split())
            if self.max_len < l :
                self.max_len = l
        return self.max_len  

    def feature_extract(self,directory) :
        # model = VGG19()
        # model.layers.pop()
        # model = Model(input=model.inputs, outputs=model.layers[-1].output)
        # print(model.summary())
        # for name in listdir(directory) :
        #     filename = directory + '/' + name
        #     image = load_img(filename, target_size=(224,224))
        #     image = img_to_array(image)
        #     image = np.expand_dims(image,axis=0)
        #     image = preprocess_input(image)
        #     feature = model.predict(image)
        #     image_id = name.split('.')[0]
        #     self.features[image_id] = feature
        # for i in self.features.keys() :
        #     self.features[i] = self.features[i].tolist()
        # with open('features.json', 'w') as f:
        #     json.dump(self.features,f)  
        with open('features.json', 'r') as f:
            self.features = json.load(f)
        for i in self.features.keys() :
            self.features[i] = np.array(self.features[i])           
        return self.features  

    def load_dataset(self,filename) :
        f = open(filename,'r')
        body = f.read()
        for line in body.split('\n') :
            image_id = line.split('.')[0]
            if len(line) < 1:
                continue
            self.dataset.append(image_id)
            self.photo[image_id] = self.features[image_id]
        f.close()
        for i in self.df.index:
            image_id = df['image_id'][i]
            if image_id in self.dataset:
                if image_id not in self.description_dataset :
                    self.description_dataset[image_id] = list()
                self.description_dataset[image_id].append(df['desc'][i])         
        return self.photo,self.dataset,self.description_dataset

    def create_sequences(self) :    
        for key, desc_list in self.description_dataset.items() :
            for d in desc_list :
                seq = [self.word_to_ix[word] for word in d.split(' ') if word in self.word_to_ix]
                for i in range(1,len(seq)) :
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq],maxlen=self.max_len)[0]
                    out_seq = to_categorical([out_seq],num_classes=len(self.vocab)+1)[0]
                    self.X1.append(self.photo[key])
                    self.X2.append(in_seq)
                    self.y.append(out_seq)
        self.X1 = np.array(self.X1)
        self.X2 = np.array(self.X2)
        self.y = np.array(self.y)
        return self.X1,self.X2,self.y

if __name__ == "__main__":
    ob  = Dp("D:\Coding_wo_cp\Image_Caption_Generator\Flickr8k.token.txt")
    d = ob.to_dictionary()
    df = ob.cleaning()
    vocab = ob.to_vocabulary()
    word_to_ix, ix_to_word = ob.word_index()
    m = ob.maxLength()
    features = ob.feature_extract('D:\Coding_wo_cp\Image_Caption_Generator\Flicker8k_Dataset')   
    photo,dataset,description_dataset = ob.load_dataset('D:\Coding_wo_cp\Image_Caption_Generator\Flickr_8k.trainImages.txt')
    X1,X2,y = ob.create_sequences()