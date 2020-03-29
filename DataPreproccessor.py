import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import csv,string
class DataPreproccesor:
    address = ''
    desc = ''
    word_count = {}
    vocab = set()
    word_to_ix = {}
    ix_to_word = {}
    max_len = 0
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
        return mapping    
    def cleaning(self):
        df = pd.read_csv(self.desc,header=0)
        # print(df['desc'])
        df['desc'] = df['desc'].str.lower()
        df['desc'] = df['desc'].str.translate(str.maketrans('', '', string.punctuation))
        df['desc'] = df['desc'].str.replace(r'\b\w\b',' ')
        d = []
        for i in df.index:
            line = df['desc'][i]
            for word in line.split():
                if word.isalpha():
                    d.append(word)
            if(len(d)>1) :
                df['desc'][i] = 'startseq ' + " ".join(d) + ' endseq' 
            else:
                df.drop(df.index[i])    
            d.clear()                   
        df.to_csv('desc.csv')
        return df                
    def to_vocabulary(self) :
        df = pd.read_csv(self.desc,header = 0)
        l = []
        for i in df.index :
            line = df['desc'][i] 
            [self.vocab.update(line.split())]
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
if __name__ == "__main__":
    ob = DataPreproccesor("D:\Coding_wo_cp\Image_Caption_Generator\Flickr8k.token.txt")
    d = ob.to_dictionary()
    df = ob.cleaning()
    vocab = ob.to_vocabulary()
    word_to_ix, ix_to_word = ob.word_index()
    m = ob.maxLength()
