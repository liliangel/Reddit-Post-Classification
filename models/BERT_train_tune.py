# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 14:32:46 2021

@author: lilian

"""
import data.data_preproccessing as dat
import numpy as np
import tensorflow as tf

import tensorflow_text as text
from tensorflow.keras.utils import to_categorical



import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tf_hub_url as hub_values
from model_template import create_BERTmodel


def BERT_train(preprocess_url, encoder_url, X_train, X_test, Y_train, Y_test, trainable = True):
    
     
    model = create_BERTmodel(preprocess_url, encoder_url)
    model.summary()
    
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="BERT_model.h5",
                                                 save_weights_only=True,
                                                 verbose=1)
    callbacks = [cp_callback]

    history = model.fit(X_train, Y_train,
                          epochs=4,
                          validation_data = (X_test, Y_test),  
                          callbacks=callbacks)                                                 
                      
    

    return history                             

    
def plot_graphs(history, metric):
    
    plt.rcParams['figure.figsize'] = (10, 6)    
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.title(metric +" Curves for BERT Model")
    plt.savefig(metric + "_BERT_Models", bbox_inches='tight')   
    plt.show()
                         

def main(trainable=False):
    
        
    with open('../data/classes.txt','r') as f:
        dict_class = f.read()
    
       
    
    df_raw= pd.read_csv('../data/reddit_data.csv')
    df = dat.clean(df_raw, dict_class)
    
    X= df['text']
    Y = df['class']
    
    
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=42)
    
    #one hot encoding
    Y_train = to_categorical(Y_train, dtype ="uint8")
    Y_test= to_categorical(Y_test, dtype ="uint8")
    
    preprocess_url = hub_values.preprocess_url
    encoder_url = hub_values.encoder_url 
    

    
    history = BERT_train(preprocess_url, encoder_url, X_train, X_test, Y_train, Y_test, trainable = True)
    
    
    print('starting evaluation...')  
    plot_graphs(history,'accuracy')
    plot_graphs(history,'loss')  
    
    
if __name__== "__main__":
    
    main()    



        
    




