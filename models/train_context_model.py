# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 15:28:32 2021

@author: lilian
"""
import data.data_preproccessing as dat
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import tf_hub_url as hub_values

from tensorflow.keras.utils import to_categorical
from model_template import create_Context_model


import matplotlib.pyplot as plt
plt.rcParams ['figure.figsize'] = (12, 8)



def contextualised_vector_model(model_url, BATCH_SIZE, EPOCHS, X_train, X_test, Y_train, Y_test, embed_size, name, trainable = False):
    
 
    model = create_Context_model(model_url, embed_size, trainable = trainable)
    model.summary()
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath= name +"_model.h5",
                                                 save_weights_only=True,
                                                 verbose=1)
    history = model.fit(X_train, Y_train,
                        epochs= EPOCHS,
                        batch_size= BATCH_SIZE,
                        validation_data = (X_test, Y_test),
                        callbacks=[tfdocs.modeling.EpochDots(), cp_callback,
                                  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')],
                                
                        verbose =0)                                 
                                  

    return history                              
                  

def train_vector_models(models_uri, emb_size, BATCH_SIZE, EPOCHS, X_train, X_test, Y_train, Y_test, trainable = False):
    
    histories = {}
    
    for model_url, n in zip(models_uri, emb_size):
        name = model_url.split('/')[-2]
        print('calculating...', name)        
        histories[name] = contextualised_vector_model(model_url, BATCH_SIZE, EPOCHS, X_train, X_test, Y_train, Y_test, embed_size=n, name=name, trainable = trainable)
    print('finishing train models')    
    return histories

def plot_acc(histories, trainable= False):
    
    print('starting Plot Accuracy')
    plt.rcParams['figure.figsize'] = (12, 8)
    plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy')
    plotter.plot(histories)
    plt.xlabel("Epochs")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    if trainable:
        plt.title("accuracy Curves for nnlm  Models Tuned")
        plt.savefig("accurancy_Context_nnlm_Models_Tuned", bbox_inches='tight')
    else :
        plt.title("Accuracy Curves for Models")
        plt.savefig("Accurancy_Context_Models", bbox_inches='tight')   
    plt.show()
    
def plot_loss(histories,  trainable=False): 
    print('starting Plot Loss')
     
    plotter = tfdocs.plots.HistoryPlotter(metric = 'loss')
    plotter.plot(histories)
    plt.xlabel("Epochs")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    if trainable:
        plt.title("Loss Curves for  nnlm Models Tuned")
        plt.savefig("Loss_Context_nnlm_Models_Tuned", bbox_inches='tight')
        
    else:    
        plt.title("Loss Curves for Models")
        plt.savefig("Loss_Context_Models", bbox_inches='tight')
    plt.show()
    
   
def main(models_uri, BATCH_SIZE, EPOCHS, trainable=False ):
    
    emb_size = [ hub_values.emb_size1, hub_values.emb_size2, hub_values.emb_size3]  
    
    with open('../data/classes.txt','r') as f:
        dict_class = f.read()
    '''
    I will Train and evaluate 3  model from tf -Hub. 
    Context-based representations may use language models to generate vectors of sentences. 
    So, instead of learning vectors for individual words in the sentence, 
    they compute a vector for sentences on the whole, 
    by taking into account the order of words and the set of co-occurring words.
    '''
        
        
    df_raw= pd.read_csv('../data/reddit_data.csv')
    df = dat.clean(df_raw, dict_class)
    
    X= df['text']
    Y = df['class']
    
    
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=42)
    
    #one hot encoding
    Y_train = to_categorical(Y_train, dtype ="uint8")
    Y_test= to_categorical(Y_test, dtype ="uint8")
    
    histories = {}
    
    print('starting train...')  
    histories = train_vector_models(models_uri, emb_size, BATCH_SIZE, EPOCHS, X_train, X_test, Y_train, Y_test, trainable=trainable)
    
    print('starting evaluation...')  
    plot_acc(histories)
    plot_loss(histories)   
    
    
if __name__== "__main__":
    
    models_uri =[hub_values.model_uri1, hub_values.model_uri2, hub_values.model_uri2]
    BATCH_SIZE = 16
    EPOCHS = 20 
    main(models_uri, BATCH_SIZE, EPOCHS)    
    

   
 
    







