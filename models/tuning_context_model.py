# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 23:45:45 2021

@author: lilian
"""

import train_context_model as contxt
import tensorflow_text as text

def main(models_uri, BATCH_SIZE, EPOCHS):

    contxt.main(models_uri, BATCH_SIZE, EPOCHS, trainable=True)
    
    
if __name__== "__main__":
    
    BATCH_SIZE= 16
    EPOCHS =40
    
    models_uri =[ "https://tfhub.dev/google/universal-sentence-encoder/4"]
    main(models_uri, BATCH_SIZE, EPOCHS)      
    
    