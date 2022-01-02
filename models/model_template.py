# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 02:32:08 2021

@author: lilian
"""


import tensorflow as tf
import tensorflow_hub as hub

 
   
    
    
def create_BERTmodel( preprocess_url, encoder_url):
        


        bert_preprocess_model = hub.KerasLayer(preprocess_url)
        bert_encoder_model = hub.KerasLayer(encoder_url,  trainable=True)
        text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = 'Inputs')
        preprocessed_text = bert_preprocess_model(text_input)
        embeed =bert_encoder_model(preprocessed_text)
        dropout = tf.keras.layers.Dropout(0.1, name = 'Dropout')(embeed['pooled_output'])
        outputs = tf.keras.layers.Dense(10, activation = 'softmax', name = 'Dense')(dropout)

        model = tf.keras.Model(inputs = [text_input], outputs = [outputs])
    
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [tf.keras.metrics.CategoricalAccuracy('accuracy')]
        

        model.compile(optimizer, loss, metrics)
        

        return model
    
def create_Context_model( model_url, embed_size, trainable = False):
        
        hub_layer = hub.KerasLayer(model_url, input_shape=[], output_shape =  [embed_size], dtype = tf.string, trainable = trainable)
        model = tf.keras.models.Sequential([
                                        hub_layer,
                                        tf.keras.layers.Dense(256, activation ='relu'),
                                        tf.keras.layers.Dense(64, activation ='relu'),
                                        tf.keras.layers.Dense(10, activation ='softmax')                                        
                                        ])    
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.0, axis=-1,
                                                       name='categorical_crossentropy')
        metrics = [tf.keras.metrics.CategoricalAccuracy('accuracy')]
        
        model.compile(optimizer, loss, metrics)
       
        
        return model
    
   
         
               