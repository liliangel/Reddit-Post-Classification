# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 09:43:59 2021

@author: lilian
"""
from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow_hub as hub

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = InteractiveSession(config=config)


class RedditClassificationModel(object):
    
    subreddit=["books","chemistry","CryptoTechnology", "Dogtraining", "EatCheapAndHealthy", "family", 
                "HomeImprovement", "languages", "musictheory", "MachineLearning"]
    
    def __init__(self, model_json_file, model_weights_file):
        with open (model_json_file, 'r') as json_file:
                   loaded_model_json = json_file.read()
                   self.loaded_model = model_from_json(loaded_model_json,custom_objects={'KerasLayer':hub.KerasLayer})
                   #json_file.close()
                   
        self.loaded_model.load_weights(model_weights_file) 
        self.loaded_model.make_predict_function()
        
         
        
    def predict_emotion(self, text):
        
        self.preds = self.loaded_model.predict(text)
        return RedditClassificationModel.subreddit[np.argmax(self.preds)]
        
    
    