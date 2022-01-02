# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 21:41:10 2021

@author: lilian
"""
from model_template import create_Context_model
from model_template import create_BERTmodel
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tf_hub_url as hub_values
from tensorflow.keras.utils import to_categorical
import data.data_preproccessing as dat
import tensorflow_text as text

def evaluate_BER(preprocess_url, encoder_url, model_file, epochs, batch_size, X_test, Y_test, class_list):
    
    #evauating BERT model   
    
    model = create_BERTmodel(preprocess_url, encoder_url)
    name = model_file.split('/')[-1].split('.')[-2]
    model.load_weights(model_file)
    print(name)
    loss, acc = model.evaluate(X_test, Y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    
    #storage values in a table
    
    df_pred = pd.DataFrame()
    
    df2= pd.DataFrame([["BERT", 4, 16, acc, loss]], columns=["Model", "Epochs", "Batch Size", "Accuracy", 'loss' ])
    
    
    #Convert model to json 
    
    model_json = model.to_json()
    with open (name+'.json','w') as json_file:
      json_file.write(model_json)
      
    # Predicting values  
      
    pred = model.predict(X_test)
    
    p_clss = [class_list[np.argmax(x)] for x in pred]
    
    df_pred["text"] = X_test
    df_pred["prediction"]=p_clss
    df_pred.to_csv(name+'_predictions.csv')
    

    pred_test = np.zeros_like(pred)
    pred_test[np.arange(len(pred)), pred.argmax(1)] = 1

     #Calculatiing confussion Matrix
    matrix =confusion_matrix(Y_test.argmax(axis=1), pred_test.argmax(axis=1))
    plot_confusion_matrix(matrix, class_list, name)
      
    return df2

def evaluate_ContextModel(model_url, embed_size, model_file, epochs, batch_size, X_test, Y_test, class_list):
    
    #evauating Context nnlm model
    
    model = create_Context_model(model_url, embed_size)  
    name = model_url.split('/')[-2]
    print(name)
    model.load_weights(model_file)
    
    loss, acc = model.evaluate(X_test, Y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    
    #storage values in a table
    
    print(name)
    
    df_pred = pd.DataFrame()
    
    df2= pd.DataFrame([[name, epochs, batch_size, acc, loss]], columns=["Model", "Epochs", "Batch Size", "Accuracy", 'loss' ])
    
    
    #Convert model to json 
    
    model_json = model.to_json()
    
    with open (name+'.json','w') as json_file:
      json_file.write(model_json)
      
    # Predicting values and store csv
    
    pred = model.predict(X_test)
    
    p_clss = [class_list[np.argmax(x)] for x in pred]
    
    df_pred["text"] = X_test
    df_pred["prediction"]=p_clss
    df_pred.to_csv(name+'_predictions.csv')
    
    pred_test = np.zeros_like(pred)
    pred_test[np.arange(len(pred)), pred.argmax(1)] = 1

    #Calculatiing confussion Matrix
    matrix =confusion_matrix(Y_test.argmax(axis=1), pred_test.argmax(axis=1))
    
    plot_confusion_matrix(matrix, class_list, name)
    
    return df2
    
    
def plot_confusion_matrix(cm, class_list, name):
    
    df_cm = pd.DataFrame(cm, class_list, class_list)
    sns.heatmap(df_cm, annot = True, annot_kws={"size":16})
    
    plt.title("Confusion Matrix "+ name)
    plt.savefig("plots/Confusion Matrix "+ name, bbox_inches='tight')
    plt.show() 
    
def plot_table(df):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
   

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    fig.tight_layout()
    plt.title("Table Model Evaluation")
    plt.savefig("plots/Table Model Evaluation", bbox_inches='tight')

    plt.show()

    
def main():
    
    '''Evalute model with test data set of new reddit posts crawled with https://oauth.reddit.com/r/xxxx/new"
    10 post of every 10 subreddits'''
    
    with open('../data/classes.txt','r') as f:
        dict_class = f.read()   
       
    df_raw= pd.read_csv('../data/reddit_data_test.csv')
    df_test = dat.clean(df_raw, dict_class)
        
    X_test= df_test['text']
    Y_test = df_test['class']        
    Y_test = to_categorical(Y_test, dtype ="uint8")    
    class_list=["books","chemistry","CryptoTechnology", "Dogtraining", "EatCheapAndHealthy", "family", 
                "HomeImprovement", "languages", "musictheory", "MachineLearning"]

    model_file = "models_trained/BERT_model.h5"
    epochs = 4    
    batch_size =16        
            
    df=pd.DataFrame( columns=["Model", "Epochs", "Batch Size", "Accuracy", 'loss' ])    
      
    df=df.append(evaluate_BER( hub_values.preprocess_url, hub_values.encoder_url, model_file, epochs, batch_size, X_test, Y_test, class_list))
   
    model_file = "models_trained/nnlm-en-dim50_model.h5"
    model_file2 ="models_trained/nnlm-en-dim128_model.h5"
    model_file3 ="models_trained/universal-sentence-encoder_model.h5"
    epochs = 100    
    batch_size =16  
    df=df.append(evaluate_ContextModel(hub_values.model_uri1, hub_values.emb_size1, model_file, epochs, batch_size, X_test, Y_test, class_list))
    df=df.append(evaluate_ContextModel(hub_values.model_uri2, hub_values.emb_size2, model_file2, epochs, batch_size, X_test, Y_test, class_list))
    epochs = 40
    df=df.append(evaluate_ContextModel(hub_values.model_uri3, hub_values.emb_size3, model_file3, epochs, batch_size, X_test, Y_test, class_list))
  
    plot_table(df)

if __name__== "__main__":
    
   
    main()
    
    
    
