# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 05:38:42 2021

@author: lilian
"""
 
import pandas as pd   
import re
import json

 

def clean(df, dict_class):
    
    dict_class=json.loads(dict_class)
    
    rows_text = []
    rows_subred =[]
    data = pd.DataFrame()
    
    # cleaning the text in every row
    
    for text, subred in zip(df['reddit_post'], df['subreddit']):
        
            #decontract
            text = re.sub(r"n\'t", " not", text)
            text = re.sub(r"\'re", " are", text)
            text = re.sub(r"\'s", " is", text)
            text = re.sub(r"\'d", " would", text)
            text = re.sub(r"\'ll", " will", text)
            text = re.sub(r"\'t", " not", text)
            text = re.sub(r"\'ve", " have", text)
            text = re.sub(r"\'m", " am", text)
            

            # Remove Punctuation and links
            text = re.sub(r'[?|!|\'|"|#]',r'',text)
            text = re.sub(r'[.|,|)|(|\|/]',r' ',text)
            text = re.sub(r'\*', r'', text)
            text = re.sub(r'\"?\\?&?gt;?', r'', text)
            text = re.sub('&amp;#x200B;/r/+', r'', text)
            text = re.sub(r'\n+', r' ', text)
            text = re.sub(r'\[.*?\]\(.*?\)', r'', text)
            text = re.sub(r'\s\s+', r' ', text)
            text = re.sub('&lt;', r'', text)
            text = re.sub(r'!(.*?)!', r'\1', text)  
            text = re.sub('`', r'', text)
            text = text.strip()
            text = text.replace("\n"," ")                 

            # Table
            text = re.sub(r'\|', ' ', text)
            text = re.sub(':-', '', text)

            # Heading
            text = re.sub('#', '', text)
            
                       
            
            rows_text.append(text)
            rows_subred.append(subred)
     
    #Creating the data frame suitable to our models        
    data['class']  =  rows_subred
    data['text'] = rows_text
    
    #Replacing numerical classes
    data['class']=data['class'].map(dict_class)
    
    #drooping the duplicated rows
    data = data.drop_duplicates(subset = ['text'])
    
    return data


            

