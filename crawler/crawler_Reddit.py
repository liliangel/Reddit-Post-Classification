# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:47:34 2021

@author: lilian
"""

import pandas as pd
import requests
import constants as c
import json



def auth_reddit_api(uri_token, passw, CLIENT_ID, SECRET, USER_AGENT):
    
    auth = auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET)
    login = json.loads(passw)
    head = {'User-Agent': USER_AGENT}

    res = requests.post(uri_token,
                        auth=auth, data=login, headers=head)

    TOKEN = res.json()['access_token']
    headers = {**head, **{'Authorization': f"bearer {TOKEN}"}}
    print(headers)


    #res = requests.get(uri_api,  headers=headers)

    return headers


def extracting_subreddit(res):
    # initialize temp dataframe for batch of data in response
    df = pd.DataFrame()

    # loop through each post pulled from res and append to df
    for post in res.json()['data']['children']:
        print('reading json')
        df = df.append({
            'subreddit': post['data']['subreddit'],
            'title': post['data']['title'],
            'text': post['data']['selftext'],
            }, ignore_index=True)

    return df

    
def pull_all_data(subreddits, params, uri_api, headers):
    
    data = pd.DataFrame()
  
    print(uri_api)

    for subreddit in subreddits:
        
        uri = uri_api + subreddit.strip()+'/hot'
        
        print(uri)
        res = requests.get(uri, headers=headers,  params=params)
      
        # get dataframe from response
        new_df = extracting_subreddit(res)
     
    
        # append new_df to data
        data = data.append(new_df, ignore_index=True)
        
    data['reddit_post']  =  data['title'] +' '+ data['text']
  
    return data
 
 
def main():    
        
    with open('password.txt','r') as f:
        passw = f.read()
        
       
        
    with open('subreddit.txt') as f:  
        subreddits = f.readlines()
        
    params = {'limit': 10}   
    
    uri_api = 'https://oauth.reddit.com/r/'
    
    uri_token = 'https://www.reddit.com/api/v1/access_token'
    
    headers = auth_reddit_api(uri_token, passw, c.CLIENT_ID, c.SECRET, c.USER_AGENT)
        
    data = pull_all_data(subreddits, params, uri_api, headers)    
    
    data.to_csv('../data/reddit_data_test.csv')
    
    
    
if __name__== "__main__":
    
    main()
 
    


