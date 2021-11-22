# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 22:40:01 2021

@author: m_nay
"""

import bz2
import pickle
import _pickle as cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st 
import pandas as pd
from nltk import word_tokenize
from nltk.stem.isri import ISRIStemmer
import re
import nltk
from stop_words import get_stop_words
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
import spacy_streamlit

import spacy
from spacy_streamlit import visualize_ner

nlp = spacy.load("en_core_web_sm")



#!pip install stop_words
nltk.download('punkt')

# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
 with bz2.BZ2File(title + '.pbz2', 'w') as f: 
  cPickle.dump(data, f)

# Load any compressed pickle file
def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data

def getsubstr(s,start,end): 
  return s[s.find(start)+len(start):s.rfind(end)]

from stop_words import get_stop_words
stop_words = get_stop_words('ar')

def remove_stopWords(s):
    '''For removing stop words
    '''
    s = ' '.join(word for word in s.split() if word not in stop_words)
    return s

sts = ISRIStemmer()
word_list = "عرض يستخدم الى التفاعل مع المستخدمين في هاذا المجال !وعلمآ تكون الخدمه للستطلاع على الخدمات والعروض المقدمة"

# Define a function
def filter(word_list):
    wordsfilter=[]
    for a in word_tokenize(word_list):
        stem = sts.stem(a)
        wordsfilter.append(stem)
    #print(wordsfilter)
    return ' '.join(wordsfilter)


df = pd.read_table(r'./123.txt')

modelname = 'modelPrimary.pickle'
vectorizername = 'vecPrimary.pickle'
clf = pickle.load(open(modelname, 'rb'))
vectorizer = pickle.load(open(vectorizername, 'rb'))

clfSecondary = decompress_pickle('modelSecondary.pbz2') 
vectorizerSecondary = pickle.load(open('vecSecondary.pickle', 'rb'))


st.markdown("<h2 style='text-align: center;'>مصنف مركز البحوث والدراسات الآلي</h2>", unsafe_allow_html=True)
  

st.markdown("<h4 style='text-align: center; color: orange;'>الآن يمكنك تصنيف بياناتك طبقا للتصنيف المعتمد لدينا</h4>", unsafe_allow_html=True)


with st.form(key='mlform'):

    st.markdown("<h6 style='text-align: center;'>ادخل النص المراد تصنيفه</h6>", unsafe_allow_html=True)

    message = st.text_area("")
    submit_message = st.form_submit_button(label='صنف')
    
if submit_message:
 
   
   
    query = " ".join(re.findall('[\w]+',message))
    query = remove_stopWords(query)
    query = filter(query)
    
    predictions =clf.predict_proba(vectorizer .transform([query]))
    preds_idx = np.argsort(-predictions) 

    classes = pd.DataFrame(clf.classes_, columns=['class_name'])

    sum = 0
    nums = 0
    for i in range(10):
      if predictions[0][preds_idx[0][i]] < 0.1:
        break;
      else:
        nums = nums +1
        sum = sum + predictions[0][preds_idx[0][i]]
        #print(classes.iloc[preds_idx[0][i]])
        #print(predictions[0][preds_idx[0][i]])

    result = pd.DataFrame(columns=['predicted_class','predicted_prob'])

    #st.markdown("<h4 style='text-align: center; color: black;'>احتمالات التصنيف طبقا للمجالات المعرفية الرئيسية</h4>", unsafe_allow_html=True)

    ys=[]
    for i in range(nums):
      #print(classes.iloc[preds_idx[0][i]])
      #print((predictions[0][preds_idx[0][i]]/sum)*100)
      s = getsubstr(str(classes.iloc[preds_idx[0][i]]),'class_name ','\n')
      ys.append(round((predictions[0][preds_idx[0][i]]/sum)*100,2))
      dict = {'predicted_class': s, 'predicted_prob': ((predictions[0][preds_idx[0][i]]/sum)*100)}
      result = result.append(dict, ignore_index = True)

      #######st.markdown("<h4 style='text-align: center;color:black'>"+  s + " ("+ str(round((predictions[0][preds_idx[0][i]]/sum)*100,2)) +"%)" +"</h4>", unsafe_allow_html=True)
      
        #pred = clf.predict(vectorizer.transform([message]))[0]
        #dd = df.loc[df['labelSecondary'] == pred]
        #dd = dd.iloc[[0]]
        #print(dd['label'])
        #st.title(pred)

        #st.markdown("<h3 style='text-align: center;color:red'>"+  pred +"</h3>", unsafe_allow_html=True)
        
    fig = px.bar(result,
                x='predicted_class',
                y='predicted_prob',
                title='احتمالات التصنيف طبقا للمجالات المعرفية الرئيسية',
                hover_name='predicted_class', color='predicted_class',
                 labels={
                     "predicted_class": "المجال المعرفي الرئيسي المحتمل",
                     "predicted_prob": "الاحتمالية",
                     "predicted_class": "المجالات المعرفية الرئيسية المحتملة"
                 })

    for i in range(nums):
     fig.data[i].text = ys[i]
    
    
    fig.update_traces(textposition='inside')
    fig.update_layout(title_x=0.5)

    st.plotly_chart(fig)
################################################
    predictions =clfSecondary.predict_proba(vectorizerSecondary .transform([query]))
    preds_idx = np.argsort(-predictions) 

    classes = pd.DataFrame(clfSecondary.classes_, columns=['class_name'])

    sum = 0
    nums = 0
    for i in range(10):
      if predictions[0][preds_idx[0][i]] < 0.1:
        break;
      else:
        nums = nums +1
        sum = sum + predictions[0][preds_idx[0][i]]
        #print(classes.iloc[preds_idx[0][i]])
        #print(predictions[0][preds_idx[0][i]])

    result = pd.DataFrame(columns=['predicted_class','predicted_prob'])

    #st.markdown("<h4 style='text-align: center; color: orange;'>---------------------------------------</h4>", unsafe_allow_html=True)

    #st.markdown("<h4 style='text-align: center; color: black;'>احتمالات التصنيف طبقا للمجالات المعرفية الفرعية</h4>", unsafe_allow_html=True)

    ys = []
    for i in range(nums):
      #print(classes.iloc[preds_idx[0][i]])
      #print((predictions[0][preds_idx[0][i]]/sum)*100)
      s = getsubstr(str(classes.iloc[preds_idx[0][i]]),'class_name ','\n')
      ys.append(round((predictions[0][preds_idx[0][i]]/sum)*100,2))
      dict = {'predicted_class': s, 'predicted_prob': (predictions[0][preds_idx[0][i]]/sum)*100}
      result = result.append(dict, ignore_index = True)
 
    fig = px.bar(result,
              x='predicted_class',
              y='predicted_prob',
              title='احتمالات التصنيف طبقا للمجالات المعرفية الفرعية',
              hover_name='predicted_class', color='predicted_class',
               labels={
                   "predicted_class": "المجال المعرفي الفرعي المحتمل",
                   "predicted_prob": "الاحتمالية",
                   "predicted_class": "المجالات المعرفية الفرعية المحتملة"
               })

    for i in range(nums):
     fig.data[i].text = ys[i]


    fig.update_traces(textposition='inside')
    fig.update_layout(title_x=0.5)

    st.plotly_chart(fig)
    
    doc = nlp(message)
    visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
    #visualize_ent(doc, labels=nlp.get_pipe("ent").labels)
    models = ["en_core_web_sm"]
    #spacy_streamlit.visualize(models, doc)
    
      #st.markdown("<h4 style='text-align: center;color:black'>"+  s + " ("+ str(round((predictions[0][preds_idx[0][i]]/sum)*100,2)) +"%)" +"</h4>", unsafe_allow_html=True)
      
        #pred = clf.predict(vectorizer.transform([message]))[0]
        #dd = df.loc[df['labelSecondary'] == pred]
        #dd = dd.iloc[[0]]
        #print(dd['label'])
        #st.title(pred)
        
    
