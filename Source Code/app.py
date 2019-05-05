from __future__ import division
from flask import Flask, jsonify, request
import numpy as np
#from sklearn.externals import joblib
import pandas as pd
import numpy as np


import pickle
import csv

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import warnings
warnings.filterwarnings("ignore")



# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################
'''def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])


def clean_text(sentence):
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    # https://gist.github.com/sebleier/554280
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
    return sentence.strip()
###################################################


'''
@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    load_model = pickle.load(open('Machine learning model/final_model.sav', 'rb'))
    to_predict_list = request.form.to_dict()
    review_text = to_predict_list['review_text']
    prediction = load_model.predict([review_text])
    prob = load_model.predict_proba([review_text])
    prob_score = prob[0][1]*20/100    
    #(print("The given statement is ",prediction[0]),
    #print("The truth probability score is ",prob[0][1]))

#function to perform search
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(':','&','|','!','?','-','$',',','@','#','%','^','*','"',"'",';',':','.','',' ')
    word_tokens = [] 
    word_tokens.append(word_tokenize(review_text))
    inputted=[]
    headlines=[]
    for j in word_tokens:
        for w in j:
            if w not in stop_words:
                inputted.append(ps.stem(w).encode('ascii', 'ignore').decode('ascii'))
    maxcount, normalize =  0, 0
    with open('Content Matching/newstitles.csv') as File:  
        reader = csv.reader(File)
        for row in reader:
            total=len(row)
            count=0
            for k in inputted:
                if k in row:
                    count=count+1
                else:
                    pass
            #print(count)
            if(count>maxcount):
                maxcount=count
                normalize=float(maxcount/total)
            else:
                pass
           
    cont_score = normalize*80/100
    #return cont_score
    final_score = prob_score+cont_score
    if(final_score>0.75):
        return jsonify({'Result': 'The given news is true'})
    elif(final_score>0.40):
        return jsonify({'Result': 'The given news may or maynot be true'})
    else:
        return jsonify({'Result': 'The given news is false'})
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
