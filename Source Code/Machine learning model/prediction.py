from __future__ import division

import pickle
import csv

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import warnings
warnings.filterwarnings("ignore")

var = input("Please enter the news text you want to verify: ")
print("You entered: " + str(var))
prob_score,cont_score = 0, 0

#function to run for prediction
def detecting_fake_news(var):    
#retrieving the best model for prediction call
    load_model = pickle.load(open('final_model.sav', 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])
    prob_score = prob[0][1]*20/100
    return prob_score 
    #(print("The given statement is ",prediction[0]),
    #print("The truth probability score is ",prob[0][1]))

#function to perform search
def news_search(var):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(':','&','|','!','?','-','$',',','@','#','%','^','*','"',"'",';',':','.','',' ')
    word_tokens = [] 
    word_tokens.append(word_tokenize(var))
    inputted=[]
    headlines=[]
    for j in word_tokens:
        for w in j:
            if w not in stop_words:
                inputted.append(ps.stem(w).encode('ascii', 'ignore').decode('ascii'))
    maxcount, normalize =  0, 0
    with open('newstitles.csv') as File:  
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
    return cont_score

if __name__ == '__main__':
    a = detecting_fake_news(var)
    b = news_search(var)
    print(a)
    print(b)
    final_score = a+b
    print(final_score)
    if(final_score>0.75):
        print('The given news is true')
    elif(final_score>0.40):
        print('The given news may or maynot be true')
    else:
        print('The given news is false')
