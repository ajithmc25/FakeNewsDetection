from pandas import DataFrame

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from newsapi import NewsApiClient

import time
import datetime

import csv

print('start')
newsapi = NewsApiClient(api_key='2e7d3edd78df459fbad4ad03ccac582f')
processed_headlines=[]

def scrape(s1,s2):
    headlines=[]
    all_articles = newsapi.get_everything(
                                      sources='the-hindu,the-times-of-india',
                                      from_param=s1,
                                      to=s2,
                                      language='en')
    for j in all_articles['articles']:
        d=j['title']
        headlines.append(d)
    print(headlines)   
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(':','&','|','!','?','-','$',',','@','#','%','^','*','"',"'",';',':','.','',' ')
    word_tokens = [] 
    for j in headlines:
        word_tokens.append(word_tokenize(j))

    filtered=[]
   
    for j in word_tokens:
        for w in j:
            if w not in stop_words:
               filtered.append(ps.stem(w).encode('ascii', 'ignore').decode('ascii'))
        processed_headlines.append(filtered)#lemma.lemmatize(w))
        myFile = open('newstitles.csv', 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(processed_headlines)
        #excel = DataFrame({'title':processed_headlines}) 
        #excel.to_excel('newsheadlines.xlsx', sheet_name='sheet1', index='true')
        filtered=[]

#real time code
'''start =datetime.datetime.now().isoformat()
time.sleep(1500)
stop=datetime.datetime.now().isoformat()
while True:
        print(start)
        print(stop)
        scrape(str(start),str(stop))
        time.sleep(1500)
        start=stop
        stop=datetime.datetime.now().isoformat()'''

#old scraper
start = datetime.datetime.strptime('26/03/19 06:00', '%d/%m/%y %H:%M')

print(start)
stop=datetime.datetime.strptime('26/03/19 06:30', '%d/%m/%y %H:%M')
while (stop!=datetime.datetime.strptime('26/03/19 12:00', '%d/%m/%y %H:%M')):
        print(str(start))
        print(str(stop))
        scrape(str(start),str(stop))
        start=stop
        stop=stop + datetime.timedelta(seconds=1800)



    
    
       
    
    
