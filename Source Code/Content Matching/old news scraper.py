from newsapi import NewsApiClient
import time
import datetime
print('start')
l=[]
newsapi = NewsApiClient(api_key='c7632aa99be04e119c107abb5501ba91')

def scrape(s1,s2):
    all_articles = newsapi.get_everything(
                                      sources='the-hindu,the-times-of-india',
                                      from_param=s1,
                                      to=s2,
                                      language='en')
    print(all_articles)
    for j in all_articles['articles']:
        d=dict(title=j['title'] , content=j['content'])
        l.append(d)
        
    print(l)
    

start = datetime.datetime.strptime('01/02/19 00:00', '%d/%m/%y %H:%M')

print(start)
stop=datetime.datetime.strptime('01/02/19 00:30', '%d/%m/%y %H:%M')
while True:
        print(str(start))
        print(str(stop))
        scrape(str(start),str(stop))
        start=stop
        stop=stop + datetime.timedelta(seconds=1800) 
        
        



    
    
       
    
    
