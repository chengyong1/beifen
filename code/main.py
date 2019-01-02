
from math import sqrt
from dataClean import dataProcess
from textSimilarity import writeSimilar 
from event import getTopKClusters, eventInfo, writeEvent 

file = r'/home/hl/cy/code/data/UK_firePro.json'
similarFile = r'/home/hl/cy/code/data/UKfiresimi.txt'
eventFile = r'/home/hl/cy/code/UKfireevent.txt'


def clean(K):
    cleaner = dataProcess(file, K=K)
    data = cleaner.jsonToDict()
    allwords = cleaner.highFreqWords 
    wordDiv, wordWithaprior = cleaner.wordsDiv(s=[2000*i for i in range(2, 7)], threshold=0.6)
    keywords = [item[0] for item in wordDiv if item[1]>=0]
    print("keywords:", keywords)
    print("keywords size:", len(keywords))
    return data, allwords, keywords, wordWithaprior


def similarity(K):
    data, allwords, keywords, wordWithaprior = clean(K)
    d = sqrt(23.4*10000/789)*100
    threshold = 2*d
    deltaT = 3600
    deltaD = d
    writeSimilar(data, similarFile+str(K), threshold, deltaT, deltaD, allwords, keywords, wordWithaprior)

def getEvent(eventNum, K, showDetail=True):
    eventClusters = getTopKClusters(eventNum, similarFile+str(K))
    data = dataProcess(file, K=K).jsonToDict()
    writeEvent(eventClusters, data, eventFile+str(K))
    if (showDetail==True):
        for count, i in enumerate(eventClusters):
            print("\n******************第{}个事件*******************".format(count+1))
            describe = eventInfo(data, i, 0.5, 15)
            print("时间:{}, 地点:{}, 关键词:{}\n".format(describe[0], describe[1], describe[2]))
#clean()

def test():
    Ks = [10, 15, 25, 30, 35, 40, 50][::-1]
    for K in Ks:
        similarity(K)
        getEvent(10, K)
#test()
from smallFunction import time2timestamp_ms
from database import Mymongo
"""settings"""
sinceData = "2018-11-17"
untilData = "2018-11-20"
databaseFile = "/home/hl/cy/code/data/USACA17_20.json"

def getDataFromDatabase(sinceData, untilData, fileName):

    Lab = Mymongo('mongodb://readAnyDatabase:Fzdwxxcl.121@121.49.99.14:30011', database='tweet_stream',
                      collection='USA')
    cursor = Lab.find_advance({"$and":[{'timestamp_ms': {'$gt': time2timestamp_ms(sinceData)}}, {'timestamp_ms': {'$lt': time2timestamp_ms(untilData)}}],'place.country': 'United States','coordinates':{"$nin": [None]}}, {'timestamp_ms': 1, 'text': 1, 'coordinates': 1, 'place.full_name': 1, 'user.followers_count': 1})
    #cursor = Lab.find_advance({"$and":[{'timestamp_ms': {'$gt': time2timestamp_ms(sinceData)}}, {'timestamp_ms': {'$lt': time2timestamp_ms(untilData)}}],'place.country': 'United States', 'coordinates': {"$nin": [None]}},{'timestamp_ms': 1, 'text': 1, 'coordinates': 1, 'place.full_name': 1, 'user.followers_count': 1})

    Lab.writeToJson(fileName=fileName, cursor=cursor)

getDataFromDatabase(sinceData, untilData, databaseFile)
