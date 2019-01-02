from geopy.geocoders import Nominatim
import community
import time
import networkx as nx 
import numpy as np
from dataClean import dataProcess
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.cluster import Birch, KMeans,SpectralClustering
from translet import totalTrans 
from test import trans 
'''
path = r'/home/hl/cy/graduation2/data/USA_CApro.json'
a = dataProcess(path,K=20)
data = a.jsonToDict()
'''
def tfidf(data, allwords):
    corpus = []
    for key, value in data.items():
        text = ""
        for word in value['words']:
            if word in allwords:
                text += word+" "
        corpus.append(text)
    X = CountVectorizer().fit_transform(corpus)
    tfidfMat = TfidfTransformer().fit_transform(X).toarray()
    print("tfidfMat size:{}*{}".format(len(tfidfMat), len(tfidfMat[0])))
    return tfidfMat
def PCADecomposition(tfidfMat, dimension):
    result = PCA(n_components=dimension).fit_transform(tfidfMat)
    print("PCADecomposition size:{}*{}".format(len(result), len(result[0])))
    return result 
def cluster_birch(result, k):
    clusters = Birch(n_clusters=k).fit_predict(result)
    return clusters
def cluster_Kmeans(result, k):
    clusters = KMeans(n_clusters=k).fit_predict(result)
    return clusters 
def cluster_spectral(result, k):
    clusters = SpectralClustering(n_clusters=k).fit_predict(result)
    return clusters 
def getClusters(clusters):
    length = len(set(clusters))
    temp = [[] for _ in range(length)]
    for i in range(len(clusters)):
        temp[clusters[i]].append(i+1)
    return temp

def load_graph(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f.readlines():
            temp = line.split('\n')[0].split('\t')
            data.append((int(temp[0]), int(temp[1]), float(temp[2])))
    return data 
def getTopKClusters(k, filename):
    data = load_graph(filename)
    G = nx.Graph()
    G.add_weighted_edges_from(data)
    partition = community.best_partition(G)
    clusterNum = len(set(partition.values()))
    res = [[]for _ in range(clusterNum)]
    for key, value in partition.items():
        res[value].append(key)    
    res = sorted(res, key=lambda b: len(b), reverse=True)
    return res[:k]


def findCloset(a, k):
    if(len(a)==0):
        return 0
    if(len(a)==1):
        return a[0]
    if(len(a)==2):
        return (a[0]+a[1])/2
    a.sort()
    dif = []
    for i in range(1, len(a)):
        dif.append(a[i]-a[i-1])
    temp = np.argsort(dif)
    anw = []
    for i in temp:
        anw.append([a[i+1], a[i]])
    n = int(len(anw)*k)  # 只取一半的
    res = anw[:n]
    res = set([item for line in res for item in line])
    sum = 0
    for item in res:
        sum += item
    return sum/len(res)

def getLocation(coordinates):
    geolocator = Nominatim()
    locStr = str(coordinates[1])+', '+str(coordinates[0])
    location = geolocator.reverse(locStr, timeout=1000)
    return location

def getTime(timestamp_ms):
    timeStamp = int(timestamp_ms)/1000
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M", timeArray)
    return otherStyleTime

def eventInfo(data, cluster, k, wordsLen):
    words = {}
    times = []
    coordinates_x = []
    coordinates_y = []
    for item in cluster:
        for word in data[item]['words']:
            words[word] = words.get(word, 0) + 1
        times.append(int(data[item]['timestamp_ms']))
        coordinates_x.append(data[item]['coordinates']['coordinates'][0])
        coordinates_y.append(data[item]['coordinates']['coordinates'][1])
    a = sorted(words.items(), key=lambda b: b[1], reverse=True)
    if(len(a) >= wordsLen):
        a = a[:wordsLen]
    else:
        a = a
    eventWords = [t[0] for t in a]
    data = getTime(findCloset(times, k))
    coordinates = [findCloset(coordinates_x, k), findCloset(coordinates_y, k)]
    return [data, coordinates, eventWords]

def writeEvent(eventClusters, data, file):
    f = open(file, 'w+', encoding='utf-8')
    f.write('       time '+'      ||     Location  ' + '  ||             describe     ' + '\n')
    for event in eventClusters:
        content = eventInfo(data, event, 0.5, 15)
        f.write(content[0]+'  ||  '+'['+str(content[1][0])[:5]+','+str(content[1][1])[:5]+']' + '  ||  ')
        for word in content[2]:
            f.write(word+' ')
        f.write('\n')
    f.close()
def timeConvert(timestamp):
    time_local = time.localtime(timestamp)
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    return dt 
if __name__ == '__main__':
    
    filename = '/home/hl/cy/code/data/simi.txt30'
    a = getTopKClusters(5, filename)
    path = r'/home/hl/cy/code/data/USA_CApro.json'
    data = dataProcess(path, K=5).jsonToDict()
    '''
    path = '/home/hl/cy/code/data/USA20_24pro.json'
    dataObj = dataProcess(path,K=5)
    data = dataObj.jsonToDict()
    allwords = dataObj.highFreqWords 
    tfidfMat = tfidf(data, allwords)
    result = PCADecomposition(tfidfMat, 30)
    clusters = cluster_Kmeans(result, 10)
    a = getClusters(clusters)
    '''
#    writeEvent(a, r'/home/hl/cy/graduation2/data/eventInfoNew.txt')
    f =  open('/home/hl/cy/code/data/eventTweet.txt', 'w+')
    for count, i in enumerate(a):
        n = input('按任意键继续')
        print("\n#######################第", count+1, "事件#######################")
        print("推文数量:", len(i))
        describe = eventInfo(data, i, 0.5, 15)
        print("\n时间：", describe[0], "    地点：", describe[1], "\n关键词描述：", describe[2], '\n')
        print(totalTrans(describe[2]))
        print("具体推文信息：")
        dif = int(len(i)/100) if len(i)>=100 else 1
        textList = []
        count += 1
        f.write("事件{}".format(count)+'\t'+str(describe[2])+'\n')
        for c, j in enumerate(i):
            f.write(str(j)+'\t'+timeConvert(int(int(data[j]['timestamp_ms'])/1000))+'\t'+data[j]['text']+'\n')
            if (c+1)%dif == 0:
                text = data[j]['text']
                flag = text.find('http')
                print({j: data[j]['text'][:flag-2]})
                textList.append(data[j]['text'][:flag-2])
    f.close()
