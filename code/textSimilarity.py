"""
计算两两推文间的文本相似度

"""
from aprori import p
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity 
from dataClean import dataProcess
from gensim import corpora, models, similarities
from  math import sqrt, radians, sin, cos, asin, log
from wordRecord import getNoisyWord
'''
path = r'/home/hl/cy/graduation2/data/dataGeo.json'
a = dataProcess(path, K=5)
data = a.jsonToDict()
allwords = a.highFreqWords
'''
def prob(words, data):
    return p(words, data)
def wordSignal(file):
    data = []
    words = []
    
    with open(file, 'r') as f:
        for line in f.readlines():
            temp = json.loads(line)
            key = list(temp.keys())[0]
            value = temp[key]['WCWTF']
            words.append(key)
            data.append(value)
    
    cosMat = cosine_similarity(data)
    res = {}
    for i in range(len(words)):
        res[words[i]] = {}
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if cosMat[i][j]>1e-4:
                res[words[i]][words[j]] = cosMat[i][j]
                res[words[j]][words[i]] = cosMat[j][i]
    return res

class textSimilar():

    def __init__(self, dict, allwords):
        self.dict = dict
        self.allwords = allwords 
        self.textSimilar = {}

    def similarMat(self):
        corpus = []
        for key, value in self.dict.items():
            text = ""
            for word in value['words']:
                if word in self.allwords:
                    text += word+" "
            corpus.append(text)
        X = CountVectorizer().fit_transform(corpus)
        tfidf = TfidfTransformer().fit_transform(X).toarray()
        print("tfidf size:", len(tfidf), len(tfidf[0]))

        return cosine_similarity(tfidf)

    '''提供字典供查询两两之间的文本相似度'''
    def textSimilarDict(self):
        count = 0
        for line in self.similarMat():
            count += 1
            self.textSimilar[count] = {}
            for i in range(count, len(line)):
                if(line[i] > 0.0):
                    self.textSimilar[count][i] = line[i]
        return self.textSimilar

class wordTimeSimilar():

    def __init__(self, dict, threshold, deltaT, deltaD, keywords, wordWithaprior):
        self.wordWithaprior = wordWithaprior 
        self.dict = dict
        self.length = len(self.dict.keys())  # 记录信息长度
        self.startTime = self.dict[1]['timestamp_ms']  # 记录信息起始时间
        self.endTime = self.dict[self.length]['timestamp_ms']  # 记录信息结束时间
        self.neibor = {}  # 记录邻居
       # self.dis = {}   # 记录距离
        self.threshold = threshold
        self.deltaT = deltaT
        self.deltaD = deltaD
        self.keywords = keywords 
        self.n = int((int(self.endTime)-int(self.startTime))/(1000*self.deltaT))


    '''按照经纬度算两点坐标  A[经度,纬度], B[经度,纬度]'''
    def distance(self, A, B):
        lon1, lat1, lon2, lat2 = map(radians, [A[0], A[1], B[0], B[1]])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # 地球平均半径，单位为公里
        return c * r * 1000

    '''
    获取推特编号为tweet_id的距离threshold内的所有推特的id
    注意这里采取边计算边保存结果策略，下次遇到相同的计算点时直接取出来用，避免重读计算       
    '''
    def getNeibor(self, tweet_id, threshold):
        A = self.dict[tweet_id]['coordinates']['coordinates']
        try:
            tweet_id_neibor = self.neibor[tweet_id]
        except:
            self.neibor[tweet_id] = []
            for i in range(self.length):
               # try:
                #    d = self.dis[tweet_id][i+1]
               # except:
                B = self.dict[i+1]['coordinates']['coordinates']
                d = self.distance(A, B)
                #    try:
                 #       self.dis[tweet_id][i+1] = d
                  #  except:
                   #     self.dis[tweet_id] = {}
                   # try:
                    #    self.dis[i+1][tweet_id] = d
                   # except:
                    #    self.dis[i+1] = {}
                if (d < threshold):
                    self.neibor[tweet_id].append(i+1)
            tweet_id_neibor = self.neibor[tweet_id]
        return tweet_id_neibor

    '''计算等长列表a，b的相似度'''
    def seriesSmilar(self, a, b):
        res = sum(map(lambda x, y: x * y, a, b)) / (sqrt(sum(map(lambda x: x ** 2, a))) * sqrt(sum(map(lambda x: x ** 2, b))))
        return res

    def wordTimeSeries(self, tweetid_i, tweetid_j):
        tweet_i = self.dict[tweetid_i]
        tweet_j = self.dict[tweetid_j]
        '''
        for word in tweet_i['words']:
            try:
                wordList = self.wordWithaprior[word]
                if set(wordList)&set(tweet_j['words']):
                    return 1
            except:
                pass
        '''
        A = tweet_i['coordinates']['coordinates']
        B = tweet_i['coordinates']['coordinates']
        dis = self.distance(A, B)
        n_all = 5
        if (dis==0):
            n_d = 0
        else:
            n_d = log(dis/self.deltaD)/log(2)
        if(n_d > 4):
            n_d = 4
        if(n_d < 0):
            n_d = 0
        n_t = n_all - n_d


        temp = [word for word in tweet_i['words'] if word in tweet_j['words']]
        commonWords = []
        for word in temp:
            if word in self.keywords:
                commonWords.append(word)

        '''如果两条推文都没有共同单词，那直接返回0，表示毫无关系'''
        if (len(commonWords)==0):
            return 0

        '''获取推特相邻threshold距离内的推特的id'''
        tweet_i_neibor = self.getNeibor(tweetid_i, self.threshold)
        tweet_j_neibor = self.getNeibor(tweetid_j, self.threshold)

        '''如果推文都没有满足在threshold距离内的邻居，该怎么办？'''
        if (len(tweet_i_neibor)==0 or len(tweet_j_neibor)==0):
            return 0

        '''接下来一个一个单词地去计算单词信号相似度，word_max_similar记录最大的单词相似度对应值'''
        word_max_similar = 0
        for word in commonWords:
            word_in_tweet_i_neibor_time = [self.dict[id]['timestamp_ms'] for id in tweet_i_neibor if word in self.dict[id]['words']]
            word_in_tweet_j_neibor_time = [self.dict[id]['timestamp_ms'] for id in tweet_j_neibor if word in self.dict[id]['words']]
            if(len(word_in_tweet_i_neibor_time)==0 or len(word_in_tweet_j_neibor_time)==0):
                continue
            temp_i = [int((int(time)-int(self.startTime))/(n_t*self.deltaT*1000)) for time in word_in_tweet_i_neibor_time]
            temp_j = [int((int(time) - int(self.startTime)) / (n_t * self.deltaT * 1000)) for time in word_in_tweet_j_neibor_time]
            signal_i = [0 for i in range(self.n)]
            signal_j = [0 for i in range(self.n)]

            for i in temp_i:
                signal_i[i] += 1
            for j in temp_j:
                signal_j[j] += 1
            index = self.seriesSmilar(signal_i, signal_j)
            if (index > word_max_similar):
                word_max_similar = index
        return word_max_similar


def writeSimilar(data, file, threshold, deltaT, deltaD, allwords, keywords, wordWithaprior):
    textSimilarity = textSimilar(data, allwords).textSimilarDict()
    temp = wordTimeSimilar(data, threshold, deltaT, deltaD, keywords, wordWithaprior)
    def similar(tweet_i, tweet_j):
        try:
            s1 = textSimilarity[tweet_i][tweet_j]
        except:
            s1 = 0
        if (s1 <= 0.01):
            return None
        s2 = temp.wordTimeSeries(tweet_i, tweet_j)
        if (s2 == 0):
            return None
        else:
            return (tweet_i, tweet_j, s1 * s2)
    n = len(data.keys())
    with open(file, 'w+', encoding='utf-8') as f:
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                t = similar(i, j)
                if (t is not None):
                    if (i%1000==0):
                        print("相似度计算已完成{}/{}".format(i, n))
                    f.write(str(i) + '\t' + str(j) + '\t' + str(t[2])[:6] + '\n')
def TweetSimilar(allwords, wordsdata, data):
    textSimilarity = textSimilar(data, allwords).textSimilarDict()
    print("textSimilarity has done!")
    print(textSimilarity.keys())
    apriori_p = prob(allwords, wordsdata)
    print("apriori_p has done")
    #wordSignalsimilar = wordSignal('/home/hl/cy/code/data/USACAwordsignal.json')
    print("wordSignal has done")
    n = len(data.keys())
    i = 0
    while(i<=n):
        i += 1
        if textSimilarity[i]=={} :
            continue 
        print(i)
        for j in range(i+1, n+1):
            try:
                s1 = textSimilarity[i][j]
            except:
                wordsi = data[i]['words']
                wordsj = data[j]['words']
                if len(wordsi)!=0 and len(wordsj)!=0:
                    sum = 0
                    for m in range(len(wordsi)):
                        for n in range(len(wordsj)):
                            try:
                                sum += apriori_p[wordsi[m]][wordsj[n]]
                            except:
                                sum += 0
                    s3 = sum/(len(wordsi)*len(wordsj))
                    if s3>0.0:
                        pass



#d = sqrt(41.4 * 10000 / 789) * 100
#threshold = 2 * d
#deltaT = 3600 * 2
#deltaD = d
#file = r'/home/hl/cy/graduation2/data/CAtest2.txt'
#writeSimilar(data, file, threshold, deltaT, deltaD, allwords, keywords, wordWithaprior)

    
path = r'/home/hl/cy/code/data/USACA17_20pro.json'
a = dataProcess(path, K=10)
data = a.jsonToDict()
allwords = a.highFreqWords
dataword = []
for key, value in data.items():
    dataword.append(value['words'])
TweetSimilar(allwords, dataword, data)
