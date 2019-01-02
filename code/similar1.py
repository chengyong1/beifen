from nltk.corpus import wordnet
from collections import Counter
import json
from math import log, sin, pi, sqrt
import numpy as np
import pywt

class dataHelper():
    def __init__(self, fileName):
        self.fileName = fileName

    def readJson(self):
        data = []
        with open(self.fileName, 'r') as f:
            for line in f:
                temp = json.loads(line)
                # vec = list(temp.values())[0][1:-1].split(',')
                # vec = [float(item[:8]) for item in vec]
                # temp[list(temp.keys())[0]] = vec
                data.append(temp)
        return data

    # 获取p小时推文
    def getData(self, p):
        data = []
        p_interval = 1000 * p * 3600
        with open(self.fileName, 'r') as f:
            count = 0
            for line in f:
                count += 1
                if (count == 1):
                    startTime = json.loads(line)['timestamp_ms']
                    endTime = str(int(startTime) + p_interval)
                temp = json.loads(line)
                data.append(temp)
                if (temp['timestamp_ms'] > endTime):
                    return data
        return data


class statics():
    def __init__(self, data):
        self.data = data
        print("All tweet Num:", self.tweetNum())
        self.Nw = {}
        self.wordFrequent = {}

    # 推特总数
    def tweetNum(self):
        return len(self.data)

    # 包含某个单词的推特数
    def containWordTweetNum(self, word):
        try:
            res = self.Nw[word]
        except:
            count = 0
            for item in self.data:
                if (word in item['words']):
                    count += 1
            self.Nw[word] = count
            res = count
        return res

    # 总的单词数目
    def totalWordsNum(self):
        count = 0
        for item in self.data:
            count += len(item['words'])
        return count

    # 所有单词出现频次，返回一个可供查询的字典
    def wordFrequecy(self):
        if self.wordFrequent:
            return self.wordFrequent
        allWords = []
        for item in self.data:
            allWords += item['words']
        self.wordFrequent = Counter(allWords)
        return self.wordFrequent

    # 取出出现频数超过K的单词
    def getTopKWords(self, K):
        res = self.wordFrequecy()
        words = []
        for key, value in res.items():
            if (value >= K) and wordnet.synsets(key)!=[]:
                words.append(key)
        print("TopKwords Num:", len(words))
        return words

    # 取出数据内所有followers数量
    def allFollowers(self):
        count = 0
        for item in self.data:
            count += item['user']['followers_count']
        return count

    # 取出数据中所有retweet数量
    def allRetweet(self):
        count = 0
        for item in self.data:
            count += item['retweet_count']
        return count


class similarity():
    def __init__(self, data, k):
        self.k = k
        self.data_p = data
        self.base_p = statics(self.data_p)
        self.k_interval = int(len(self.data_p) / k)  # 将数据划分，k表示是p的k分之一
        self.data_k = []
        for i in range(k):
            temp = self.data_p[i * self.k_interval: (i + 1) * self.k_interval]
            self.data_k.append(temp)
        self.wordFrequcy_p = self.base_p.wordFrequecy()
        self.totalWords_p = self.base_p.totalWordsNum()
        self.totalFollowers_p = self.base_p.allFollowers()

    # 这里用正弦函数作为模糊函数，0到90度
    @classmethod
    def fuzzyFunc(cls, window_with):
        delta = 90 / window_with
        seta = [90 - i * delta for i in range(window_with)][::-1]
        weight = map(lambda x: sin(x * pi / 180), seta)
        return list(weight)

    def CWTF_ITWTF(self, word):
        CWTF = map(lambda x: x / self.k_interval, [statics(item).containWordTweetNum(word) for item in self.data_k])
        ITWTF = self.base_p.tweetNum() / self.base_p.containWordTweetNum(word)
        return list(map(lambda x: x * log(ITWTF) / log(2), CWTF))

    def WF_ITWF(self, word):
        WF = map(lambda x, y: x / y, [statics(item).wordFrequecy()[word] for item in self.data_k],
                 [statics(item).totalWordsNum() for item in self.data_k])
        ITWF = self.totalWords_p / self.wordFrequcy_p[word]
        return list(map(lambda x: x * log(ITWF) / log(2), WF))

    def WCWTF_ITWCWTF(self, word):
        WCWTF_molecular = [0 for i in range(self.k)]
        WCWTF_denominator = [0 for i in range(self.k)]
        WCWTF = [0 for i in range(self.k)]
        for i in range(self.k):
            for item in self.data_k[i]:
                if (word in item['words']):
                    WCWTF_molecular[i] += item['user']['followers_count']
                WCWTF_denominator[i] += item['user']['followers_count']
            WCWTF[i] = WCWTF_molecular[i] / WCWTF_denominator[i]

        ITWCWTF_molecular = 0
        for item in self.data_p:
            if (word in item['words']):
                ITWCWTF_molecular += item['user']['followers_count']
        ITWCWTF_denominator = self.totalFollowers_p
        ITWCWTF = ITWCWTF_denominator / (ITWCWTF_molecular+1)

        return list(map(lambda x: x * log(ITWCWTF) / log(2), WCWTF))

    @classmethod
    def Fuzzy(cls, list, window_with):
        res = [0 for _ in range(len(list))]
        weight = cls.fuzzyFunc(window_with)
        if (len(list) <= len(weight)):
            for i in range(len(list)):
                for j in range(i + 1):
                    res[i] += weight[j] * list[j]
        else:
            for i in range(len(weight)):
                for j in range(i + 1):
                    res[i] += weight[j] * list[j]
            for i in range(len(weight), len(list)):
                for j in range(i - len(weight) + 1, i + 1):
                    res[i] += weight[j - i + len(weight) - 1] * list[j]
        return res


class wordsSimilarity():
    def __init__(self, data, k, topkwords):
        self.data = data
        self.k = k
        self.topkwords = topkwords

    def wordsSignal(self):
        topKwords = statics(self.data).getTopKWords(self.topkwords)
        simili = similarity(self.data, self.k)
        res = []
        for word in topKwords:
            temp = simili.WCWTF_ITWCWTF(word)
            signal = simili.Fuzzy(temp)
            res.append(signal)
        return res

    @classmethod
    def crossCorrelation(cls, vec1, vec2):
        try:
            return sum(map(lambda x, y: x * y, vec1, vec2)) / (
            sqrt(sum(map(lambda x: x ** 2, vec1))) * sqrt(sum(map(lambda x: x ** 2, vec2))))
        except:
            return 0

    @classmethod
    def filterNoisy(cls, y, level):
        coeffs = pywt.wavedec(y, 'db1', level=level)
        def sgn(x):
            if x>0:
                return 1.0
            elif x==0:
                return 0.0
            else:
                return -1.0
        thcoeffs = []
        a = 0.5  # 软硬阈值折衷法 a 参数
        for i in range(1, len(coeffs)):
            tmp = coeffs[i].copy()
            Sum = 0.0
            for j in coeffs[i]:
                Sum = Sum + abs(j)
            N = len(coeffs[i])
            Sum = (1.0 / float(N)) * Sum
            sigma = (1.0 / 0.6745) * Sum
            lamda = sigma * math.sqrt(2.0 * math.log(float(N), math.e))
            for k in range(len(tmp)):
                if (abs(tmp[k]) >= lamda):
                    tmp[k] = sgn(tmp[k]) * (abs(tmp[k]) - a * lamda)
                else:
                    tmp[k] = 0.0
            thcoeffs.append(tmp)
        usecoeffs = []
        usecoeffs.append(coeffs[0])
        usecoeffs.extend(thcoeffs)
        y_processed = pywt.waverec(usecoeffs, 'db1')
        return y_processed, coeffs[0]

    def similarDict(self, wordSignalList):
        res = {}
        length = len(wordSignalList)
        for i in range(length):
            key_i = list(wordSignalList[i].keys())[0]
            value_i = list(wordSignalList[i].values())[0]
            res[key_i] = {}
            if (i%1000==0):
                print(i)
            for j in range(length):
                if ( i!=j ):
                    key_j = list(wordSignalList[j].keys())[0]
                    value_j = list(wordSignalList[j].values())[0]
                    dis = self.crossCorrelation(value_i, value_j)
                    res[key_i][key_j] = dis
        return res



    def similarMat(self):
        vec = self.wordsSignal()
        length = len(vec)
        Mat = [[0 for i in range(length)] for j in range(length)]
        for i in range(length):
            for j in range(i + 1, length):
                Mat[i][j] = Mat[j][i] = self.crossCorrelation(vec[i], vec[j])
        return Mat

    def writeSimilarMat(self, fileName):
        mat = self.similarMat()
        with open(r'C:\Users\程勇\PycharmProjects\graduation3\data\\' + fileName, 'w+', encoding='utf-8') as f:
            for line in mat:
                for item in line:
                    f.write(item + '\t')
                f.write('\n')

class tweetSimilar():
    def __init__(self, data, similarDict):
        self.data = data
        self.similarDict = similarDict

    def tweetSimilarity(self, threshold):
        num = len(self.data)
        fw = open('similarity.txt', 'w+', encoding='utf-8')
        for i in range(num):
            words_i = self.data[i]['words']
            if(i%1000==0):
                print(i)
            for j in range(i+1, num):
                words_j = self.data[j]['words']
                commonWords = [word for word in words_i if word in words_j]
                if ( len(commonWords)==0 ):
                    continue
                maxSimilar = 0
                for m in range(len(commonWords)):
                    for n in range(len(commonWords)):
                        if ( m!=n ):
                            try:
                                t = self.similarDict[commonWords[m]][commonWords[n]]
                            except:
                                t = 0
                            if ( t > maxSimilar ):
                                maxSimilar = t
                if ( maxSimilar >= threshold ):
                    print(i+1, j+1, maxSimilar)
                    fw.write(str(i+1)+'\t'+str(j+1)+'\t'+str(maxSimilar)[:6]+'\n')
        fw.close()


if __name__ == '__main__':
    a = dataHelper(r'/home/hl/cy/code/data/USACA17_20pro.json')
    data = a.getData(24*3)
    b = statics(data)
    keyWords = b.getTopKWords(10)
    k_interval = int(24*3*60/10)
    c = similarity(data, k_interval)

    window = 60
    q = int(k_interval/2)
    
    length = len(keyWords)
    count = 0
    fw = open('/home/hl/cy/code/data/USACAwordsignal.json', 'w+', encoding='utf-8')
    for word in keyWords:
        count += 1
        if (count % 10 == 0):
            print("{}/{}".format(count, length))
        temp = c.WCWTF_ITWCWTF(word)
        signal = c.Fuzzy(list=temp, window_with=window)
        dwt = list(np.array(pywt.dwt(signal, 'haar')).reshape(1, -1)[0][:q])
        content = {word: {'WCWTF': temp, 'fuzzy': signal, 'dwt': dwt}}
        json_str = json.dumps(content)
        fw.write(json_str + '\n')
    fw.close()

