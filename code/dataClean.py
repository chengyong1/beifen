"""
1读入原始数据，进行处理
2、提供一个大字典
"""

import json
from wordRecord import getStopWords, lemmatize_sentence
from collections import Counter
from math import radians, asin, sqrt, sin, cos, pi
import pyfpgrowth
from nltk.corpus import wordnet

class dataProcess():

    def __init__(self, file, K):
        self.file = file
        self.K = K
        self.stopWords = getStopWords()
        self.data = self.readJson()
        self.dict = self.jsonToDict()
        self.dis = {}
        self.highFreqWords = self.getHighFreqWords(self.K)[::-1]

    def readJson(self):
        data = []
        with open(self.file, 'r', encoding='utf-8') as f:
            for line in f:
                temp = json.loads(line)
                data.append(temp)
        print("--------------------一共{}条信息--------------------".format(len(data)))
        return data

    def getHighFreqWords(self, K):
        words = []
        for data in self.data:
            words += list(set(data['words']))
        wordsFreq = Counter(words).most_common(10000)[80:]
        words = []
        for word, freq in wordsFreq:
            if (freq < K):
                return words
            if (wordnet.synsets(word)!=[]):  # 判断是不是单词
                words.append(word)
        return words

    def textProcess(self, text):
        newStr = ' '
        for i in range(len(text)):
            if (text[i:i + 4] == 'http'):  # 去除网址
                break
            if (text[i].isalpha() or text[i].isdigit() or text[i] == '\''):  # 去除符号
                newStr += text[i].lower()
            elif newStr[-1] != ' ':
                newStr += ' '
        temp = newStr.split(' ')
        words = []
        for word in temp:
            if word == '' or word.isdigit():  # 去除空白和数字单词
                continue
            words.append(word)
        words = lemmatize_sentence(words)  # 词形还原
        bingji = set(words) & set(self.stopWords)
        words = list(set(words) ^ bingji)  # 去除停用词
        return words

    def writeJson(self, path):
        with open(path, 'w+', encoding='utf-8') as f:
            data = self.data 
            num = len(data)
            count = 0
            for item in data:
                if item['place']['full_name'][-2:]=='CA':
                    count += 1
                    if count==5000:
                        break
                    if (count%100==0):
                        print("{}/{}".format(count, num))
                    item['words'] = self.textProcess(item['text'])
                    item['@'] = dataProcess.getAitewords(item['text'])
                    item['hashtag'] = dataProcess.getHashtag(item['text'])
                    item['place'] = dataProcess.getPlace(item['text'])
                    item['followers_count'] = item['user']["followers_count"]
                    json_str = json.dumps(item)
                    f.write(json_str+'\n')

    def jsonToDict(self):
        data = self.data
        new = {}
        count = 0
        for item in data:
            count += 1
            new[count] = item
        return new

    @classmethod
    def getAitewords(cls, text):
        res = []
        index = 0
        end = 0
        while (index < len(text)):
            if text[index:index + 1] == '@':
                index += 1
                if (index > len(text) - 1):
                    break
                start = index
                if text[index] == ' ':
                    start += 1
                    while (text[index].isalpha() or text[index] == ' ' or text[index] == '\''):
                        if (index == len(text) - 1):
                            break
                        index += 1
                        end = index
                    temp = text[start: end].split(' ')
                    for item in temp:
                        if item.isalpha():
                            res.append(item.lower())
                else:
                    while (text[index].isalpha()):
                        if (index == len(text) - 1):
                            break
                        index += 1
                        end = index
                    res.append(text[start: end].lower())
            index += 1
        return res

    @classmethod
    def getHashtag(cls, text):
        res = []
        index = 0
        end = 0
        while (index < len(text)):
            if text[index:index + 1] == '#':
                index += 1
                if (index > len(text) - 1):
                    break
                start = index
                if text[index] == ' ':
                    start += 1
                    while (text[index].isalpha() or text[index] == ' ' or text[index] == '\''):
                        if (index == len(text) - 1):
                            break
                        index += 1
                        end = index
                    temp = text[start: end].split(' ')
                    for item in temp:
                        if item.isalpha():
                            res.append(item.lower())
                else:
                    while (text[index].isalpha()):
                        if (index == len(text) - 1):
                            break
                        index += 1
                        end = index
                    res.append(text[start: end].lower())
            index += 1
        return res

    @classmethod
    def getPlace(cls, text):
        res = []
        index = 0
        end = 0
        while (index < len(text)):
            if text[index - 1:index + 1] == 'in' or text[index - 1:index + 1] == 'of':
                index += 1
                if (index > len(text) - 1):
                    break
                start = index
                if text[index] == ' ':
                    start += 1
                    while (text[index].isalpha() or text[index] == ' ' or text[index] == '\''):
                        if (index == len(text) - 1):
                            break
                        index += 1
                        end = index
                    temp = text[start: end].split(' ')
                    for item in temp:
                        if item.isalpha():
                            res.append(item.lower())
                else:
                    while (text[index].isalpha()):
                        if (index == len(text) - 1):
                            break
                        index += 1
                        end = index
                    res.append(text[start: end].lower())
            index += 1
        return res

    def isNoisyWord(self, word):
        data = self.getTweetContainSpencifyWord(word)
        length = len(data)
        count = 0
        for item in data:
            AiteWords = item['@']+item['place']
            if word in AiteWords:
                count += 1
        if (length==0):
            return False
#        print(word + " {}/{}={}".format(count, length, count / length), end=' ')
        if count/length>=0.4:
 #           print(", is @ word")
            return True
  #      print(word, "is not a @ word")
        return False

    @classmethod
    def isRepeat(cls, text1, text2):
        def getCommonStr(text1, text2):
            record = [[0 for _ in range(len(text2) + 1)] for _ in range(len(text1) + 1)]
            maxNum = 0  # 记录匹配长度
            p = 0  # 匹配的起始位
            for i in range(len(text1)):
                for j in range(len(text2)):
                    if (text1[i] == text2[j]):
                        record[i + 1][j + 1] = record[i][j] + 1
                        if (record[i + 1][j + 1] > maxNum):
                            maxNum = record[i + 1][j + 1]
                            p = i + 1
            return text1[p - maxNum: p], maxNum
        maxCommonLength = getCommonStr(text1, text2)
        if maxCommonLength[1] > min(len(text1), len(text2))*0.4:
            return True
        return False

    @classmethod
    def commonwords(cls, data_i, data_j):
        return [word for word in data_i['words'] if word in data_j['words']]

    def isKeyWord(self, word):
        word = word.lower()
        if self.isNoisyWord(word):
            return False
        data = self.getTweetContainSpencifyWord(word)
        startLen = len(data)
        i = 0
        samecount = 0
        endLen = len(data)
        while (i < endLen):
            j = i + 1
            while (j < endLen):
                if (data[i]['coordinates']['coordinates']==data[j]['coordinates']['coordinates']
                    or len(dataProcess.commonwords(data[i], data[j])) > min(len(data[i]['words']), len(data[j]['words']))*0.4
                   # or dataProcess.isRepeat(data[i]['text'], data[j]['text'])
                    or (data[i]['text'][:10]==data[j]['text'][:10])):
                    if (data[i]['words'] == data[j]['words']):
                        samecount += 1
                    del data[j]
                    endLen -= 1
                else:
                    j += 1
            i += 1
        if startLen==0:
            return True
   #     print(word, "{}/{}={}".format(endLen-samecount, startLen, (endLen-samecount) / startLen), end=' ')
        if (endLen-samecount)/startLen<=0.35 or endLen <= 5:
    #        print("in repeated tweets!")
            return False
     #   print(word, "can be reserved!")
        return True

    '''按照经纬度算两点坐标  A[经度,纬度], B[经度,纬度]'''
    def distance(self, A, B):
        lon1, lat1, lon2, lat2 = map(radians, [A[0], A[1], B[0], B[1]])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # 地球平均半径，单位为公里
        return c * r * 1000

    """
        word: 传入一个关键词
        s: 传入一个距离向量, eg.[100, 200, 300, 400]
        return：返回这个单词的空间CSR指数
        """
    def spaceCSR(self, word, s):
        # 取出包含word的推文的编号
        idList = [key for key, value in self.dict.items() if word in value['words']]
        length = len(idList)
        count = [0 for i in range(len(s))]  # 记录满足距离小于d的推文的对数
        for i in range(length):
            for j in range(i + 1, length):
                try:
                    d = self.dis[(idList[i], idList[j])]
                except:
                    A = self.dict[idList[i]]['coordinates']['coordinates']  # 取出两个点的经纬度
                    B = self.dict[idList[j]]['coordinates']['coordinates']
                    self.dis[(idList[i], idList[j])] = self.distance(A, B)  # 计算球面距离
                    d = self.dis[(idList[i], idList[j])]
                for i in range(len(s)):
                    if (d <= s[i]):
                        count[i] += 1
        k = [400000 * n / (length ** 2) for n in count]
        l = [abs(((k[i] / pi) ** 0.5 - s[i] / 1000) / 100) for i in range(len(s))]
        return sum(l) / len(l)
    def timeCSR(self, word, s, timeInterval, verbose):
        idList = [int(value['timestamp_ms'])/1000 for key, value in self.dict.items() if word in value['words']]
        length = len(idList)
        count = [0 for i in range(len(s))]
        for i in range(length):
            for j in range(i+1, length):
                timeDif = abs(idList[i]-idList[j])
                for i in range(len(s)):
                    if (timeDif <= s[i]):
                        count[i] += 1
        k = [timeInterval*2*n/length**2 for n in count]
        l = [abs((k[i]) - s[i]) for i in range(len(s))]
        if verbose==True:
            print("word:{}  s={} tweetNum:{}\ncount={}\nk={}\nl={}".format(word, s, length, count, k, l))
        return sum(l)/len(l)

    def allWordsCSR(self, s):
        allWords = self.highFreqWords
        res = {}
        length = len(allWords)
        count = 0
        for word in allWords:
            count += 1
            if (count % 100 == 0):
                print("allWordsCSR已完成{}/{}".format(count, length))
            res[word] = self.spaceCSR(word, s=s)
        return res
    def allWordsTimeCSR(self, s, verbose=True):
        allwords = self.highFreqWords 
        startTime = self.data[1]['timestamp_ms']
        endTime = self.data[-1]['timestamp_ms']
        timeInterval = (int(endTime)-int(startTime))/1000
        res = {}
        for word in allwords:
            res[word] = self.timeCSR(word, s, timeInterval, verbose)
            if (verbose==True):
                print("{}:{}".format(word, res[word]))
        return res 

    def getTweetContainSpencifyWord(self, word):
        res = []
        for item in self.data:
            if word in item['words']:
                res.append(item)
        return res

    def apriori(self, minconfidence):
        transactions = []
        for data in self.data:
            transactions.append(data['words'])
        patterns = pyfpgrowth.find_frequent_patterns(transactions, self.K)
        rules = pyfpgrowth.generate_association_rules(patterns, minconfidence)
        words = self.highFreqWords
        length = len(words)
        count = 0
        wordsWithApriori = {}
        for word in words:
            count += 1
            if (count%100==0):
                print("apriori已完成{}/{}".format(count, length))
            temp = ()
            for key, value in rules.items():
                if (word in key or word in value[0]):
                    temp += key
                    temp += value[0]
            wordList = []
            for item in set(temp):
                wordList.append(item)
            wordsWithApriori[word] = wordList
        return wordsWithApriori
    
    def wordsDiv(self, s, threshold):
        words = self.highFreqWords
        wordsDict = {word: 0 for word in words}
        wordsWithApriori = self.apriori(minconfidence=1/self.K)
        wordsWithCSR = self.allWordsCSR(s=s)
        length = len(words)
        count = 0
        for word, csr in wordsDict.items():
            count += 1
            if (count % 100 == 0):
                print("wordDiv已完成{}/{}".format(count, length))
            flag = 1
            if (wordsWithCSR[word] < threshold):
                wordsDict[word] -= 3
                flag = 0
            else:
                if self.isKeyWord(word):
                    wordsDict[word] += 3
                else:
                    wordsDict[word] -= 3
                    flag = 0
            aprioriWords = wordsWithApriori[word]
            for item in aprioriWords:
                if item in words:
                    if (flag == 0):
                        wordsDict[item] -= 1
                    else:
                        wordsDict[word] += 1
        res = sorted(wordsDict.items(), key=lambda item: -item[1])
        for i in range(len(res)):
            res[i] = (res[i][0], res[i][1], wordsWithCSR[res[i][0]])
        return res, wordsWithApriori 

class similar():
    def __init__(self, data, eventWords):
        '''data是一个大字典，包含所有推文数据'''
        for key, value in data.items():
            temp = []
            for word in value['words']:
                if word in eventWords:
                    temp.append(word)
            value['words'] = temp
        self.data = {}
        count = 0
        for key, value in data.items():
            if not value['words']:
                continue
            count += 0
            self.data[count] = value 
        print("处理后数据大小:{}".format(count))
        self.eventWords = eventWords

    def textSimilar(self):
        corpus = []
        for key, value in self.data.items():
            text = ""
            for word in value['words']:
                text += word+" "
            corpus.append(text)
        X = CountVectorizer().fit_transform(corpus)
        tfidf = TfidfTransformer().fit_transform(X).toarray()
        print("tfidf size:", len(tfidf), len(tfidf[0]))
        temp = cosine_similarity(tfidf)
        print("cosMat size:{}*{}".format(len(temp), len(temp[0]))) 
        return temp
    def apriori_p(self):
        data = []
        for key, value in self.data.items():
            data.append(value['words'])
        words = self.eventWords 
        stats = {}
        for item in words:
            stats[item] = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] in words:
                    stats[data[i][j]].append(i+1)
        p = {}
        for i in range(len(words)):
            p[words[i]] = {}
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                common = set(stats[words[i]])&set(stats[words[j]])
                p[words[i]][words[j]] = len(common)/len(stats[words[i]])
                p[words[j]][words[i]] = len(common)/len(stats[words[j]])
        reurn p
    def wordSignal(self):


if __name__ == '__main__':
    file = r'data/USA_CApro.json'
    dataPrepare = dataProcess(file, K=100)
    dataPrepare.timeCSR(word='night', timeInterval=72*3600, s=[1800, 3600, 7200], verbose=True)
   # a = dataProcess(path, K=10)
   # print(a.highFreqWords)
   # print(len(a.highFreqWords))
 #   res = a.wordsDiv(s=[2000*i for i in range(2, 7)], threshold=0.7)
  #  print(res)
    # a.writeJson(path)






