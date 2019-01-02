from dataClean import dataProcess
import json
fileName = r'/home/hl/cy/code/data/UK_firePro.json'
a = dataProcess(fileName, K=10)
data = a.jsonToDict()

class tweetEventHelper():
    def __init__(self, dict):
        self.dict = dict

    def search(self, keyword):
        keyword = keyword.lower()
        word = keyword.split(' ')
        res = []
        for key, value in self.dict.items():
            flag = 1
            # if value['text'].lower().find(keyword) != -1:
            #     res.append(key)
            #     continue
            for item in word:
                if (item not in value['words']):
                    flag = 0
                    break
            if(flag==1):
                res.append(key)
        # if(len(res)>500):
        #     bei = len(res)/500
        #     res = [res[i] for i in range(len(res)) if (i%int(bei)==0)]
        return res

    def showText(self, list):
        count = 0
        for item in list:
            count += 1
            temp = {}
            text = self.dict[item]['text']
            flag = text.find('http')
            text = text[:flag]
            temp[count] = text
#            temp['coordinate'] = self.dict[item]['coordinates']['coordinates']
            print(temp)


b = tweetEventHelper(data)
#word = ['brexit', 'fire', 'chester', 'chester fire']
n = 2
word = input("请输入你想查找的单词:")
res1 = b.search(word)  # brexit, Christmas
b.showText(res1)


