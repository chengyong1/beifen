"""
提供单词查询
"""
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(words):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(words):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return res

def getStopWords():
    stopWords = []
    with open(r'/home/hl/cy/code/data/stopWords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stopWords.append(line.split('\n')[0])
    return stopWords

def getNoisyWord():
    noisyWords = {}
    with open(r'/home/hl/cy/code/data/eventBased.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            temp = line.split('\n')[0].split(',')
            noisyWords[temp[0][2:-1]] = 1
    return noisyWords

if __name__ == '__main__':
    print(getNoisyWord()['lose'])
