import pyfpgrowth 

def p(words, data):
    # input: a 2-dimension words list
    # output: a dict for Conditional probability of occurrence of two words
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
            if len(common)<=1e-4:
                continue
            p[words[i]][words[j]] = len(common)/len(stats[words[i]])
            p[words[j]][words[i]] = len(common)/len(stats[words[j]])
    return p
if __name__=="__main__":
    from collections import Counter
    import json
    data = []
    words = []
    with open('/home/hl/cy/code/data/USA_CApro.json', 'r') as f:
        for line in f.readlines():
            temp = json.loads(line)
            data.append(temp['words'])
            for word in temp['words']:
                words.append(word)
    words = Counter(words)
    new = []
    for key, value in words.items():
        if value>=50:
            new.append(key)
    print(len(new))
    print(p(new, data))
