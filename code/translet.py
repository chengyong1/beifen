from googletrans import Translator
import multiprocessing
from dataClean import dataProcess
import json
import urllib
def trans(line):
    url='http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&sessionFrom='
    data={}
    data['i'] = line
    data['from'] = 'AUTO'
    data['to'] = 'AUTO'
    data['smartresult'] = 'dict'
    data['client'] = 'fanyideskweb'
    data['salt'] = '1517799189818'
    data['sign'] = '8682192c0707c52ecdffbc98f77a17ac'

    data['doctype'] = 'json'
    data['version'] = '2.1'
    data['keyfrom'] = 'fanyi.web'
    data['action'] = 'FY_BY_CLICKBUTTION'
    data['typoResult'] = 'true'
    data = urllib.parse.urlencode(data).encode('utf-8')
    response = urllib.request.urlopen(url,data)
    html = response.read().decode('utf-8')
    translate_results = json.loads(html)
    translate_results = translate_results['translateResult'][0][0]['tgt']
    return translate_results 
def totalTrans(sourceList):
    textList = []
    pool = multiprocessing.Pool(processes=120)
    for source in sourceList:
        a = pool.apply_async(trans, (source, ))
        textList.append(a.get())
    pool.close()
    pool.join()
    return textList 

if __name__=="__main__":
    src = ["程", "中国","强大", "暗示", "厉害"]
    print(totalTrans(src))
