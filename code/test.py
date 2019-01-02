import urllib.request
import urllib.parse
import json
from dataClean import dataProcess 
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

if __name__=="__main__":
    print(trans("程勇.中国强大.恐怖如斯"))
    path = r'/home/hl/cy/code/data/USA_CApro.json'
    data = dataProcess(path, K=5).jsonToDict()
    text = data[100]['text']+"."+data[20]['text']+'.'+data[2000]['text']+'.'
    with open('data/dataGeo.json', 'r') as f:
        print(f.readlines())
    print(text)
    print(type(text))
    print(trans(text))
