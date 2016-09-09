#! -*- coding:utf-8 -*-
 
'''
基于迁移学习和双向LSTM的核心实体识别
 
迁移学习体现在：
1、用训练语料和测试语料一起训练Word2Vec，使得词向量本捕捉了测试语料的语义；
2、用训练语料训练模型；
3、得到模型后，对测试语料预测，把预测结果跟训练语料一起训练新的模型；
4、用新的模型预测，模型效果会有一定提升；
5、对比两次预测结果，如果两次预测结果都一样，那说明这个预测结果很有可能是对的，用这部分“很有可能是对的”的测试结果来训练模型；
6、用更新的模型预测；
7、如果你愿意，可以继续重复第4、5、6步。
 
双向LSTM的思路：
1、分词；
2、转换为5tag标注问题（0:非核心实体，1:单词的核心实体，2:多词核心实体的首词，3:多词核心实体的中间部分，4:多词核心实体的末词）；
3、通过双向LSTM，直接对输入句子输出预测标注序列；
4、通过viterbi算法来获得标注结果；
5、因为常规的LSTM存在后面的词比前面的词更重要的弊端，因此用双向LSTM。
'''
 
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
import re
 
d = pd.read_json('data.json') #训练数据已经被预处理成为标准json格式
d.index = range(len(d)) #重新定义一下索引，当然这只是优化显示效果
word_size = 128 #词向量维度
maxlen = 80 #句子截断长度
 
'''
修改分词函数，主要是：
1、英文和数字部分不分词，直接返回；
2、双书名号里边的内容不分词；
3、双引号里边如果是十字以内的内容不分词；
4、超出范围内的字符全部替换为空格；
5、分词使用结巴分词，并关闭新词发现功能。
'''
 
not_cuts = re.compile(u'([\da-zA-Z \.]+)|《(.*?)》|“(.{1,10})”')
re_replace = re.compile(u'[^\u4e00-\u9fa50-9a-zA-Z《》\(\)（）“”·\.]')
def mycut(s):
    result = []
    j = 0
    s = re_replace.sub(' ', s)
    for i in not_cuts.finditer(s):
        result.extend(jieba.lcut(s[j:i.start()], HMM=False))
        if s[i.start()] in [u'《', u'“']:
            result.extend([s[i.start()], s[i.start()+1:i.end()-1], s[i.end()-1]])
        else:
            result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(jieba.lcut(s[j:], HMM=False))
    return result
 
d['words'] = d['content'].apply(mycut) #分词
 
def label(k): #将输出结果转换为标签序列
    s = d['words'][k]
    r = ['0']*len(s)
    for i in range(len(s)):
        for j in d['core_entity'][k]:
            if s[i] in j:
                r[i] = '1'
                break
    s = ''.join(r)
    r = [0]*len(s)
    for i in re.finditer('1+', s):
        if i.end() - i.start() > 1:
            r[i.start()] = 2
            r[i.end()-1] = 4
            for j in range(i.start()+1, i.end()-1):
                r[j] = 3
        else:
            r[i.start()] = 1
    return r
 
d['label'] = map(label, tqdm(iter(d.index))) #输出tags
 
#随机打乱数据
idx = range(len(d))
d.index = idx
np.random.shuffle(idx)
d = d.loc[idx]
d.index = range(len(d))
 
#读入测试数据并进行分词
dd = open('opendata_20w').read().decode('utf-8').split('\n')
dd = pd.DataFrame([dd]).T
dd.columns = ['content']
dd = dd[:-1]
print u'测试语料分词中......'
dd['words'] = dd['content'].apply(mycut)
 
 
'''
用gensim来训练Word2Vec：
1、联合训练语料和测试语料一起训练；
2、经过测试用skip gram效果会好些。
'''
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
word2vec = gensim.models.Word2Vec(dd['words'].append(d['words']), 
                                  min_count=1, 
                                  size=word_size, 
                                  workers=20,
                                  iter=20,
                                  window=8,
                                  negative=8,
                                  sg=1)
word2vec.save('word2vec_words_final.model')
word2vec.init_sims(replace=True) #预先归一化，使得词向量不受尺度影响
 
 
print u'正在进行第一次训练......'
 
'''
用最新版本的Keras训练模型，使用GPU加速（我的是GTX 960）
其中Bidirectional函数目前要在github版本才有
'''
from keras.layers import Dense, LSTM, Lambda, TimeDistributed, Input, Masking, Bidirectional
from keras.models import Model
from keras.utils import np_utils
from keras.regularizers import activity_l1 #通过L1正则项，使得输出更加稀疏
 
sequence = Input(shape=(maxlen, word_size))
mask = Masking(mask_value=0.)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(mask)
blstm = Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(blstm)
output = TimeDistributed(Dense(5, activation='softmax', activity_regularizer=activity_l1(0.01)))(blstm)
model = Model(input=sequence, output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
'''
gen_matrix实现从分词后的list来输出训练样本
gen_target实现将输出序列转换为one hot形式的目标
超过maxlen则截断，不足补0
'''
gen_matrix = lambda z: np.vstack((word2vec[z[:maxlen]], np.zeros((maxlen-len(z[:maxlen]), word_size))))
gen_target = lambda z: np_utils.to_categorical(np.array(z[:maxlen] + [0]*(maxlen-len(z[:maxlen]))), 5)
 
#从节省内存的角度，通过生成器的方式来训练
def data_generator(data, targets, batch_size): 
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)/batch_size+1)]
    while True:
        for i in batches:
            xx, yy = np.array(map(gen_matrix, data[i])), np.array(map(gen_target, targets[i]))
            yield (xx, yy)
 
batch_size = 1024
history = model.fit_generator(data_generator(d['words'], d['label'], batch_size), samples_per_epoch=len(d), nb_epoch=200)
model.save_weights('words_seq2seq_final_1.model')
 
#输出预测结果（原始数据，未整理）
def predict_data(data, batch_size):
    batches = [range(batch_size*i, min(len(data), batch_size*(i+1))) for i in range(len(data)/batch_size+1)]
    p = model.predict(np.array(map(gen_matrix, data[batches[0]])), verbose=1)
    for i in batches[1:]:
        print min(i), 'done.'
        p = np.vstack((p, model.predict(np.array(map(gen_matrix, data[i])), verbose=1)))
    return p
 
d['predict'] = list(predict_data(d['words'], batch_size))
dd['predict'] = list(predict_data(dd['words'], batch_size))
 
 
 
'''
动态规划部分：
1、zy是转移矩阵，用了对数概率；概率的数值是大概估计的，事实上，这个数值的精确意义不是很大。
2、viterbi是动态规划算法。
'''
zy = {'00':0.15, 
      '01':0.15, 
      '02':0.7, 
      '10':1.0, 
      '23':0.5, 
      '24':0.5,
      '33':0.5,
      '34':0.5, 
      '40':1.0
     }
 
zy = {i:np.log(zy[i]) for i in zy.keys()}
 
def viterbi(nodes):
    paths = nodes[0]
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]
 
 
'''
整理输出结果，即生成提交数据所需要的格式。
整个过程包括：动态规划、结果提取。
'''
 
def predict(i):
    nodes = [dict(zip(['0','1','2','3','4'], k)) for k in np.log(dd['predict'][i][:len(dd['words'][i])])]
    r = viterbi(nodes)
    result = []
    words = dd['words'][i]
    for j in re.finditer('2.*?4|1', r):
        result.append((''.join(words[j.start():j.end()]), np.mean([nodes[k][r[k]] for k in range(j.start(),j.end())])))
    if result:
        result = pd.DataFrame(result)
        return [result[0][result[1].argmax()]]
    else:
        return result
 
dd['core_entity'] = map(predict, tqdm(iter(dd.index), desc=u'第一次预测'))
 
 
'''
导出提交的JSON格式
'''
gen = lambda i:'[{"content": "'+dd.iloc[i]['content']+'", "core_entity": ["'+''.join(dd.iloc[i]['core_entity'])+'"]}]'
ssss = map(gen, tqdm(range(len(dd))))
result='\n'.join(ssss)
import codecs
f=codecs.open('result1.txt', 'w', encoding='utf-8')
f.write(result)
f.close()
import os
os.system('rm result1.zip')
os.system('zip result1.zip result1.txt')
 
 
print u'正在进行第一次迁移学习......'
 
'''
开始迁移学习。
'''
 
def label(k): #将输出结果转换为标签序列
    s = dd['words'][k]
    r = ['0']*len(s)
    for i in range(len(s)):
        for j in dd['core_entity'][k]:
            if s[i] in j:
                r[i] = '1'
                break
    s = ''.join(r)
    r = [0]*len(s)
    for i in re.finditer('1+', s):
        if i.end() - i.start() > 1:
            r[i.start()] = 2
            r[i.end()-1] = 4
            for j in range(i.start()+1, i.end()-1):
                r[j] = 3
        else:
            r[i.start()] = 1
    return r
 
dd['label'] = map(label, tqdm(iter(dd.index))) #输出tags
 
 
'''
将测试集和训练集一起放到模型中训练，
其中测试集的样本权重设置为1，训练集为10
'''
w = np.array([1]*len(dd) + [10]*len(d))
def data_generator(data, targets, batch_size): 
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)/batch_size+1)]
    while True:
        for i in batches:
            xx, yy = np.array(map(gen_matrix, data[i])), np.array(map(gen_target, targets[i]))
            yield (xx, yy, w[i])
 
history = model.fit_generator(data_generator(
                                    dd[['words']].append(d[['words']], ignore_index=True)['words'], 
                                    dd[['label']].append(d[['label']], ignore_index=True)['label'], 
                                    batch_size), 
                              samples_per_epoch=len(dd)+len(d), 
                              nb_epoch=20)
 
model.save_weights('words_seq2seq_final_2.model')
d['predict'] = list(predict_data(d['words'], batch_size))
dd['predict'] = list(predict_data(dd['words'], batch_size))
dd['core_entity_2'] = map(predict, tqdm(iter(dd.index), desc=u'第一次迁移学习预测'))
 
'''
导出提交的JSON格式
'''
gen = lambda i:'[{"content": "'+dd.iloc[i]['content']+'", "core_entity": ["'+''.join(dd.iloc[i]['core_entity_2'])+'"]}]'
ssss = map(gen, tqdm(range(len(dd))))
result='\n'.join(ssss)
import codecs
f=codecs.open('result2.txt', 'w', encoding='utf-8')
f.write(result)
f.close()
import os
os.system('rm result2.zip')
os.system('zip result2.zip result2.txt')
 
 
print u'正在进行第二次迁移学习......'
 
 
'''
开始迁移学习2。
'''
 
ddd = dd[dd['core_entity'] == dd['core_entity_2']].copy()
 
'''
将测试集和训练集一起放到模型中训练，
其中测试集的样本权重设置为1，训练集为5
'''
w = np.array([1]*len(ddd) + [5]*len(d))
def data_generator(data, targets, batch_size): 
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)/batch_size+1)]
    while True:
        for i in batches:
            xx, yy = np.array(map(gen_matrix, data[i])), np.array(map(gen_target, targets[i]))
            yield (xx, yy, w[i])
 
history = model.fit_generator(data_generator(
                                    ddd[['words']].append(d[['words']], ignore_index=True)['words'], 
                                    ddd[['label']].append(d[['label']], ignore_index=True)['label'], 
                                    batch_size), 
                              samples_per_epoch=len(ddd)+len(d), 
                              nb_epoch=20)
 
model.save_weights('words_seq2seq_final_3.model')
d['predict'] = list(predict_data(d['words'], batch_size))
dd['predict'] = list(predict_data(dd['words'], batch_size))
dd['core_entity_3'] = map(predict, tqdm(iter(dd.index), desc=u'第二次迁移学习预测'))
 
'''
导出提交的JSON格式
'''
gen = lambda i:'[{"content": "'+dd.iloc[i]['content']+'", "core_entity": ["'+''.join(dd.iloc[i]['core_entity_3'])+'"]}]'
ssss = map(gen, tqdm(range(len(dd))))
result='\n'.join(ssss)
import codecs
f=codecs.open('result3.txt', 'w', encoding='utf-8')
f.write(result)
f.close()
import os
os.system('rm result3.zip')
os.system('zip result3.zip result3.txt')
 
 
 
print u'正在进行第三次迁移学习......'
 
 
'''
开始迁移学习3。
'''
 
ddd = dd[dd['core_entity'] == dd['core_entity_2']].copy()
ddd = ddd[ddd['core_entity_3'] == ddd['core_entity_2']].copy()
 
'''
将测试集和训练集一起放到模型中训练，
其中测试集的样本权重设置为1，训练集为1
'''
w = np.array([1]*len(ddd) + [1]*len(d))
def data_generator(data, targets, batch_size): 
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)/batch_size+1)]
    while True:
        for i in batches:
            xx, yy = np.array(map(gen_matrix, data[i])), np.array(map(gen_target, targets[i]))
            yield (xx, yy, w[i])
 
history = model.fit_generator(data_generator(
                                    ddd[['words']].append(d[['words']], ignore_index=True)['words'], 
                                    ddd[['label']].append(d[['label']], ignore_index=True)['label'], 
                                    batch_size), 
                              samples_per_epoch=len(ddd)+len(d), 
                              nb_epoch=20)
 
model.save_weights('words_seq2seq_final_4.model')
d['predict'] = list(predict_data(d['words'], batch_size))
dd['predict'] = list(predict_data(dd['words'], batch_size))
dd['core_entity_4'] = map(predict, tqdm(iter(dd.index), desc=u'第三次迁移学习预测'))
 
'''
导出提交的JSON格式
'''
gen = lambda i:'[{"content": "'+dd.iloc[i]['content']+'", "core_entity": ["'+''.join(dd.iloc[i]['core_entity_4'])+'"]}]'
ssss = map(gen, tqdm(range(len(dd))))
result='\n'.join(ssss)
import codecs
f=codecs.open('result4.txt', 'w', encoding='utf-8')
f.write(result)
f.close()
import os
os.system('rm result4.zip')
os.system('zip result4.zip result4.txt')
 
 
 
print u'正在进行第四次迁移学习......'
 
 
'''
开始迁移学习4。
'''
 
ddd = dd[dd['core_entity'] == dd['core_entity_2']].copy()
ddd = ddd[ddd['core_entity_3'] == ddd['core_entity_2']].copy()
ddd = ddd[ddd['core_entity_4'] == ddd['core_entity_2']].copy()
 
'''
将测试集和训练集一起放到模型中训练，
其中测试集的样本权重设置为1，训练集为1
'''
w = np.array([1]*len(ddd) + [1]*len(d))
def data_generator(data, targets, batch_size): 
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)/batch_size+1)]
    while True:
        for i in batches:
            xx, yy = np.array(map(gen_matrix, data[i])), np.array(map(gen_target, targets[i]))
            yield (xx, yy, w[i])
 
history = model.fit_generator(data_generator(
                                    ddd[['words']].append(d[['words']], ignore_index=True)['words'], 
                                    ddd[['label']].append(d[['label']], ignore_index=True)['label'], 
                                    batch_size), 
                              samples_per_epoch=len(ddd)+len(d), 
                              nb_epoch=20)
 
model.save_weights('words_seq2seq_final_5.model')
d['predict'] = list(predict_data(d['words'], batch_size))
dd['predict'] = list(predict_data(dd['words'], batch_size))
dd['core_entity_5'] = map(predict, tqdm(iter(dd.index), desc=u'第四次迁移学习预测'))
 
'''
导出提交的JSON格式
'''
gen = lambda i:'[{"content": "'+dd.iloc[i]['content']+'", "core_entity": ["'+''.join(dd.iloc[i]['core_entity_5'])+'"]}]'
ssss = map(gen, tqdm(range(len(dd))))
result='\n'.join(ssss)
import codecs
f=codecs.open('result5.txt', 'w', encoding='utf-8')
f.write(result)
f.close()
import os
os.system('rm result5.zip')
os.system('zip result5.zip result5.txt')
