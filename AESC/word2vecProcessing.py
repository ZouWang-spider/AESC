# -*- coding: utf-8 -*-
import nltk
from ABSAli1 import preProcessing
from ABSAli1 import entity
import logging
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
import pickle
import pandas

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

#词嵌入，并将两种词向量作为位置特征字典
#训练词向量
def trainingForWords(texts,fnum,filepath,sg,hs):
    #设定词向量参数

    sentences=[str(text).split(' ') for text in texts]  #.lower()
    # print(sentences)
    print('分词完成，开始训练')
    
    num_features=fnum #词向量的维度
    min_word_count=3 #词频数最低阈值
    num_workers=8 #线程数,想要随机种子生效的话，设为1
    context=10 #上下文窗口大小
    downsampling=1e-3 #与自适应学习率有关
    num_iter=15 #没有加额外数据时，设置为100最佳(添加额外数据后30最佳)
    hs=hs
    sg=sg#是否使用skip-gram模型
    
    model_path=filepath
    model=Word2Vec(sentences,workers=num_workers,hs=hs,
                   size=num_features,min_count=min_word_count,seed=77,iter=num_iter,
                   window=context,sample=downsampling,sg=sg)
    model.init_sims(replace=True)#锁定训练好的word2vec,之后不能对其进行更新 
    model.save(model_path)#讲训练好的模型保存到文件中
    print('训练完成')
    return model
    
#加载word2vec模型
def loadForWord(filepath):
    model=Word2Vec.load(filepath)
    print('word2vec模型读取完毕')
    return model
    
def trainingEmbedding(vector_len=150,d_type='re',add_extra=False):
    if d_type=='re':
        d_name='Restaurants'
        extraFile='data/extra/yelp/Restaurants_Raw.csv'
    else:
        d_name='LapTops'
        extraFile='data/extra/amzon/LapTops_Raw.csv'
    
    print('------训练%s数据的Word2Vec------'%d_name)
    train_corpus=preProcessing.loadXML('data/origin/%s_Train_v2.xml'%d_name)                #判断训练集合法
    test_corpus=preProcessing.loadXML('data/origin/%s_Test_Data_PhaseA.xml'%d_name)         #判断测试集合法
    print('数据集合并完成')
    
    corpus=train_corpus.corpus
    corpus=train_corpus.corpus+test_corpus.corpus

    
    del train_corpus
    del test_corpus
    
    bio_entity=entity.BIO_Entity(corpus,d_type)    
    texts=bio_entity.texts


    #向SemEval-2014文件中的'Restaurants'和'LapTops'添加额外的数据
    if add_extra==True:
        print('添加额外语料:%s'%extraFile)
        extra_csv=pandas.read_csv(extraFile)  #,encoding='gbk'
        extra_texts=list(extra_csv['text'])
        texts=texts+extra_texts
        del extra_csv
        del extra_texts
        print('额外语料加载完成')
    
    print('创建WordEmbedding')
    trainingForWords(texts,vector_len,'model/%s.w2v'%d_name,1,0)        #Word2Vec中采用Skipgram
    print('创建WordEmbedding_CBOW')
    trainingForWords(texts,vector_len,'model/%s.w2v_cbow'%d_name,0,0)  #Word2Vec中采用CBOW
#采用skipgram模型和cbow模型，之所以训练两组是为了丰富后面的特征。


#为什么采用Keman处理
#在后面要进行特征抽取任务中，我们当然不能直接把单词的词向量丢到CRF里去学习（这可不是LSTM这种NN模型，要CRF学习词向量的每一维的转换关系，消耗上是不能接受的，且没有意义），
#这里合理做法是我之前提到的TOP选手的做法，将全体单词词向量做Kmeans聚类，每一个单词都会得到一个聚类类别，反映了其大体语义，我们将单词的类别作为CRF的位置特征。



def kmeansClusterForW2V(filepath,outpath,cluster_num):
    W2Vmodel=loadForWord(filepath)
    vocab=list(W2Vmodel.wv.vocab.keys())
    vectors=[W2Vmodel[vocab[i]] for i in (range(len(vocab)))]
    print('开始聚类')
    clf=KMeans(n_clusters=cluster_num,random_state=77)
    clf.fit(vectors)
    print('聚类完成，开始讲词典转化为类别字典')
    dict_re={vocab[i]:clf.labels_[i] for i in range(len(vocab))}
    print('保存字典。。。。')
    with open(outpath,'wb') as f:
        pickle.dump(dict_re,f)
    return dict_re
    
def loadDict(dictpath):
    print('载入字典：%s'%dictpath)
    with open(dictpath,'rb') as f:
        dict_re=pickle.load(f)
    return dict_re

#对词向量进行聚类处理，    位置特征字典序列  簇的类别为10
def createCluster(cluster_num=10,d_type='re'):
    if d_type=='re':
        d_name='Restaurants'
    else:
        d_name='LapTops'
            
    print('创建W2V聚类')
    kmeansClusterForW2V('model/%s.w2v'%d_name,'cluster/%s_w2v.pkl'%d_type,cluster_num)
    print('W2V聚类群创建完毕！')
    print('创建W2V_CBOW聚类')
    kmeansClusterForW2V('model/%s.w2v_cbow'%d_name,'cluster/%s_w2v_c.pkl'%d_type,cluster_num)
    print('W2V聚类群创建完毕！')
    

if __name__=='__main__':
     print('word2vecProcessing')
     # trainingEmbedding(300,'re')
     # createCluster(100,'re',  True,True,True)

     # trainingEmbedding(300,'lp',True,   True,True,True)
     # createCluster(200,'lp',  True,True,True)

     # W2V=loadForWord('embeddingModels/Laptops.w2v')
    
