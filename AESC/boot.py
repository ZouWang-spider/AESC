# -*- coding: utf-8 -*-

from ABSAli1.preProcessing import transformJSONFiles,createAllDependence,loadDependenceInformation

from ABSAli1.word2vecProcessing import trainingEmbedding, createCluster
from ABSAli1.exam_CRF import evaluate
from ABSAli1.contextProcessing import createAllForPol
from ABSAli1.polClassification_ML import examByML

if __name__ == '__main__':
    print('#### Step1. Preprocessing')                           #preProcessing.py
    # 对原始数据做预处理   jsion---->CSV
    # transformJSONFiles(d_type='re')    #生成Restaurants_Raw.csv文件+Restaurants_Train_v2.xml
    # transformJSONFiles(d_type='lp')    #生成LapTops_Raw.csv文件+LapTops_Train_v2.xml
    # createAllDependence()              #对文本进行依存句法分析，并保存结果
    # depend_list=loadDependenceInformation('dependences/re_train.dep')


    print('#### Step1. Word2Vec and Clutering')                  #word2vecProcessing.py
    # 训练词向量与聚类
    # trainingEmbedding(300,'re',True)   #采用CBOW和Skipgram两种词嵌入方式进行词嵌入  restaurant  300为词向量的长度
    # createCluster(100,'re')            #将词向量进行簇聚类来构造位置特征字典   100为簇的数量
    # trainingEmbedding(300,'lp',True)   #Laptop
    # createCluster(200,'lp')
    
    print('#### Step3 evaluate for Aspect Term Extraction')       #exam_CRF.py
    # 采用CRF进行Aspect Extraction任务
    #all_terms,all_offsets,origin_text_test,true_offsets=evaluate(False,'re')       #restauants
    # all_terms2,all_offsets2,origin_text_test2,true_offsets2=evaluate(False,'lp')   #laptops
    # all_terms（预测的所有测试文本对应的情感对象）
    # all_offsets（预测情感对象在文本中的位置）
    # origin_text_test（原始测试文本集）
    # true_offsets（测试文本真实的情感对象位置）
    # for i in range(50,70):
    #   print('电脑数据集中id=103的文本为:',origin_text_test2[i])
    #   print('抽取的方面术语为：',all_terms2[i])
    #   print('测试的方面术语的在文本中位置:',all_offsets2[i])
    #   print('真实方面术语位置:',true_offsets2[i])
    #print('方面术语的情感极性为：{}：{},{}：{}'.format('design','Positive', 'aluminum casing','Positive'))


    
    print('#### Step4 context Processing')                        #contextProcessing.py
    #抽取文本方面术语的上下文类似于LCFS,相关上下文的构建
    # createAllForPol(d_type='re',context=5)        #提取方面词的上下文信息来判断方面词的情感倾向
    # createAllForPol(d_type='lp',context=5)
    
    print('#### Step5 evaluate for Aspect Term Classification')    #polClassification_ML.py
    #使用机器学习算法实现方面术语的上下文的情感极性分类
    examByML('re','SVM',0.8)
    # examByML('lp','SVM',0.8)    #可以采用