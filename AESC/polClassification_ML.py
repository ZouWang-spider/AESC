# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ABSAli1 import contextProcessing
from ABSAli1 import word2vecProcessing
from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from keras.layers import *
from keras.models import Sequential,Model
from keras.layers import LSTM
from keras.utils import to_categorical,np_utils
from keras.models import load_model



def getFeaturesFromContext(aspectContext,W2V):
    w2v_feature=[]
    for word in aspectContext.context.split(' '):
        try:
            w2v_feature.append(W2V[word.lower()])
        except:
            w2v_feature.append([0 for i in range(W2V.vector_size)])
            #print('not find :%s'%word.lower())
    w2v_feature=np.array(w2v_feature).mean(axis=0)
    
    dep_feature=[]
    for dep in aspectContext.dep_context:
        for word in dep:
            try:
                dep_feature.append(W2V[word.lower()])
            except:
                dep_feature.append([0 for i in range(W2V.vector_size)])
                #print('not find :%s'%word.lower())
    dep_feature=np.array(dep_feature).mean(axis=0)
    
    return np.concatenate((w2v_feature,dep_feature)).tolist()
    
def getInfoFromList(aspectContextList,W2V):
    features=[getFeaturesFromContext(ac,W2V) for ac in aspectContextList]
    pols=[ac.pol for ac in aspectContextList]
    return features,pols
    
def getFeaturesAndPolsFromFile(filepath,d_type='re',per=0.8):
    if d_type=='re':
        d_name='Restaurants'
    else:
        d_name='LapTops'
    train_data,test_data=contextProcessing.splitContextFile(filepath,per)
    print('获取特征与情感分类中。。。。。')
    W2V=word2vecProcessing.loadForWord('model/%s.w2v'%d_name)
    
    trainX,trainY=getInfoFromList(train_data,W2V)
    testX,testY=getInfoFromList(test_data,W2V)
    print('获取完成')
    
    trainX=np.array(trainX)
    trainY=np.array(trainY)    #需要将标签进行处理
    trainY = np.asarray(pd.get_dummies(trainY))
    # trainY = trainY.argmax(axis=1)   #one-hot---->[1,2,3,3..]
    # print(label)

    testX=np.array(testX)
    testY=np.array(testY)
    testY = np.asarray(pd.get_dummies(testY))
    # testY = testY.argmax(axis=1)
    
    return trainX,trainY,testX,testY

# from  sklearn.ensemble import RandomForestClassifier
# #训练
# def trainClassifier(trainX,trainY,classifier='SVM'):
#     if classifier=='SVM':
#         print('使用SVM进行情感分类器训练')
#         clf=LinearSVC()
#         # clf=RandomForestClassifier(n_estimators=700,criterion='gini')
#         clf=clf.fit(trainX,trainY)
#         print('训练完成')
#     else:
#         print('使用逻辑回归LR进行情感分类训练')
#         lr=LogisticRegression()
#         lr=lr.fit(trainX,trainY)#用训练数据来拟合模型
#         print('训练完成')
#         clf=lr
#
#     return clf
#
# #预测
# def predict(testX,testY,clf):
#     print('开始预测')
#     true_result=clf.predict(testX)
#     pre_result=testY
#
#     print('分类报告: \n')
#     print(classification_report(true_result, pre_result,digits=4))
#
#     clf.score(testX,testY)


def trainmodel(trainX,trainY):
     print('使用BiLSTM进行情感分类器训练')
     input=Input(shape=(600,))      #(None, 600)   #向量长度600
     # print(input.shape)
     embed1 = Embedding(1000+ 1, 32)(input)  #(None, 600, 32)     #词表大小，我们需要寻找到词表大小
     # print(embed1.shape)        #
     bilstm=LSTM(32)(embed1)
     drop=Dropout(0.2)(bilstm)
     dense2 = Dense(256)(drop)
     drop2=Dropout(0.2)(dense2)
     dense3=Dense(32)(drop2)
     dense3=Dense(4,activation='sigmoid')(dense3)
     model=Model(inputs=input,outputs=dense3)
     model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
     lstm=model.fit(trainX,trainY,batch_size=28,epochs=10,verbose=1)
     print('训练完成')
     model.save('nn.h5')
     return lstm



from sklearn.metrics import precision_score, accuracy_score
    
def examByML(d_type='re',per=0.8):
    if d_type=='re':
        filepath='contextFiles/re_train.cox'
    else:
        filepath='contextFiles/lp_train.cox'
    # print(trainX.shape)  #(2952,600)   (trainY)  (2952,)
    trainX,trainY,testX,testY=getFeaturesAndPolsFromFile(filepath,d_type,per)   #获取特征和情感极性  y需要转换
    crf=trainmodel(trainX, trainY)

    # crf=load_model('nn.h5')
    # predict=crf.predict(testX)
    # print(predict)
    # #计算准确率
    # precision = accuracy_score(testY, predict)
    # print('准确率：')
    # print(precision)

if __name__=='__main__':
    examByML('re',0.8)   #0.8数据划分比例
    
    
    
    
        