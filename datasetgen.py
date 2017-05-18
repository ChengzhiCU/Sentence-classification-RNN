import cPickle
import gzip
import os
import sys
import copy

import gc
import numpy
from numpy import genfromtxt
import numpy as np
import csv
import gzip,cPickle

def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

total_len = 231 #1014
savesize=2000
#total_len = 10
csvFile_test = "/home/srthu2/merveille/HWFinal/data/ag_news_csv/test.csv"
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

test_y = []
raw_test_x = []
test_x = []    
test_cnt=0
ns=1
with open(csvFile_test,'rb') as cf:
    spamreader = csv.reader(cf,delimiter = ',')
    for row in spamreader:
        #print ','.join(row)
        test_y.append(int(ord(row[0]) - ord('0'))-1)
        oneline = ','.join(row[1:])        
        line_index = []
        for each in oneline:
            if each.lower() in alphabet:
                line_index.append(alphabet.index(each.lower())+1)   #start from 1
            else:
                line_index.append(0)
        length = len(line_index)
        sentence=[]
        sentence = np.asarray(sentence)
        if length<=total_len:
            for each in line_index:
                temp = np.zeros([70],dtype='float32')
                if each>0:
                    temp[each-1]=1                
                sentence = np.concatenate([sentence,temp],axis=0)
            sentence = np.concatenate([sentence,np.zeros([(total_len - length)*70],dtype='float32')],axis=0)
        else:
            for each in line_index[0:total_len]:
                temp = np.zeros([70],dtype='float32')
                if each>0:
                    temp[each-1]=1
                sentence = np.concatenate([sentence,temp],axis=0)

        test_x.append(sentence)
        test_cnt += 1
        
        if test_cnt%savesize==0:            
            test_x = np.asarray(test_x).reshape((1,test_cnt,1,total_len,70))
            test_cnt = 0
            test_y = one_hot(test_y,4)            
            gc.collect()
            test_set=(test_x,test_y)
            testsave = np.asarray(test_set)
            np.save('./data/AgNew_onehot_Test_n'+str(ns), testsave)
            testsave=[]
            gc.collect()
            ns += 1
            test_y = []
            raw_test_x = []
            test_x = []
#test_y = np.asarray(test_y)
#print test_y[0:100]
test_x = np.asarray(test_x).reshape((1,test_cnt,1,total_len,70))#note that I find
# if the first 1 in reshape is not added, then the array will be saved as test_cnt different array, which impede our future work
test_y = one_hot(test_y,4)
print 'test_y',test_y
#print type(test_y),test_y.shape
gc.collect()
print 'shape_test',test_x.shape
#print test_cnt
test_set=(test_x,test_y)
#print test_set
testsave = np.asarray(test_set)
#testsave = test_set
#print testsave.shape
#print '0shape',testsave[0].shape
#print testsave[0]
print testsave[1].shape
np.save('./data/AgNew_onehot_Test_n'+str(ns), testsave)
testsave=[]
gc.collect()

csvFile_train = "/home/srthu2/merveille/HWFinal/data/ag_news_csv/train.csv"
train_y = []
train_x = []
cnt=0
namestr = 0
with open(csvFile_train,'rb') as cf:
    spamreader = csv.reader(cf,delimiter = ',')
    for row in spamreader:
        #print ','.join(row)
        train_y.append(int(ord(row[0]) - ord('0'))-1)
        oneline = ','.join(row[1:])        
        line_index = []
        for each in oneline:
            if each.lower() in alphabet:
                line_index.append(alphabet.index(each.lower())+1)   #start from 1
            else:
                line_index.append(0)
        length = len(line_index)
        sentence=[]
        sentence = np.asarray(sentence)
        if length<=total_len:
            for each in line_index:
                temp = np.zeros([70],dtype='float32')
                if each>0:
                    temp[each-1]=1                
                sentence = np.concatenate([sentence,temp],axis=0)
            sentence = np.concatenate([sentence,np.zeros([(total_len - length)*70],dtype='float32')],axis=0)
        else:
            for each in line_index[0:total_len]:
                temp = np.zeros([70],dtype='float32')
                if each>0:
                    temp[each-1]=1
                sentence = np.concatenate([sentence,temp],axis=0)

        train_x.append(sentence);
        cnt = cnt + 1
        if cnt % savesize == 0:
            namestr += 1
            print cnt,'finished'
            #break
            train_x = np.asarray(train_x).reshape((1,cnt,1,total_len,70))
            train_y = one_hot(train_y,4)
            #print 'trainxshape=',train_x.shape
            train_set=(train_x,train_y)
            train_x=[]
            train_y = []
            gc.collect()
            #train_set = np.asarray(train_set)
            np.save('./data/AgNew_onehot_train_n'+str(namestr), np.asarray(train_set))      
            cnt = 0
            gc.collect()
            '''
            if namestr==6:
                break
            '''
            
            

'''
train_x = np.asarray(train_x)
train_y = one_hot(train_y,4)
print 'trainxshape=',train_x.shape
train_set=(train_x.reshape((cnt-1,1,total_len,70)),train_y)
#print train_set

rval=[train_set,test_set]
rvalsave = np.asarray(rval)
np.save('AgNew_onehotbedding', rvalsave)
'''
