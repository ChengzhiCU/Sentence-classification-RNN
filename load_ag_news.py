import numpy
import numpy as np
import csv

import theano
import theano.tensor as T

def load_ag_Glove_embed(group=24):    
    tpath = './../yuan/data/AgNew_Glove_Test_n'
    for i in xrange(1):
        testpath = tpath + '.npy'
        infile = open(testpath,'rb')
        test = np.load(infile)
        if i==0:
            test_data = test[0][0]
            test_label = test[1]
        else:
            test_data = np.concatenate([test_data,test[0][0]],axis=0)
            test_label = np.concatenate([test_label,test[1]],axis=0)
        print 'test loading finished'

    test_set = (test_data,test_label)
    
    for i in xrange(group):
        path = './../yuan/data/AgNew_Glove_train_n' +str(i+1)+'.npy'
        
        infile = open(path, 'rb')
        train_set = np.load(infile)
        if i==0:
            train_data = train_set[0][0]
            train_label = train_set[1]
        else:
            train_data = np.concatenate([train_data,train_set[0][0]],axis = 0)
            train_label = np.concatenate([train_label,train_set[1]],axis = 0)
        print '5000finished'
    train = (train_data,train_label)
    return (train,test)
    

def load_ag_onehotembed():
    
    for i in xrange(5):
        path = './../yuan/data/AgNew_onehot_train_n' +str(i+1)+'.npy'
        infile = open(path, 'rb')
        train_set = np.load(infile)
        #print 'train_set.shape',train_set.shape
        if i==0:
            train_data = train_set[0][0]
            train_label = train_set[1]
            
        else:
            train_data = np.concatenate([train_data,train_set[0][0]],axis = 0)
            train_label = np.concatenate([train_label,train_set[1]],axis = 0)
        print '2000finished'
    train = (train_data,train_label)
    
    for i in xrange(4):
        path = './../yuan/data/AgNew_onehot_Test_n' +str(i+1)+'.npy'
        infile = open(path, 'rb')
        test_set = np.load(infile)
        
        if i==0:
            test_data = test_set[0][0]
            test_label = test_set[1]
            
        else:
            test_data = np.concatenate([test_data,test_set[0][0]],axis = 0)
            test_label = np.concatenate([test_label,test_set[1]],axis = 0)
        print '2000finished'
    test = (test_data,test_label)    
    return (train,test)
    
def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

