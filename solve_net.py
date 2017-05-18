from utils import LOG_INFO
import numpy as np
#import matplotlib.pyplot as plt

def data_iterator(x, y, batch_size, shuffle=True):
    indx = range(len(x))
    if shuffle:
        np.random.shuffle(indx)
    dx = []
    dy = []
    for i in indx:
        dx.append(x[i])
        dy.append(y[i])

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[start_idx: end_idx], y[start_idx: end_idx]

def solve_net(model, optim, train_x, train_y, test_x, test_y,
              batch_size, max_epoch, disp_freq, test_freq, half_step=10, network_name=''):

    iter_counter = 0
    loss_list = []
    accuracy_list = []
    test_acc = []
    test_loss = []
    si=test_x.shape
    if len(si)==5:
        test_x = test_x.reshape((si[1:]))

    tsi = train_x.shape
    if len(tsi) == 3:
        train_x = train_x.reshape((tsi[0],1,tsi[1],tsi[2]))

    print 'train_label_size=',train_y.shape
    print 'train_data',train_x.shape
    print 'test_label=',test_y.shape
    print 'test_data',test_x.shape

    all_train_loss=[]
    all_train_accuracy=[]
    all_test_loss=[]
    all_test_accuracy=[]

    for k in range(max_epoch):
        
        if (k+1)%half_step == 0:
            lr = model.get_lr(optim)
            lr /= 2
            print "current lr=",lr
            model.set_lr(lr, optim)
        
        for x, y in data_iterator(train_x, train_y, batch_size,True):

            iter_counter += 1
            loss, accuracy = model.train(x, y)

            #print model.params[0].get_value()[0:10,0,0,0]
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            all_train_loss.append(loss)
            all_train_accuracy.append(accuracy)
            

            if iter_counter % disp_freq == 0:
                msg = 'Training iter %d, mean loss %.5f (batch loss %.5f), mean acc %.5f' % (iter_counter,
                                                                                             np.mean(loss_list),
                                                                                             loss_list[-1],
                                                                                             np.mean(accuracy_list))
                
                #a,b= model.debuger(x)
                #print 'debug a',a
                #print 'debug b7',b

                LOG_INFO(msg)
                loss_list = []
                accuracy_list = []

            if iter_counter % test_freq == 0:
                LOG_INFO('    Testing...')
                for tx, ty in data_iterator(test_x, test_y, batch_size, shuffle=False):
                    t_accuracy, t_loss = model.test(tx, ty)
                    test_acc.append(t_accuracy)
                    test_loss.append(t_loss)

                msg = '    Testing iter %d, mean loss %.5f, mean acc %.5f' % (iter_counter,
                                                                              np.mean(test_loss),
                                                                              np.mean(test_acc))
                
                all_test_loss.append(np.mean(test_loss))
                all_test_accuracy.append(np.mean(test_acc))
                LOG_INFO(msg)
                test_acc = []
                test_loss = []


    ss='./log/log_Glove' + network_name
    print ss
    f = open(ss + 'trainloss.txt','w')
    for i in xrange(len(all_train_loss)):
        f.write("%f " % all_train_loss[i])
    f.close()

    f = open(ss + 'trainacc.txt','w')
    for i in xrange(len(all_train_accuracy)):
        f.write("%f " % all_train_accuracy[i])
    f.close()

    f = open(ss + 'testloss.txt','w')
    for i in xrange(len(all_test_loss)):
        f.write("%f " % all_test_loss[i])
    f.close()

    f = open(ss + 'testacc.txt','w')
    for i in xrange(len(all_test_accuracy)):
        f.write("%f " % all_test_accuracy[i])
    f.close()