import gc
import numpy as np


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x-1] = 1
    return o_h


emdLength=50
vec = []
word2id = {}
f = open('./data/glove50d.txt')
while True:
	content = f.readline()
	if content == '':
		break
	content = content.strip().split()
	word2id[content[0]] = len(word2id)
	# vector 50
	content = content[1:]
	content = [(float)(i) for i in content]
	vec.append(content)

f.close()
# create dictionary
# word ->  index
word2id['BLANK'] = len(word2id)
temp = [0.0 for i in range(len(vec[0]))] # 50 * 0
vec.append(temp)
vec = np.array(vec,dtype=np.float32)
# vec[end] = zeros

test_y=[]
test_x=[]
total_len=30###################################
print 'dictionary built up'

f = open('./../data/ag_news_csv/test.csv')
count=0
while True:
	content = f.readline()

	if content == '':
		break
	content = content.replace('"',' ')
	content = content.lower()
	content = content.replace("'s"," 's")
	content = content.replace("-"," - ")
	content = content.replace("("," ( ")
	content = content.replace(")"," ) ")
	content = content.replace(","," ,")
	content = content.replace(";"," ;")
	content = content.replace('"',' " ')
	content = content.replace("$","$ ")
	content = content.replace("%"," %")
	content = content.replace("'m"," 'm")
	content = content.replace("'ve"," 've")
	content = content.replace("'re"," 're")
	content = content.replace("n't"," n't")
	content = content.replace(":"," :")
	content = content.replace("!"," ! ")
	content = content.replace("?"," ? ")	
	content = content.replace("'d"," 'd")
	
	content = content.strip().split()
	test_y.append(int(ord(content[0])-ord('0')))
	
	content = content[2:]
	sentence = []
	sentence = np.asarray(sentence)
	test_x = np.asarray(test_x)
	for i in range(len(content)):
		
		if not content[i] in word2id:
			#sentence.append(vec[word2id['BLANK']])
			sentence = np.concatenate([sentence,vec[word2id['BLANK']]],axis=0)
		else:
			sentence = np.concatenate([sentence,vec[word2id[content[i]]]], axis=0)
		if i+1==total_len:
			break
	# sentence is not long enough
	if (len(content) < total_len):
		#print 'need padding'
		while len(sentence) < emdLength * total_len:
			sentence = np.concatenate([sentence,sentence[0:min(len(content),(total_len - len(content)))*emdLength]], axis=0)
		if len(sentence) != emdLength * total_len:
			print 'error'
	# using cycle sentence to padding
	if len(sentence) != emdLength * total_len:
		print 'error',len(sentence)
	test_x = np.concatenate([test_x,sentence],axis = 0)
	count += 1
	#if count>23:
	#	break
	
	
f.close()
# test x I use list bebore, in datagen, succeed, but here not knowing why I have to use np array
#print test_x[0][0].shape
test_x = test_x.reshape((1,count,1,total_len,50))
#print test_x
#print 'test_x',test_x[0].shape
#print 'test_y',one_hot(test_y,4) 
test_y = np.asarray(test_y)

test_set=(test_x,test_y)
testsave = np.asarray(test_set)
np.save('./data/AgNew_Glove_Test_n', testsave)
testsave=[]
gc.collect()

print 'test done'

train_y=[]
train_x=[]
total_len=30###################################
ns=1
f = open('./../data/ag_news_csv/train.csv')
cnttrain=0
cnt=0
while True:
	content = f.readline()
	cnt += 1
	if content == '':
		break
	content = content.replace('"',' ')
	content = content.lower()
	content = content.replace("'s"," 's")
	content = content.replace("-"," - ")
	content = content.replace("("," ( ")
	content = content.replace(")"," ) ")
	content = content.replace(","," ,")
	content = content.replace(";"," ;")
	content = content.replace('"',' " ')
	content = content.replace("$","$ ")
	content = content.replace("%"," %")
	content = content.replace("'m"," 'm")
	content = content.replace("'ve"," 've")
	content = content.replace("'re"," 're")
	content = content.replace("n't"," n't")
	content = content.replace(":"," :")
	content = content.replace("!"," ! ")
	content = content.replace("?"," ? ")	
	content = content.replace("'d"," 'd")

	content = content.strip().split()	
	train_y.append(int(ord(content[0])-ord('0')))	
	content = content[2:]
	sentence = []
	sentence = np.asarray(sentence)
	train_x = np.asarray(train_x)
	for i in range(len(content)):	

		if not content[i] in word2id:
			#sentence.append(vec[word2id['BLANK']])
			sentence = np.concatenate([sentence,vec[word2id['BLANK']]],axis=0)
		else:
			sentence = np.concatenate([sentence,vec[word2id[content[i]]]], axis=0)
		if i+1==total_len:
			break
	# sentence is not long enough
	if (len(content) < total_len):
		#print 'need padding'
		while len(sentence) < emdLength * total_len:
			sentence = np.concatenate([sentence,sentence[0:min(len(content),(total_len - len(content)))*emdLength]], axis=0)
		if len(sentence) != emdLength * total_len:
			print 'error'
	# using cycle sentence to padding
	if len(sentence) != emdLength * total_len:
		print 'error',len(sentence)
	train_x = np.concatenate([train_x,sentence],axis = 0)
	#print 'once',cnt
	if cnt == 10000:
		print cnt,'finished'
		train_x = train_x.reshape((1,cnt,total_len,emdLength))
		train_y = one_hot(train_y,4)
		train_set = (train_x,train_y)
		train_x=[]
		train_y=[]
		#gc.collect()
		np.save('./data/AgNew_Glove_train_n'+str(ns), np.asarray(train_set))      
		#print 'save successfully'
		cnt = 0
		#gc.collect()
		



	'''
	content = content.strip().split()
	train_y.append(content[0])
	content = content[2:]
	sentence = []
	for i in range(len(content)):
		if not content[i] in word2id:
			sentence.append(vec[word2id['BLANK']])
		else:
			sentence.append(vec[word2id[content[i]]])

	if (len(content)<=total_len):
		for i in range(total_len-len(content)) :
			sentence.append(sentence[i])
	else:
		temp=sentence[0:total_len]

	train_x.append(sentence);
	'''
print 'print cnt',cnt

f.close()
train_y = np.asarray(test_y)
train_set=(test_x,test_y)
		




