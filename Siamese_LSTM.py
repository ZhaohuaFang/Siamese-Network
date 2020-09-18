import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import nltk
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential
import numpy as np
import pandas as pd
import random as rnd
nltk.download('punkt')
# set random seeds
rnd.seed(34)

data = pd.read_csv('/content/drive/My Drive/Siamese Network/questions.csv')
N=len(data)

N_train = 300000
N_test  = 10*1024
#data = pd.read_csv("questions.csv")
data_train = data[:N_train]
data_test  = data[N_train:N_train+N_test]
print("Train set:", len(data_train), "Test set:", len(data_test))
del(data) # remove to free memory

#为何这样就保证同一组的句子不是duplicate
#[False False False ... False  True False]
td_index = (data_train['is_duplicate'] == 1).to_numpy()
#[5, 7, 11, 12, 13, 15, 16, 18, 20, 29....]
td_index = [i for i, x in enumerate(td_index) if x] 
print('number of duplicate questions: ', len(td_index))
print('indexes of first ten duplicate questions:', td_index[:10])

#随便输出一行数据，它们不是duplicate
print(data_train['question1'][4]) 
print(data_train['question2'][4])
print('is_duplicate: ', data_train['is_duplicate'][4])

#使训练用的数据，本身已经是duplicate
Q1_train_words = np.array(data_train['question1'][td_index])
Q2_train_words = np.array(data_train['question2'][td_index])

Q1_test_words = np.array(data_test['question1'])
Q2_test_words = np.array(data_test['question2'])
y_test  = np.array(data_test['is_duplicate'])

#create arrays
#array([None, None, None, ..., None, None, None], dtype=object)
Q1_train = np.empty_like(Q1_train_words)
Q2_train = np.empty_like(Q2_train_words)

Q1_test = np.empty_like(Q1_test_words)
Q2_test = np.empty_like(Q2_test_words)

# Building the vocabulary with the train set         (this might take a minute)
from collections import defaultdict
#遇到不属于字典的键时，返回0，不报错
vocab = defaultdict(lambda: 0)
#vocab = dict()
#vocab['<PAD>'] = 1

for idx in range(len(Q1_train_words)):
    Q1_train[idx] = nltk.word_tokenize(Q1_train_words[idx])
    Q2_train[idx] = nltk.word_tokenize(Q2_train_words[idx])
    q = Q1_train[idx] + Q2_train[idx]
    for word in q:
        if word not in vocab:
            vocab[word] = len(vocab) + 1
            
print('The length of the vocabulary is: ', len(vocab))

for idx in range(len(Q1_test_words)): 
    Q1_test[idx] = nltk.word_tokenize(Q1_test_words[idx])
    Q2_test[idx] = nltk.word_tokenize(Q2_test_words[idx])

# Converting questions to array of integers
for i in range(len(Q1_train)):
    Q1_train[i] = [vocab[word] for word in Q1_train[i]]
    Q2_train[i] = [vocab[word] for word in Q2_train[i]]

        
for i in range(len(Q1_test)):
    Q1_test[i] = [vocab[word] for word in Q1_test[i]]
    Q2_test[i] = [vocab[word] for word in Q2_test[i]]

max_review_len=0
for i in range(len(Q1_train)):
    m=len(Q1_train[i])
    n=len(Q2_train[i])
    if m>n & m>max_review_len:
        max_review_len=m
    else:
        if n>max_review_len:
            max_review_len=n
for i in range(len(Q1_test)):
    m=len(Q1_test[i])
    n=len(Q2_test[i])
    if m>n & m>max_review_len:
        max_review_len=m
    else:
        if n>max_review_len:
            max_review_len=n
print(max_review_len)

batchsz = 128
total_words = 40000
embedding_len = 128
Q1_train = keras.preprocessing.sequence.pad_sequences(Q1_train, maxlen=max_review_len)
Q2_train = keras.preprocessing.sequence.pad_sequences(Q2_train, maxlen=max_review_len)

Q1_test = keras.preprocessing.sequence.pad_sequences(Q1_test, maxlen=max_review_len)
Q2_test = keras.preprocessing.sequence.pad_sequences(Q2_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((Q1_train, Q2_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)

db_test = tf.data.Dataset.from_tensor_slices((Q1_test, Q2_test))
db_test = db_test.batch(batchsz, drop_remainder=True)

net_embedding = Sequential([layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len) ])
net_LSTM = Sequential([layers.LSTM(64, dropout=0.5, return_sequences=True, unroll=True),
            
                       layers.LSTM(64, dropout=0.5, unroll=True)
                     ])
optimizer = optimizers.Adam(lr=1e-3)

net_embedding.build(input_shape=[None, max_review_len])
net_LSTM.build(input_shape=[None, max_review_len,embedding_len])
variables = net_embedding.trainable_variables  + net_LSTM.trainable_variables 

def TripletLossFn(v1, v2, margin=0.25):
#         v1,v2 =>tensor
#         v1  (batch_size, model_dimension) associated to Q1.
#         v2  (batch_size, model_dimension) associated to Q2.
    scores = tf.matmul(v1,tf.transpose(v2))
    batch_size = scores.shape[0]
    positive = tf.linalg.tensor_diag_part(scores)  
    negative_without_positive = scores-tf.eye(batch_size)*2.0
    
    closest_negative = tf.reduce_max(negative_without_positive,axis = 1,keepdims=True)
    
    negative_zero_on_duplicate = tf.multiply(scores,1.0-tf.eye(batch_size))
    mean_negative = tf.reduce_sum(negative_zero_on_duplicate,axis=1,keepdims=True)/(batch_size-1)
    
    triplet_loss1 = tf.math.maximum(closest_negative-tf.reshape(positive,[batch_size,1])+margin,0.0)
    triplet_loss2 = tf.math.maximum(mean_negative-tf.reshape(positive,[batch_size,1])+margin,0.0)
    
    triplet_loss = tf.reduce_mean(triplet_loss1+triplet_loss2)
    
    return triplet_loss

threshold=0.7
for epoch in range(200):
     for step, (x, y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            out1=net_embedding(x)
            out1=net_LSTM(out1)
            out2=net_embedding(y)
            out2=net_LSTM(out2)
            loss=TripletLossFn(out1, out2, margin=0.25)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        if step %100 == 0:
            print(epoch, step, 'loss:', float(loss))
    
     total_num = 0
     total_correct = 0
     for i,(x,y) in enumerate(db_test):
            
            out1=net_embedding(x)
            out1=net_LSTM(out1)
            out2=net_embedding(y)
            out2=net_LSTM(out2)
            y_testt = y_test[i:i+batchsz]
            correct=0
            for j in range(batchsz):
                d = tf.matmul(tf.reshape(out1[j],[1,-1]),tf.reshape(out2[j],[-1,1]))
                d=tf.reduce_sum(d)
                res = d > threshold
                res=tf.cast(res,dtype=tf.int64)
                correct += tf.cast((y_testt[j] == res),dtype=tf.int64)
            
            total_num += x.shape[0]
            total_correct += int(correct)
     acc = total_correct / total_num
     print(epoch, 'acc:', acc)
