Here, I implement Siamese Network with TensorFlow 2.x to identify whether two sentences have the same meaning. The core idea of this algorithm is one-shot learning.

The network structure can be seen in the figure below.

![image](https://github.com/ZhaohuaFang/Siamese-Network/blob/master/Siamese_LSTM.PNG)

Given two sentences, we first get word embeddings of each word. Then we send two sentences encoded by word embeddings to LSTMs, which share the same set of parameters, to obtain the meaning of two sentences respectively. Finally, we compute the cosine similarity of two outputs to determine whether two sentences represent the same meaning. 

When defining the loss function, we use the idea of hard triplets to help the model learn more. We combine two kinds of loss functions that are mean negative and closest negative, which can be seen in the function TripletLossFn() shown in Siamese_LSTM.py.

The dataset is downloaded from assignment of Coursera Natural Language Processing Specialization, Part 3, which is accessible via this link:
https://drive.google.com/file/d/16fAkNRpTMD-a2NIaOs1si8SWuDNK3jTA/view?usp=sharing
