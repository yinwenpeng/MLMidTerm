
import nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



import numpy as np

import math

'''
loss="hinge" -> SVM
loss="log_loss" -> logistic regression

penalty="l2", "l1", "elasticnet" (l2 is default)

max_iter, 1000 is default

tol, stopping parameter
early_stopping, true or false
n_iter_no_change,
score, 
'''

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


def __main__():
    #get feature vectors
    train = fileToFeatureMatrix("train_with_label.txt")
    x_train, y_train = train.makeFeatures()
    
    dev = fileToFeatureMatrix("dev_with_label.txt")
    x_dev, y_dev = dev.makeFeatures()

    test = fileToFeatureMatrix("test_without_label.txt")
    x_test = test.makeFeatures()

    #scale
    sc = StandardScaler()
    sc.fit(x_train)
    
    x_train_scaled = sc.transform(x_train)
    x_dev_scaled = sc.transform(x_dev)
    x_test_scaled = sc.transform(x_test)
    

    while True:
        count += 1
        svmSGD_scaled = SGDClassifier(loss="hinge", tol = 0.0001, max_iter=10**8).fit(x_train_scaled, y_train)
        if accuracy(y_dev, svmSGD_scaled.predict(x_dev_scaled)) > 0.69:
            break

    print("SGD svm scaled result scaled: ", accuracy(y_dev, svmSGD_scaled.predict(x_dev_scaled)))
    

    y_prediction = svmSGD_scaled.predict(x_test_scaled)

    f = open("SeanBritt_test_result.txt", "w")

    for i in range(0, len(y_prediction)):
        string = str(i) + "\t" + str(y_prediction[i]) +"\n"
        f.write(string)

    f.close()
    
    f=open("SeanBritt_test_result.txt", "r")
    f.readlines()

    f.close()


    exit

def accuracy(gold, pred):
    count = 0
    for i in range(0, len(pred)):
        if gold[i] == pred[i]:
            count += 1
        

    return (count/len(gold))



class fileToFeatureMatrix:
    
    def __init__ (self, file):
        self.file = file
        self.vectorCount = 0
        self.trainDimensions = 5
        self.dataID = []
        self.data = []
        self.label = []
        self.features = 0
        self.stemmer = PorterStemmer()
        self.rawDataLines = []
        self.stop_words = set(stopwords.words('english'))

   
    def makeFeatures(self):

        #open the text file for reading
        f = open(self.file, "r")
        
        '''
        Look at each line of the file and turn it into a sentence pair and a label
        '''
        #initial splitting of the lines by /t and sending the info where it needs to go
        for line in f:
            
            tabSplit = line.split("\t")

            #these are the two sentences and the label for the pair
            '''
            1. make the sentence lower case
            2. tokenize the sentence
            3. clean the token string 
            '''

            sent1 = self.cleanTok(nltk.word_tokenize(tabSplit[1].lower()))
            sent2 = self.cleanTok(nltk.word_tokenize(tabSplit[2].lower()))
            self.data.append((sent1, sent2))

            #this checks for the test case
            if len(tabSplit) > 3:
                self.label.append(int(tabSplit[-1]))

            #keep the count of sentence pairs.  this will be equal to the length of self.data and self.label
            self.vectorCount += 1

        #start the feature list with all 0s
        self.features = np.zeros(shape=(self.vectorCount, self.trainDimensions))
        for i in range(0, self.vectorCount):
            sent1 = self.data[i][0]
            sent2 = self.data[i][1]
        
            weights = [(1, 0, 0, 0), (1./2., 1./2., 0, 0), (1./3., 1./3., 1./3., 0), (1./4., 1./4., 1./4., 1./4.)]
            bleu_scores1 = sentence_bleu([sent1], sent2, weights, smoothing_function=SmoothingFunction().method1)
            
            self.features[i][0] = bleu_scores1[0]
            self.features[i][1] = bleu_scores1[1]
            self.features[i][2] = bleu_scores1[2]
            self.features[i][3] = bleu_scores1[3]

            self.features[i][4] = math.exp(abs(len(sent1)-len(sent2)))
            

        f.close()
        if len(self.label) == 0:
            return self.features
        else:        
            return self.features, self.label


    '''
    if a token is longer than 1 character and is not included in the stop_words
    1. stem the token
    2. add it to the list of tokens to return

    then, return the list of cleaned tokens
    '''

    def cleanTok(self, line):
        cleantoks = []
        
        for tok in line:
            if(len(tok) > 1 and (tok not in self.stop_words)):
                cleantoks.append(self.stemmer.stem(tok))
        return cleantoks
    
    '''
    -----------------FEATURE CONGLOMERATION
    '''

    #compare the lengths of the tokens as a feature 
    def ft_length(self, line1, line2):
        return 1/(max(1, (len(line1)-len(line2))**2) )
        

    
__main__()
