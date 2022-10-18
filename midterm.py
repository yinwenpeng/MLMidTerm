
#import sklearn as sk
import nltk
import numpy as np


def __main__():
    tf = trainFile("train_with_label.txt")
    tf.parseWithNLTK()
    exit





class trainFile:
    def __init__ (self, file):
        self.file = file
        self.trainCount = 0
        self.trainDimensions = 10
        self.dataID = []
        self.data = []
        self.label = []
        self.features = 0


    def parseFile(self):
        f = open(self.file, "r")
        
        for line in f:
            tabSplit = line.split("\t")
            
            self.dataID.append(tabSplit[0])
            self.label.append(tabSplit[-1])

            sent1 = tabSplit[1]
            sent2 = tabSplit[2]

            self.data.append([sent1, sent2])

            self.label.append(tabSplit[-1])

                    
        f.close()

    def parseWithNLTK(self):
        f = open(self.file, "r")
        for line in f:
            tabSplit = line.split("\t")

            self.dataID.append(tabSplit[0])

            sent1 = tabSplit[1]
            sent2 = tabSplit[2]
            self.data.append([sent1, sent2])
            
            self.label.append(tabSplit[-1])

            self.trainCount += 1

        self.features = np.zeros(shape=(self.trainCount, self.trainDimensions))

        for line in self.data:
            

            
            line[0] = line[0].lower()
            line[1] = line[1].lower()
            line[0] = nltk.word_tokenize(line[0])
            
            line[1] = nltk.word_tokenize(line[1])
            
        

        print(self.data[0][0])
        print(self.data[-1][0])
        print(self.trainCount)
        print(self.features[0])
        print(len(self.data[0][0]))

        f.close()

    def compareLength(self, line1, line2):
        line1 = line1

    
__main__()
