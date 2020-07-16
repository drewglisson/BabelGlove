import random
import numpy as np
import glob
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def main():
    
    ## glob the list of txt files in the current dir to loop 
    txtFiles = glob.glob("*.txt")
    
    ## pulling data from txt filess 
    X, y = pullTxtSamples(txtFiles)

    # pullTxtSamples(txtFiles, samples)
    # return

    ## slip data into training and test cases (9721 samples)
    train_data = X[:7776]
    train_label = y[:7776]

    test_data = X[7776:]
    test_label = y[7776:]   

    ## creating pipline and fitting data into SVC
    print('Pre Learning...')
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    clf.fit(train_data, train_label)

    ## finding accuracy of model with test data
    print("Fit Score")
    print(clf.score(test_data, test_label))
    

def pullTxtSamples(txtFiles):

    print("Collecting samples from txt files...")

    fullarray = np.empty((0,11))
    y = np.empty(0)
    
    for file in txtFiles:
        ## copies the file to pull samples
        data = np.loadtxt(fname = file)
        # print(file)
        
        while(1):
            ## goes thru the file pulling each sample 
            ## when run out of samples move to next file 
            if (data.shape == (0,)):
                break
            
            ## get the group of samples, 11 values each
            temp = np.empty(11)
            temp = data[:11]
            temp = np.resize(temp, (1,11))

            ## delete the pulled sample from the file
            data = np.delete(data, (0,1,2,3,4,5,6,7,8,9,10))

            ## add that sample to master array
            fullarray = np.append(fullarray, temp, axis=0)

            ## update label array with file name 
            y = np.append(y, os.path.splitext(file)[0])

    # print(fullarray.shape)
    # print(y.size)

    print("Complete\n")
    ## create unison-shuffled arrays 
    p = np.random.permutation(y.size)

    return fullarray[p], y[p]

if __name__ == '__main__':
    main()
