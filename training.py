import random
import numpy as np
import glob
import os
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def main():
    
    ## glob the list of txt files in the current dir to loop 
    txtFiles = glob.glob("data\*.txt")
    # txtFiles2 = glob.glob("data_new\*.txt")

    ## pulling data from txt filess 
    X, y = pullTxtSamples(txtFiles)

    ## slip data into training and test cases 
    train_data, test_data, train_label , test_label = train_test_split(
        X, y, test_size=0.2, random_state=42)

    ## grid search looking for best parameters for the estimator
    # param_grid = [
    #     {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    #     {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    # ]

    # clf = GridSearchCV(SVC(), param_grid, verbose = 5, n_jobs=-1)
    # clf.fit(train_data, train_label)
    
    # print(clf.best_params_) 
    # print(clf.best_estimator_) ------------> SVC(C=1000, gamma=0.001. kernel='rbf')
    

    ## creating pipline and fitting data into SVC
    print('Pre Learning...')
    clf = make_pipeline(StandardScaler(), SVC(C=1000, gamma=0.001, kernel='rbf'))
    clf.fit(train_data, train_label)

    ## finding accuracy of model with test data
    print("Fit Score")
    print(clf.score(test_data, test_label))

    pickle.dump(clf, open('trained_model.sav', 'wb'))
    
    
def pullTxtSamples(txtFiles):

    print("Collecting samples from txt files...")

    fullarray = np.empty((0,11))
    y = np.empty(0)
    
    for file in txtFiles:
        ## copies the file to pull samples
        print(file)
        data = np.loadtxt(fname = file)
        file = os.path.splitext(file)[0]
        
        while(1):
            ## goes thru the file pulling each sample 
            ## when run out of samples move to next file 
            if (data.shape == (0,)):
                break
            
            ## get the group of samples, 11 values each
            temp = np.empty(11)
            temp = data[:11]

            # print(temp[10])
            # if (temp[10] < 100.0) | (temp[9] < 100.0) | (temp[8] < 100.0) | (temp[7] < 100.0) | (temp[6] < 100.0) :
            #     print(temp)
            #     return
                
            temp = np.resize(temp, (1,11))

            ## delete the pulled sample from the file
            data = np.delete(data, (0,1,2,3,4,5,6,7,8,9,10))

            ## add that sample to master array
            fullarray = np.append(fullarray, temp, axis=0)

            ## update label array with file name             
            y = np.append(y, file.replace('data\\',''))

    print("Complete\n")

    return fullarray, y

if __name__ == '__main__':
    main()
