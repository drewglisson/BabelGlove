## predict script
import pickle

clf = pickle.load(open('trained_model.sav', 'rb'))

print(clf.predict([[-0.1152,-0.9199,-0.3704,1.4000,-5.4600,-2.1000,407,481,516,442,855]]))