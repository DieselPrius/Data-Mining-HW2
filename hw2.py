from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
import graphviz


traindf = pd.read_csv('train5.csv')
zeros = []
for i in range(len(traindf)):
    zeros.append(0)
#============================================ DATA PRE-PROCESSING =========================================================
traindf = traindf.sort_values("Label")                                                    #sort by label value

#convert Media feature to binary features for each meadia outlet
traindf["NBC"] = zeros
traindf["ESPN"] = zeros
traindf["FOX"] = zeros
traindf["ABC"] = zeros
traindf["CBS"] = zeros
traindf.loc[traindf.Media == "1-NBC", 'NBC'] = 1
traindf.loc[traindf.Media == "2-ESPN", 'ESPN'] = 1
traindf.loc[traindf.Media == "3-FOX", 'FOX'] = 1
traindf.loc[traindf.Media == "4-ABC", 'ABC'] = 1
traindf.loc[traindf.Media == "5-CBS", 'CBS'] = 1

traindf = traindf[["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","NBC","ESPN","FOX","ABC","CBS","Label"]]    #only select important columns

#convert win/lose labels to ints
traindf.loc[traindf.Label == 'Lose', 'Label'] = 0                                         #replace Lose labels with '0'
traindf.loc[traindf.Label == 'Win', 'Label'] = 1                                          #replace Win labels with '1'
traindf['Label'] = traindf['Label'].apply(pd.to_numeric)                                  #convert Label column from string to int data type

#convert location values to ints
traindf.loc[traindf.Is_Home_or_Away == 'Home', 'Is_Home_or_Away'] = '1'                     #replace Home with '1'
traindf.loc[traindf.Is_Home_or_Away == 'Away', 'Is_Home_or_Away'] = '0'                     #replace Away with '0'
traindf['Is_Home_or_Away'] = traindf['Is_Home_or_Away'].apply(pd.to_numeric)              #convert column to int

#replace AP top 25 with int values (like above)
traindf.loc[traindf.Is_Opponent_in_AP25_Preseason == 'In', 'Is_Opponent_in_AP25_Preseason'] = '1'
traindf.loc[traindf.Is_Opponent_in_AP25_Preseason == 'Out', 'Is_Opponent_in_AP25_Preseason'] = '0'
traindf['Is_Opponent_in_AP25_Preseason'] = traindf['Is_Opponent_in_AP25_Preseason'].apply(pd.to_numeric)  

# #replace media values with ints (like above)
# traindf.loc[traindf.Media == "1-NBC", 'Media'] = '1'
# traindf.loc[traindf.Media == "2-ESPN", 'Media'] = '2'
# traindf.loc[traindf.Media == "3-FOX", 'Media'] = '3'
# traindf.loc[traindf.Media == "4-ABC", 'Media'] = '4'
# traindf.loc[traindf.Media == "5-CBS", 'Media'] = '5'
# traindf['Media'] = traindf['Media'].apply(pd.to_numeric) #convert column to int

trainData = traindf[["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","NBC","ESPN","FOX","ABC","CBS"]].values    #convert sorting features into 2-d array
trainLabels = traindf["Label"].values                                                      #convert labels into array





testdf = pd.read_csv('test5.csv')
zeros = []
for i in range(len(testdf)):
    zeros.append(0)
#convert Media feature to binary features for each meadia outlet
testdf["NBC"] = zeros
testdf["ESPN"] = zeros
testdf["FOX"] = zeros
testdf["ABC"] = zeros
testdf["CBS"] = zeros
testdf.loc[testdf.Media == "1-NBC", 'NBC'] = 1
testdf.loc[testdf.Media == "2-ESPN", 'ESPN'] = 1
testdf.loc[testdf.Media == "3-FOX", 'FOX'] = 1
testdf.loc[testdf.Media == "4-ABC", 'ABC'] = 1
testdf.loc[testdf.Media == "5-CBS", 'CBS'] = 1

testdf = testdf[["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","NBC","ESPN","FOX","ABC","CBS"]]    #only select important columns

# #convert win/lose labels to ints
# testdf.loc[testdf.Label == 'Lose', 'Label'] = 0                                         #replace Lose labels with '0'
# testdf.loc[testdf.Label == 'Win', 'Label'] = 1                                          #replace Win labels with '1'
# testdf['Label'] = testdf['Label'].apply(pd.to_numeric)                                  #convert Label column from string to int data type

#convert location values to ints
testdf.loc[testdf.Is_Home_or_Away == 'Home', 'Is_Home_or_Away'] = '1'                     #replace Home with '1'
testdf.loc[testdf.Is_Home_or_Away == 'Away', 'Is_Home_or_Away'] = '0'                     #replace Away with '0'
testdf['Is_Home_or_Away'] = testdf['Is_Home_or_Away'].apply(pd.to_numeric)              #convert column to int

#replace AP top 25 with int values (like above)
testdf.loc[testdf.Is_Opponent_in_AP25_Preseason == 'In', 'Is_Opponent_in_AP25_Preseason'] = '1'
testdf.loc[testdf.Is_Opponent_in_AP25_Preseason == 'Out', 'Is_Opponent_in_AP25_Preseason'] = '0'
testdf['Is_Opponent_in_AP25_Preseason'] = testdf['Is_Opponent_in_AP25_Preseason'].apply(pd.to_numeric)  

testData = testdf[["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","NBC","ESPN","FOX","ABC","CBS"]].values    #convert sorting features into 2-d array
#============================================================================================================================


#===================create the decision tree ==================================================================
clf = tree.DecisionTreeClassifier(criterion="gini")
clf = clf.fit(trainData, trainLabels)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","NBC","ESPN","FOX","ABC","CBS"], class_names=['Lose','Win']) 
graph = graphviz.Source(dot_data) 
graph.render("tree")
#===================================================================================================================

#================== predict labels for test data ====================================================
print(testdf)
print(clf.predict(testData))
predictionResults = {'Test ID': [25,26,27,28,29,20,31,32,33,34,35,36], 'Prediction': clf.predict(testData)}
resultsdf = pd.DataFrame(data=predictionResults)
print(resultsdf)