from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
import graphviz


testdf = pd.read_csv('train5.csv')
zeros = []
for i in range(len(testdf)):
    zeros.append(0)
#============================================ DATA PRE-PROCESSING =========================================================
testdf = testdf.sort_values("Label")                                                    #sort by label value

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

testdf = testdf[["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","NBC","ESPN","FOX","ABC","CBS","Label"]]    #only select important columns

#convert win/lose labels to ints
testdf.loc[testdf.Label == 'Lose', 'Label'] = 0                                         #replace Lose labels with '0'
testdf.loc[testdf.Label == 'Win', 'Label'] = 1                                          #replace Win labels with '1'
testdf['Label'] = testdf['Label'].apply(pd.to_numeric)                                  #convert Label column from string to int data type

#convert location values to ints
testdf.loc[testdf.Is_Home_or_Away == 'Home', 'Is_Home_or_Away'] = '1'                     #replace Home with '1'
testdf.loc[testdf.Is_Home_or_Away == 'Away', 'Is_Home_or_Away'] = '0'                     #replace Away with '0'
testdf['Is_Home_or_Away'] = testdf['Is_Home_or_Away'].apply(pd.to_numeric)              #convert column to int

#replace AP top 25 with int values (like above)
testdf.loc[testdf.Is_Opponent_in_AP25_Preseason == 'In', 'Is_Opponent_in_AP25_Preseason'] = '1'
testdf.loc[testdf.Is_Opponent_in_AP25_Preseason == 'Out', 'Is_Opponent_in_AP25_Preseason'] = '0'
testdf['Is_Opponent_in_AP25_Preseason'] = testdf['Is_Opponent_in_AP25_Preseason'].apply(pd.to_numeric)  


# #replace media values with ints (like above)
# testdf.loc[testdf.Media == "1-NBC", 'Media'] = '1'
# testdf.loc[testdf.Media == "2-ESPN", 'Media'] = '2'
# testdf.loc[testdf.Media == "3-FOX", 'Media'] = '3'
# testdf.loc[testdf.Media == "4-ABC", 'Media'] = '4'
# testdf.loc[testdf.Media == "5-CBS", 'Media'] = '5'
# testdf['Media'] = testdf['Media'].apply(pd.to_numeric) #convert column to int

testData = testdf[["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","NBC","ESPN","FOX","ABC","CBS"]].values    #convert sorting features into 2-d array
testLabels = testdf["Label"].values                                                      #convert labels into array
#============================================================================================================================

print(testdf)


clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(testData, testLabels)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","NBC","ESPN","FOX","ABC","CBS"], class_names=['Lose','Win']) 
graph = graphviz.Source(dot_data) 
graph.render("tree")











# def findBestSplit(E,F,LabelName):
#     print(E)
#     for feature in F:
#         print(feature[0])
#         for value in feature[1]:
#             print("     " + str(value))

# class treeNode:
#     def __init__(self):
#         label = ""                  #only leaf nodes will have labels
#         testAttribute = ""          #only internal nodes will have test cond
#         inBoundCond = []            #every node except the root will have an inbound condition
#         children = []               #only internal nodes will have children nodes


# def stopping_cond(E):
#     rowLabels = E["Label"].values #get the class labels of all rows in E as an array
#     if len(rowLabels) > 0: #if E is not empty
#         firstRowLabel = rowLabels[0]
#     for label in rowLabels: #check to see if all the labels in E are the same
#         if label != firstRowLabel:
#             return False #return false if not all records in E have the same class label

#     return True

# def Classify(E):
#     labels = E["Label"].values
#     if len(labels) > 0:
#         return labels[0]
#     else:
#         Exception("Error: there are no records in E")




# def createDecisionTree(E,F,labelName):
#     if stopping_cond(E) or True:
#         print("leaf")
#         leaf = treeNode()
#         leaf.label = Classify(E)
#         return leaf
#     else:
#         print("internal node")
#         root = treeNode()
#         root.testAttribute = findBestSplit(E,F)


# testdf = pd.read_csv('train5.csv')
# #findBestSplit(testdf,[ ["Is_Home_or_Away",["Home","Away"]] , ["Is_Opponent_in_AP25_Preseason",["In","Out"]] , ["Media",["1-NBC","2-ESPN","3-FOX","4-ABC","5-CBS"]] ], "Label")
# myTree = createDecisionTree(testdf,[ ["Is_Home_or_Away",["Home","Away"]] , ["Is_Opponent_in_AP25_Preseason",["In","Out"]] , ["Media",["1-NBC","2-ESPN","3-FOX","4-ABC","5-CBS"]] ], "Label")
# print(myTree.label)