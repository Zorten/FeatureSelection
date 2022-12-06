import pandas as pd
import random

data = pd.read_csv("CS170_Small_Data__114.txt", sep="  ", header=None, engine='python')
#print(data.head(10))


####TACKLE SEARCH FIRST

#STUB FUNCTION
def accuracy(): #leave_one_out_cross_validation(data, current_set, feature_to_add)
    accuracy = random.random()
    return accuracy

#Search function
def feature_search(data):
    #In python, range() function is exclusive on its upper boundary, and for that reason 
    # I don't -1 from the features since the range will go from 1 to numFeatures-1 automatically
    numFeatures = len(data.columns)

    #For loop to walk down the levels of search tree
    for i in range(1, numFeatures):
        print("On the " + str(i) + "th level of the search tree")  
        
        #Loop to consider each feature individually
        for k in range(1, numFeatures):
            print("--Considering adding the " + str(k) + " feature")


feature_search(data)    

