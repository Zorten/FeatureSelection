import pandas as pd
import random

data = pd.read_csv("CS170_Small_Data__114.txt", sep="  ", header=None, engine='python')
#print(data.head(10))


####TACKLE SEARCH FIRST

#STUB FUNCTION
def leave_one_out_cross_validation(data, current_set, feature_to_add):
    accuracy = random.random()
    return accuracy

#Search function
def feature_search(data):
    #Initialize empty set
    current_set_of_features = []

    #In python, range() function is exclusive on its upper boundary, and for that reason 
    # I don't -1 from the features since the range will go from 1 to numFeatures-1 automatically
    numFeatures = len(data.columns)

    #For loop to walk down the levels of search tree
    for i in range(1, numFeatures):
        print("On the " + str(i) + "th level of the search tree")  
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0
        
        #Loop to consider each feature individually
        for k in range(1, numFeatures):
            #Only consider adding if not already added
            if (current_set_of_features.count(k) <= 0):
                print("--Considering adding the " + str(k) + " feature")
                #K-fold cross validation to calculate accuracy
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k+1)

                #Add feature that returns the highest accuracy
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        #Update working set of features
        current_set_of_features.append(feature_to_add_at_this_level)
        print("On level " + str(i) + " I added feature " + str(feature_to_add_at_this_level) + " to current set")

    print(current_set_of_features)

feature_search(data)  

