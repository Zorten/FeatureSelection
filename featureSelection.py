import pandas as pd
import math
import random
import copy

#Cross Validation
def leave_one_out_cross_validation(data, current_set, feature_to_add): 
    current_set = copy.deepcopy(current_set)
    current_set.append(feature_to_add)
    
    # print("   Using feature(s) {", end='')
    # for feature in current_set:
    #     print(str(feature) + ", ", end='')

    # print("} ", end='')

    number_correctly_classified = 0
    #Loop to traverse the instances
    for i in range(0, len(data)):
        #Initialize variables

        featureDF = [] ###/
        for feature in current_set: 
            featureDF.append(data.iloc[i, feature])
        object_to_classify = pd.Series(featureDF) ###\


        label_object_to_classify = data.iloc[i, 0]
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        #Loop to compare each instance with all its neighbors
        for k in range(0, len(data)):
            #Don't compare to yourself
            if (k != i):

                featureDF = [] ###/
                for feature in current_set: 
                    featureDF.append(data.iloc[k, feature])
                neighbor = pd.Series(featureDF) ###\


                distance = math.dist(object_to_classify, neighbor)
                if (distance < nearest_neighbor_distance):
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data.iloc[nearest_neighbor_location, 0]

        # print("Object " + str(i+1) + " is class " + str(label_object_to_classify))
        # print("-- Its nearest neighbor is " + str(nearest_neighbor_location + 1) + " which is in class " + str(nearest_neighbor_label))

        if (label_object_to_classify == nearest_neighbor_label):
            number_correctly_classified += 1

    accuracy = float(number_correctly_classified / len(data))
    # print("accuracy is " + str(round(accuracy*100, 1)) + "%")
    return accuracy


#Search function
def feature_search_forward_selection(data):
    #Initialize empty set
    current_set_of_features = []

    highestAccuracy = 0
    best_set_of_features = []

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
                #K-fold cross validation to calculate accuracy
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)

                if not current_set_of_features:
                    print("Using feature(s) {" + str(k) + "} accuracy is " + str(round(accuracy * 100, 1)) + "%")
                else:
                    print("Using feature(s) {" + str(current_set_of_features)[1:-1] + ", " + str(k) + "} accuracy is " + str(round(accuracy * 100, 1)) + "%")

                #Add feature that returns the highest accuracy
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        #Update working set of features
        current_set_of_features.append(feature_to_add_at_this_level)
       # print("On level " + str(i) + " I added feature " + str(feature_to_add_at_this_level) + " to current set, given an accuracy of " + str(best_so_far_accuracy))
        print("Feature set: {" + str(current_set_of_features)[1:-1] + "} was best, accuracy is " + str(round(best_so_far_accuracy * 100, 1)))
        print()


        if (best_so_far_accuracy > highestAccuracy):
            highestAccuracy = best_so_far_accuracy
            best_set_of_features = copy.deepcopy(current_set_of_features)

    print("Finished Search! The best feature subset is: " + str(best_set_of_features)[1:-1] + "}, which has an accuracy of: " + str(round(highestAccuracy * 100, 1)))


def main():
    print("Welcome to Zergio's Feature Selection Algorithm")
    filename = input("Type in the name of the file to test: ")

    try:
        data = pd.read_csv(filename, sep="  ", header=None, engine='python')
    except:
        print("No such file exists. Goodbye!")
        exit()

    print("Type the number of the algorithm you want to run.")
    print("   1) Forward Selection")
    print("   2) Backward Elimination (FIXME)")
    algorithm = int(input())

    if (algorithm == 1):
        print("This dataset has " + str(len(data.columns) - 1) + " features (not including the class attribute), with " + str(len(data)) + " instances.")
        print("Running nearest neighbor with sets of all " + str(len(data.columns) - 1) + " features, using \"leaving-one-out\" evaluation.", end='')
        print("Beginning Search")
        print()
        feature_search_forward_selection(data)
    elif (algorithm == 2):
        print("FIXME!")
    else:
        print("Not an option. Goodbye!")
        exit()


##Run program
main()







