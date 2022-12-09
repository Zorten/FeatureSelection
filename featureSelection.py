import math
import copy
import time
import numpy as np

#Cross Validation using nearest neighbor
def leave_one_out_cross_validation(data, current_set, feature_to_test, choice): 
    #Create entire set of features to be tested (current_set + feature_to_add)
    current_set = copy.deepcopy(current_set)

    if (choice == 1):
        current_set.append(feature_to_test)
    elif (choice == 2):
        current_set.remove(feature_to_test)

    #Initialize variables
    num_rows = len(data)
    number_correctly_classified = 0

    #Loop to traverse the instances
    for i in range(0, num_rows):
        #Create object with only features being tested
        object_to_classify = []
        for feature in current_set: 
           object_to_classify.append(data[i][feature])


        #Initialize variables
        label_object_to_classify = data[i][0]
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        #Loop to compare each instance with all its neighbors
        for k in range(0, num_rows):
            #Don't compare to yourself
            if (k != i):
                #Create neighbor to compare to, with only features being tested
                neighbor = []
                for feature in current_set: 
                    neighbor.append(data[k][feature])

                #Calculate distancce between current object and current neighbor
                distance = math.dist(object_to_classify, neighbor)

                #If a new nearest neighbor was found, update variables
                if (distance < nearest_neighbor_distance):
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location][0]

        #After comparing curring object with all neighbors and finding a neearest one,
        #see if classification was correct
        if (label_object_to_classify == nearest_neighbor_label):
            number_correctly_classified += 1

    #Calculate accuracy based on number of correct nearest neighbor calculations
    accuracy = float(number_correctly_classified / num_rows)
    return accuracy

#Forward selection algorithm function
def feature_search_forward_selection(data):
    #Initialize empty set
    current_set_of_features = []

    #Variables to keep track of final answer
    highestAccuracy = 0
    best_set_of_features = []

    #In python, range() function is exclusive on its upper boundary, and for that reason 
    # I don't -1 from the features since the range will go from 1 to numFeatures-1 automatically
    numFeatures = len(data[0])

    #For-loop to walk down the levels of search tree
    for i in range(1, numFeatures):
        #Output formatting
        print("==============================================")
        print("On the " + str(i) + "th level of the search tree")  
        print()

        #Initialize variables
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0
        
        #Loop to consider each feature individually
        for k in range(1, numFeatures):
            #Only consider adding if not already added
            if (current_set_of_features.count(k) <= 0):
                #K-fold cross validation to calculate accuracy
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k, 1)

                #Output formatting
                if not current_set_of_features:
                    print("  Using feature(s) {" + str(k) + "} accuracy is " + str(round(accuracy * 100, 1)) + "%")
                else:
                    print("  Using feature(s) {" + str(current_set_of_features)[1:-1] + ", " + str(k) + "} accuracy is " + str(round(accuracy * 100, 1)) + "%")

                #Add feature that returns the highest accuracy, and update highest accuracy
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        #Update working set of features
        current_set_of_features.append(feature_to_add_at_this_level)
        print()
        print("Feature set {" + str(current_set_of_features)[1:-1] + "} was best, accuracy is " + str(round(best_so_far_accuracy * 100, 1)) + "%")

        #Keeps track of best set and accuracy to determine final answer
        if (best_so_far_accuracy > highestAccuracy):
            highestAccuracy = best_so_far_accuracy
            best_set_of_features = copy.deepcopy(current_set_of_features)

    #Search
    print("==============================================")
    print("Finished Search! The best feature subset is {" + str(best_set_of_features)[1:-1] + "}, which has an accuracy of: " + str(round(highestAccuracy * 100, 1)) + "%")

#Backward elimination algorithm function
def feature_search_backward_elimination(data):
    #In python, range() function is exclusive on its upper boundary, and for that reason 
    # I don't -1 from the features since the range will go from 1 to numFeatures-1 automatically
    numFeatures = len(data[0])

    #Initialize set with all of the features
    current_set_of_features = [*range(1, numFeatures)]

    #Variables to keep track of final answer
    highestAccuracy = 0
    best_set_of_features = []


    #For-loop to walk down the levels of search tree
    for i in range(1, numFeatures-1):
        #Output formatting
        print("==============================================")
        print("On the " + str(i) + "th level of the search tree")  
        print()

        #Initialize variables
        feature_to_remove_at_this_level = 0
        best_so_far_accuracy = 0
        
        #Loop to consider each feature individually
        for k in range(1, numFeatures):
            #Only consider adding if not already added
            if (current_set_of_features.count(k) > 0):
                #K-fold cross validation to calculate accuracy
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k, 2)

                #Output formatting
                # if not current_set_of_features:
                #     print("  Using feature(s) {" + str(k) + "} accuracy is " + str(round(accuracy * 100, 1)) + "%")
                # else:
                display_set = copy.deepcopy(current_set_of_features)
                display_set.remove(k)
                print("  Using feature(s) {" + str(display_set)[1:-1] + "} accuracy is " + str(round(accuracy * 100, 1)) + "%")

                #Add feature that returns the highest accuracy, and update highest accuracy
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = k

        #Update working set of features
        current_set_of_features.remove(feature_to_remove_at_this_level)
        print()
        print("Feature set {" + str(current_set_of_features)[1:-1] + "} was best, accuracy is " + str(round(best_so_far_accuracy * 100, 1)) + "%")

        #Keeps track of best set and accuracy to determine final answer
        if (best_so_far_accuracy > highestAccuracy):
            highestAccuracy = best_so_far_accuracy
            best_set_of_features = copy.deepcopy(current_set_of_features)

    #Search
    print("==============================================")
    print("Finished Search! The best feature subset is {" + str(best_set_of_features)[1:-1] + "}, which has an accuracy of: " + str(round(highestAccuracy * 100, 1)) + "%")
    print()

#Main function
def main():
    #Display welcome message and get filename from user
    print("Welcome to Zergio's Feature Selection Algorithm")
    filename = input("Type in the name of the file to test: ")

    #Attempt to load Data into numpy array, if fails then exit program
    try:
        data = np.loadtxt(filename)
        data = data.tolist()
        
    except:
        print("No such file exists. Goodbye!")
        exit()

    #Get user input for algorithm selection
    print("Type the number of the algorithm you want to run.")
    print("   1) Forward Selection")
    print("   2) Backward Elimination")
    algorithm = int(input())

    #Get and display basic info for the data
    num_rows = len(data)
    num_columns = len(data[0])
    print()
    print("This dataset has " + str(num_columns - 1) + " features (not including the class attribute), with " + str(num_rows) + " instances.")
    print("Running nearest neighbor with all " + str(num_columns - 1) + " features, using \"leaving-one-out\" evaluation, I get accuracy: ", end='')

    #Forward selection algorithm
    if (algorithm == 1):
        accuracy = leave_one_out_cross_validation(data, [*range(1, num_columns - 1)], num_columns-1, 1)
        print(str(round(accuracy * 100, 1)) + "%")
        print()
        print("Beginning Forward Selection Search")
        startTime = time.time()
        feature_search_forward_selection(data)
        endTime = time.time()
        printTime(startTime, endTime)
    
    #Backward Elimination algorithm
    elif (algorithm == 2):
        accuracy = leave_one_out_cross_validation(data, [*range(0, num_columns)], 0, 2)
        print(str(round(accuracy * 100, 1)) + "%")
        print()
        print("Beginning Backward Elimination Search")
        startTime = time.time()
        feature_search_backward_elimination(data)
        endTime = time.time()
        printTime(startTime, endTime)

    #Incorrect input
    else:
        print("Not an option. Goodbye!")
        exit()

def printTime(startTime, endTime):
    print("==============================================")
    totalTime = endTime - startTime
    if (totalTime >= 60):
        totalTime = totalTime / 60
        totalTime = round(totalTime, 1)
        print("Time elapsed: " + str(totalTime) + " minutes")
    elif (totalTime >= 1):
        totalTime = round(totalTime, 1)
        print("Time elapsed: " + str(totalTime) + " seconds")
    else:
        totalTime = totalTime * 1000
        totalTime = round(totalTime)
        print("Time elapsed: " + str(totalTime) + " milliseconds")


##Run program
main()







