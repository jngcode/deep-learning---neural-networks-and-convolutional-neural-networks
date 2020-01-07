"""
This file contains, CNN, CNN without convolutional layer and both KNN models.
Model Evaluation contains cross validation (Applies to all learning algoirthms) and confusion matrices (only for SKLearn KNN Model)
Copyrighted by Jay Ng
Github: @jngcode 
"""
#import python libraries needed for f1 and initialise global variables.
import numpy as np
import pandas as pd
import seaborn as sea
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import sklearn.metrics as metrics
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, Dense, Activation
#Initialise global variable, row and column of the digits dataset array is 8x8 (64).
data_row, data_col = 8, 8

def load_data():
    #Loading digits dataset to split into train dataset and test dataset.
    digits = load_digits()
    x = digits.data
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    #Divided by 16 (the highest number) for smaller values, so it would be easier for to compute.
    x_train /= 16
    x_test /= 16
    return x_train, x_test, y_train, y_test, x, y

def my_dnn_model(x_train, x_test, y_train, y_test, x, y):
    global dnn_pred
    print("(non-convolutional layers) DNN Model Running... ")
    #Import data into DNN my model.
    x_train = x_train.reshape(x_train.shape[0], data_row, data_col, 1)
    x_test = x_test.reshape(x_test.shape[0], data_row, data_col, 1)
    #CNN without convolutional layers, by using the keras and tensorflow library.
    dnn_model = Sequential()
    #Flattening the dimensions to 1 dimension.
    dnn_model.add(Flatten(input_shape=(data_row, data_col, 1)))
    dnn_model.add(Dense(128)) #Relu
    dnn_model.add(Activation("relu"))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(10))
    dnn_model.add(Activation("softmax"))
    #Compiling all DNN results for the output. I found that 'adam' is the most efficient optimizer.
    dnn_model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

    #Fit the model and run through epochs 15 times - I found 15 to be the 'best' value. (little difference if using higher epoch inexpense for longer waiting time).
    dnn_model.fit(x_train, y_train, epochs=15)

    #Gaining results from evaluating data - calculates loss and accuracy. 
    #Results is an array, in which results[1] element is the results accuracy score. results[0] is loss of data.
    results = dnn_model.evaluate(x_test, y_test)
    results_accuracy = results[1]
    
    #Saving DNN model to the same project file path, in .h5 filetype.
    print("Saving DNN model...")
    dnn_model.save('dnn_model.h5')
    print("Model saved to disk to the same path, as project.")
    print()
    # Return result of accuracy for printing.
    return results_accuracy
    
def my_cnn_model(x_train, x_test, y_train, y_test, x, y):
    global cnn_pred
    print("(convolutional layers) CNN Model Running... ")
    #Reshape the digits data for CNN algorithm to process to fit.
    x_train = x_train.reshape(x_train.shape[0], data_row, data_col, 1)
    x_test = x_test.reshape(x_test.shape[0], data_row, data_col, 1)
    #CNN algorithm, using keras and tensorflow library with convolutional layer 2d.
    cnn_model = Sequential()
    #Convolutional layer, where kernel_size is the window size of collecting 9 elements of the larger matrix.
    cnn_model.add(Conv2D(64, kernel_size=(3,3), input_shape=(data_row, data_col, 1)))
    cnn_model.add(Activation("relu"))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2))) #Pooling
    #Adding another convolutional 2D layer for a higher accuracy. 
    cnn_model.add(Conv2D(32, kernel_size=(3,3)))
    cnn_model.add(Activation("relu"))
    cnn_model.add(Dropout(0.2))
    #Flat the matrix to 1-Dimensional for easier computation.
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128)) #dense the matrix.
    cnn_model.add(Activation("relu"))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Dense(10)) #dense to 10
    cnn_model.add(Activation("softmax"))
    #Compiling all CNN results for the output. I found that 'adam' is the most efficient optimizer.
    cnn_model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])
    
    #Fit the model and run through epochs 15 times. 
    cnn_model.fit(x_train, y_train, epochs=15)
    
    #Gaining results from evaluating data - calculates loss and accuracy.
    #Results is an array, in which results[1] element is the accuracy score.
    results = cnn_model.evaluate(x_test, y_test)
    results_accuracy = results[1]

    #Saving cnn model to the same file path in .h5 filetype.
    print("Saving CNN model...")
    cnn_model.save('cnn_model.h5')
    print("Model saved to disk to the same path, as project.")
    print()
    # Return result of accuracy for printing.
    return results_accuracy

def results(results):
    #Print results function for all learning algorithms in percentages.
    print("========================= Results =========================")
    print("Accuracy: ", str(results*100) + "%")
    print("Error: ", str(100 - (results*100)) + "%")
    return

#F2 Part 1/3 - Cross Validation:
def cross_val(x_train, x_test, y_train, y_test, x, y):
    #Initialise splits into 5 folds.
    num_splits = 5
    kf = KFold(num_splits)
    kf.get_n_splits(x)
    kf_array_resultsDNN = []
    kf_array_resultsCNN = []
    kf_array_results_skl_knn = []
    kf_array_results_my_knn = []
    #initialise k nearest neighbor number for KNN models.
    knn_value = 5
    
    for i, j in kf.split(x):
        x_train, x_test = x[i], x[j]
        y_train, y_test = y[i], y[j]
        result_dnn = my_dnn_model(x_train, x_test, y_train, y_test, x, y)
        result_cnn = my_cnn_model(x_train, x_test, y_train, y_test, x, y)
        result_knn_skl = knn_skl(x_train, x_test, y_train, y_test, x, y)
        print()
        result_my_knn = myKNN(x_train, y_train, x_test, y_test, knn_value)
        print()
        # result_X[1] - due to [0] is the 'loss' and [1] is the accuracy in the array.
        kf_array_resultsDNN.append(result_dnn)
        kf_array_resultsCNN.append(result_cnn)
        kf_array_results_my_knn.append(result_my_knn)
        kf_array_results_skl_knn.append(result_knn_skl)
    
    #Calculating average accuracies from the trained data, kfolds.
    average_accuracyDNN = ((sum(kf_array_resultsDNN)*100)/num_splits)
    average_accuracyCNN = ((sum(kf_array_resultsCNN)*100)/num_splits)
    average_accuracy_skl_knn = ((sum(kf_array_results_skl_knn)*100)/num_splits)
    average_accuracy_my_knn = ((sum(kf_array_results_my_knn)*100)/num_splits)
    #Call to print cross validations.
    cross_val_results(average_accuracyDNN, 
                      average_accuracyCNN,
                      average_accuracy_skl_knn,
                      average_accuracy_my_knn,
                      kf_array_resultsDNN,
                      kf_array_resultsCNN,
                      kf_array_results_skl_knn,
                      kf_array_results_my_knn)
    return

def cross_val_results(average_accuracyDNN, 
                      average_accuracyCNN,
                      average_accuracy_skl_knn,
                      average_accuracy_my_knn,
                      kf_array_resultsDNN,
                      kf_array_resultsCNN,
                      kf_array_results_skl_knn,
                      kf_array_results_my_knn):
    #Printing the cross validation results for all algorithms (including KNN classifier)
    print("================ Cross Validation Results =====================")
    print("Note: 5 K-Folds for all algorithms")
    print("All DNN Accuracies (non-convolutional layers):")
    print(kf_array_resultsDNN)
    print("DNN Average Accuracy: ", str(average_accuracyDNN) + "%")
    print()
    print("All CNN Accuracies (convolutional layers):")
    print(kf_array_resultsCNN)
    print("CNN Average Accuracy: ", str(average_accuracyCNN) + "%")
    print()
    print("All SKL KNN Accuracies:")
    print(kf_array_results_skl_knn)
    print("SKL KNN Model Average Accuracy: ", str(average_accuracy_skl_knn) + "%")
    print()
    print("All KNN Accuracies:")
    print(kf_array_results_my_knn)
    print("My KNN Model Average Accuracy: ", str(average_accuracy_my_knn) + "%")
    print()
    return

# ============================================================================

def confusion_matrix_model(x_train, x_test, y_train, y_test, x, y):
    #KNN Model SKL
    knn_skl(x_train, x_test, y_train, y_test, x, y)
    print()
    print("=============== SKLearn - KNN Confusion Matrice: ===============")
    confusion_matrix(y_test, skl_knn_pred)
    print()
    return

#F2 Part 2/3 - Confusion Matrices:
def confusion_matrix(y_test, y_pred):
    #Using SKLearn Library for Confusion Matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, range(len(cm)), range(len(cm)))
    ax = plt.axes()
    sea.set(font_scale=1)
    #Using Seaborn to plot with heat bar - level.
    sea.heatmap(df_cm, xticklabels=False, yticklabels=False, annot=True)
    #Axis labels.
    ax.set(xlabel="Predicted", ylabel="True")
    ax.set_title("Confusion matrix")
    #Enable the graph for the confusion matrix.
    plt.show()
    return

#F2 Model Evaluation - Function Call to Parts 1/2/3: 
def model_eval(x_train, x_test, y_train, y_test, x, y):
    cross_val(x_train, x_test, y_train, y_test, x, y)
    confusion_matrix_model(x_train, x_test, y_train, y_test, x, y)
    return
    
#Navigation - Main Menu function:
def main():
    #Load data for learning algorithm - data parameters.
    knn_value = 5
    x_train, x_test, y_train, y_test, x, y = load_data()
    print("Menu: ")
    print("1. DNN Model (non-convolutional layers)")
    print("2. CNN Model (convolutional layers)")
    print("3. Model Evaluation (Confusion matrices - only applies to SKLearn KNN)")
    print("4. Run CNN, DNN, SKL KNN and My KNN Models")
    print("5. Load previous CNN and DNN models and summaries.")
    print("0. Exit")
    #user input allows user to select what function they want.
    user = input("Select an option, 1, 2, 3, 4, 5 or 0: ")
    if user == "1":
        resultsDNN = my_dnn_model(x_train, x_test, y_train, y_test, x, y)
        results(resultsDNN)
        print()
        main()
    elif user == "2":
        resultsCNN = my_cnn_model(x_train, x_test, y_train, y_test, x, y)
        results(resultsCNN)
        print()
        main()
    elif user == "3":
        model_eval(x_train, x_test, y_train, y_test, x, y)
        print()
        main()
    elif user == "4":
        run_all(x_train, x_test, y_train, y_test, x, y, knn_value)
        print()
        main()
    elif user == "5":
        load_prev_models(x_train, x_test, y_train, y_test, x, y)
        print()
        main()
    elif user == "0":
        exit()
    else:
        print("Error: invalid input...")
        print()
        main()
      
def run_all(x_train, x_test, y_train, y_test, x, y, knn_value):
    resultsDNN = my_dnn_model(x_train, x_test, y_train, y_test, x, y)
    resultsCNN = my_cnn_model(x_train, x_test, y_train, y_test, x, y)
    resultsSKL_KNN = knn_skl(x_train, x_test, y_train, y_test, x, y)
    print("DNN Model Results:")
    results(resultsDNN)
    print("CNN Model Results:")
    results(resultsCNN)
    print("SKL KNN Model Results:")
    results(resultsSKL_KNN)
    print()
    print("My KNN Model Results:")
    results(myKNN(x_train, y_train, x_test, y_test, knn_value))
    return
        
def load_prev_models(x_train, x_test, y_train, y_test, x, y):
    #KNN MODEL LOAD:
    print("==================================================================")
    print("Loading SKL Knn Model...")
    skl_knn_model_load = load("skl_knn_model.joblib")
    knn_model_load = load("my_knn_model.joblib")
    print()
    print("=============== SKL KNN Model Classifier ===============")
    print("Accuracy: ", str((skl_knn_model_load.score(x_test, y_test))*100), "%")
    print("Loss:", 1-(skl_knn_model_load.score(x_test, y_test)))
    print()
    print(skl_knn_model_load)
    print()
    print("=============== MY KNN Model Classifier ===============")
    print(knn_model_load)
    print()
    # DNN MODEL LOAD:
    print("=============== DNN Model Loaded =============== ")
    dnn_model_load = load_model('dnn_model.h5')
    print("DNN Load Summary:")
    dnn_model_load.summary()
    print("DNN Weights:")
    dnn_model_load.get_weights()
    print("DNN Optimizer:")
    dnn_model_load.optimizer
    print(dnn_model_load)
    print()
    #Load section for DNN Model as a code comment - error due to dimensionality.
    '''score = dnn_model_load.evaluate(x_train, x_test)
    #print(score + "%")'''
    print()
    #CNN MODEL LOAD:
    print("=============== CNN Model Loaded =============== ")
    cnn_model_load = load_model('cnn_model.h5')
    print("CNN Load Summary:")
    cnn_model_load.summary()
    print("CNN Weights:")
    cnn_model_load.get_weights()
    print("CNN Optimizer:")
    cnn_model_load.optimizer
    print()
    print(cnn_model_load)
    #Load section for CNN Model as a code comment - error due to dimensionality.
    '''scoreCNN = cnn_model_load.evaluate(x_train, x_test)
    #print(scoreCNN + "%")'''
    print("==================================================================")
    return
    
# ========================= Assignment 1 attachment ==========================

# Assignment 1 - Using KNN-SKLearn libraries.
def knn_skl(x_train, x_test, y_train, y_test, x, y):
    global skl_knn_pred
    print("SKL KNN-Model running...")
    #Use SKLearn libraries built in - for the SKLearn model. KNN.
    knnCLF = KNeighborsClassifier(n_neighbors = 5).fit(x_train, y_train)
    knn_pred = knnCLF.predict(x_test)
    skl_knn_pred = knn_pred
    skl_knn_accuracy = accuracy_score(y_test, knn_pred) #Accuracy of the training data sklmodel
    dump(knnCLF, "skl_knn_model.joblib")
    print("SKL KNN-Model finished.")
    #Return results of KNN accuracy to print.
    return skl_knn_accuracy

#My KNN Algorithm without SKLearn
def myKNN(x_train, y_train, x_test, y_test, k):
    print("My KNN-Model running...")
    #storing all predicted distances in an array.
    global allpredictions
    allpredictions = [] 
    #predicting the numeber closest to a class
    for i in range(len(x_test)):
        #Add predicted value to the allpredictions array.
        allpredictions.append(prediction(x_train, y_train, x_test[i, :], k))
    #knn accuracy of the score to the data.
    KNNaccuracy = accuracy_score(y_test, allpredictions)
    dump(myKNN, "my_knn_model.joblib")
    print("My KNN-Model finished.")
    #Return results of KNN accuracy to print.
    return KNNaccuracy

#prediction via neighbors and euclidean distance
def prediction(x_train, y_train, x_test, k):
    #Initialise variable arrays for Euclidean distance and target.
    ed = [] 
    target = []
    for i in range(len(x_train)):
        # sqrt all values in the array and calculating the euclidean distance and adding it to the array.
        ed.append([np.sqrt(np.sum(np.square(x_test - x_train[i, :]))), i])
    ed = sorted(ed)
    #getting the neighbors using for loop.
    for i in range(k): 
        element = ed[i][1]
        target.append(y_train[element])
    vote = Counter(target).most_common(1)[0][0]
    return vote

#Start
main()

