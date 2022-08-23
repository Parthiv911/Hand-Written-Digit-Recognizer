import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np
import math
import pandas as pd

#import data sets for train and test
digits_data_train = pd.read_csv(r"optdigits.tra")
digits_data_test = pd.read_csv(r"optdigits.tes")

#split train data set into attributes and corresponding class
digits_data_train_y = digits_data_train.iloc[:,-1]
digits_data_train_x = digits_data_train.iloc[:,0:-1]

#split test data set into attributes and corresponding class

n = int(len(digits_data_test)/2)
digits_data_test_y = digits_data_test.iloc[:n,-1]
digits_data_test_x = digits_data_test.iloc[:n,0:-1]

#splitting to form the cross validation data set
digits_data_crossvalidate_y = digits_data_test.iloc[n:,-1]
digits_data_crossvalidate_x = digits_data_test.iloc[n:,0:-1]

#reshape test and train attributes to 64 long list for each image
digits_data_train_x = digits_data_train_x.values.reshape(-1,64)
digits_data_test_x = digits_data_test_x.values.reshape(-1,64)

classifier=svm.SVC(kernel='rbf',gamma=0.001,C=1)

#train the model
classifier.fit(digits_data_train_x,digits_data_train_y)

#test the model with cross validation
expected = digits_data_crossvalidate_y
predicted = classifier.predict(digits_data_crossvalidate_x)

#analyze the confusion matrix and accuracy score after cross validation
print("Confusion matrix and Accuracy Score after cross validation:")
print("Confusion matrix:")
confusion=metrics.confusion_matrix(expected,predicted)
labels = ['0', '1', '2', '3','4','5','6','7','8','9']
df = pd.DataFrame(confusion, columns=labels, index=labels)

print(df)
print("Accuracy Score: ",accuracy_score(expected,predicted))

#test the model
expected = digits_data_test_y
predicted = classifier.predict(digits_data_test_x)

#analyze the confusion matrix and accuracy score after testing
print("Confusion matrix and Accuracy Score after testing:")
print("Confusion matrix:")
confusion=metrics.confusion_matrix(expected,predicted)
labels = ['0', '1', '2', '3','4','5','6','7','8','9']
df = pd.DataFrame(confusion, columns=labels, index=labels)

print(df)
print("Accuracy Score: ",accuracy_score(expected,predicted))

#
# Predict hand written image
#

#reading image to be classified
img=cv2.imread(r'C:\Users\USER\Desktop\Hand-Written-Digit-Recognizer-main\two3.jpg',0)
img=255-img

#printing image in matrix form
print("\nInput grayscale image (8x8):")
print(img)

#reshaping matrix to array of length 64
img2=img.reshape(-1,64)

#identifying the digit in image using the model
print("Predicted digit: ",classifier.predict(img2))

#printing image
plt.imshow(img,cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()
