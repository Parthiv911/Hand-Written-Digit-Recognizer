import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np
import math

digits = datasets.load_digits()

images_and_labels=list(zip(digits.images,digits.target))

data=digits.images.reshape(-1,64)

classifier=svm.SVC(gamma=0.001)

train_test_split=int(len(digits.images)*0.75)
classifier.fit(data[:train_test_split],digits.target[:train_test_split])

expected= digits.target[train_test_split:]
predicted = classifier.predict(data[train_test_split:])

print("Confusion matrix and Accuracy Score after testing:")
print("Confusion matrix:")
print(metrics.confusion_matrix(expected,predicted))
print("Accuracy Score: ",accuracy_score(expected,predicted))

#reading image to be classified
img=cv2.imread(r'C:\Users\ASUS\OneDrive\Desktop\MLDATA\two3.jpg',0)
img=255-img

print("\nInput grayscale image (8x8):")
print(img)

img2=img.reshape(-1,64)

print("Predicted digit: ",classifier.predict(img2))
plt.imshow(img,cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()