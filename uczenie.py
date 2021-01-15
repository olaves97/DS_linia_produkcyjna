from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import random
import cv2
from tensorflow.keras import layers
import numpy as np
import os
from keras import models



def data_loading():
    #Global variables
    global data,labels

    #Enter the path of your image data folder
    image_data_folder_path = "C:/Users/Dawid/Desktop/baza_danych_augmentacja"

    # initialize the data and labels as an empty list
    #we will reshape the image data and append it in the list-data
    #we will encode the image labels and append it in the list-labels
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(image_data_folder_path)))

    #total number images
    total_number_of_images = len(imagePaths)
    print("\n")
    print("Total number of images----->",total_number_of_images)

    #randomly shuffle all the image file name
    random.shuffle(imagePaths)

    print("Data processing...")
    # loop over the shuffled input images
    for imagePath in imagePaths:

        #Read the image into a numpy array using opencv
        #all the read images are of different shapes
        image = cv2.imread(imagePath)

        #resize the image to be 32x32 pixels (ignoring aspect ratio)
        #After reshape size of all the images will become 32x32x3
        #Total number of pixels in every image = 32x32x3=3072
        image = cv2.resize(image, (150, 150))

        #flatten converts every 3D image (32x32x3) into 1D numpy array of shape (3072,)
        #(3072,) is the shape of the flatten image
        #(3072,) shape means 3072 columns and 1 row
        #mage_flatten = image.flatten()

        #Append each image data 1D array to the data list
        data.append(image)

        # extract the class label from the image path and update the
        label = imagePath.split(os.path.sep)[-2]

        #Append each image label to the labels list
        labels.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    #convert the data and label list to numpy array
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("Data processing finished")
def data_split():

    global trainX,testX,trainY,testY

    print("Data splitting...")

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    # train_test_split is a scikit-learn's function which helps us to split train and test images kept in the same folders
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    print("Number of training images--->", len(trainX), ",", "Number of training labels--->", len(trainY))
    print("Number of testing images--->", len(testX), ",", "Number of testing labels--->", len(testY))

    # convert the labels from integers to vectors
    # perform One hot encoding of all the labels using scikit-learn's function LabelBinarizer
    # LabelBinarizer fit_transform finds all the labels

    lb = preprocessing.LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    #trainY = to_categorical(trainY)
    #testY = to_categorical(testY)

    print("\n")
    print("Classes found to train", )
    train_classes = lb.classes_
    print(train_classes)
    binary_rep_each_class = lb.transform(train_classes)
    print("Binary representation of each class")
    print(binary_rep_each_class)
    print("\n")
def plot():
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.title("Porównanie dokładności rozpoznania zbioru testowego i walidacyjnego")
    plt.xlabel("Epoka#")
    plt.ylabel("Dokładność [%]")
    plt.legend()
    plt.show()

data_loading()
data_split()

EPOCHS = 15

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())#model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation = 'relu',input_shape=(150, 150, 3)))
model.add(layers.Dense(30, activation = 'relu'))
model.add(layers.Dense(3, activation = 'softmax'))
model.summary()



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=500)

plot()

model.save('hammer_screwdriver_4.h5')