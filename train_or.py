# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, ZeroPadding2D
from keras.layers.core import Dense
from keras.optimizers import Adam
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
import datetime as dt
import random
import pickle
import cv2
import os
import matplotlib

matplotlib.use("Agg")

# initialize the data and labels
data, labels = [], []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("PlantVillage")))
random.seed(42)
random.shuffle(imagePaths)


print("[INFO] loading images...")
# loop over the input images
i = 0
for imagePath in imagePaths:
    i = i + 1
    if np.mod(i, 100) == 0:
        print("[INFO] loading... "+str(i)+"/"+str(len(imagePaths))+" images")

    # data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    data.append(image)

    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

print("[INFO] loading " + str(np.size(imagePaths)) + " images")

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initializing
model = Sequential()

model.add(Conv2D(16, (4, 4), input_shape=(64, 64, 3), activation='relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(16, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(lb.classes_), activation="softmax"))

EPOCHS = 20

print("[INFO] model summary...")
model.summary()
print("[INFO] training network...")
opt = Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32, verbose=2)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
rp = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
print(rp)

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_accuracy")
plt.plot(H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("or_plot_acc&loss.png")

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save("or_model.model")
f = open("or_lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()