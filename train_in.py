# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import sklearn.model_selection
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import EarlyStopping
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

def create_model(classes_size=2):
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
    model.add(Dense(classes_size, activation="softmax", name="out"))

    return model

def update_model(model, classes_size, new_weights_initializer='average'):
    
    previous_weights = model.get_layer("out").get_weights()
    # remove last layer
    model.pop()
    # add new output_layer
    predict_layer = Dense(classes_size, activation='softmax', name="out")
    model.add(predict_layer)

    weights_new = model.layers[-1].get_weights()

    # copy the original weights back
    weights_new[0][:, :-1] = previous_weights[0]
    weights_new[1][:-1] = previous_weights[1]

    if new_weights_initializer == 'average':
        # use the average weight to init the new class.
        weights_new[0][:, -1] = np.mean(previous_weights[0], axis=1)
        weights_new[1][-1] = np.mean(previous_weights[1])

    model.layers[-1].set_weights(weights_new)
    
    return model

def reduce_class_data(trainX, trainY, current_label, labels, model, data_size=0.5, data_selector='random'):
    # predict traning data
    predictions = model.predict(trainX)
    # fetch data for the current label or class$
    x_true, y_pred = [], []
    for j in range(len(trainX)):
        if trainY[j, current_label] == 1:
            x_true.append(trainX[j])
            y_pred.append(predictions[j])
    x_y = list(zip(x_true, y_pred))

    data_size = int(data_size * len(x_true))

    if data_selector =='random':
        # choose data_size samples
        selected_data = random.sample(x_y, data_size)
        selected_data = np.asarray(selected_data)[:, 0]
    elif data_selector =='bad':
        # sort by bad
        selected_data = sorted(x_y, key=lambda sample: sample[1][current_label], reverse=False)
        selected_data = np.asarray(selected_data)
        # choose first data_size samples
        selected_data = selected_data[0:data_size, 0]
    elif data_selector == 'best':
        # sort by best
        selected_data = sorted(x_y, key=lambda sample: sample[1][current_label], reverse=True)
        selected_data = np.asarray(selected_data)
        # choose first data_size samples
        selected_data = selected_data[0:data_size, 0]

    print("[Info] from "+labels[current_label]+" : selecting "+str(data_size)+"/"+str(len(x_y))+" samples for incremental learning...")
    data, labs = [], []
    for sample in selected_data:
        data.append(sample)
        labs.append(labels[current_label])
    return data, labs

# main
previous_trainX, previous_trainY, previous_testX, previous_testY = [], [], [], []
data_list, labels_list = [], []
loss, val_loss, accuracy, val_accuracy = [], [], [], []

dataSet_path = 'PlantVillage'
EPOCHS = 20
batch_size = 32
opt = Adam(lr=0.001)
model = Model
lb = LabelBinarizer()
folders = os.listdir(dataSet_path)
num_labels = len(folders)

for folder, current_class in zip(folders, range(num_labels)):
    print('[Info] loading images from "' + folder + '".........')
    images_path = sorted(list(paths.list_images(dataSet_path + "/" + folder)))
    for image_path in images_path:
        # data_ordinaire list
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))
        data_list.append(image)

        # labels list
        labels_list.append(folder)

    print('[Info] loading from        "' + folder + '" ' + str(np.size(images_path)) + ' images')

    if current_class == 1:
        print('[Info] training network for 2 classes....')
        data_list = np.array(data_list, dtype="float") / 255.0
        labels_list = np.array(labels_list)
        # split data
        (trainX, testX, trainY, testY) = train_test_split(data_list, labels_list, test_size=0.25, random_state=42)
        data_list, labels_list = [], []

        previous_testX += np.array(testX).tolist()
        previous_testY += np.array(testY).tolist()

        trainY = lb.fit_transform(trainY)
        trainY = np.hstack((1 - trainY, trainY))
        testY = lb.transform(testY)
        testY = np.hstack((1 - testY, testY))

        model = create_model(classes_size=len(lb.classes_))
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # train the neural network
        H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=batch_size, verbose=2)
        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX)
        rp = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
        print(rp)
        previous_data_1, previous_labels_1 = reduce_class_data(trainX=trainX, trainY=trainY, current_label=0,
                                                               labels=folders, model=model)
        previous_data_2, previous_labels_2 = reduce_class_data(trainX=trainX, trainY=trainY, current_label=1,
                                                               labels=folders, model=model)
        previous_trainX = previous_data_1 + previous_data_2
        previous_trainY = previous_labels_1 + previous_labels_2
        previous_data_1, previous_labels_1, previous_data_2, previous_labels_2 = [], [], [], []

        loss.append(H.history['loss'][len(H.history['loss'])-1])
        accuracy.append(H.history['accuracy'][len(H.history['accuracy']) - 1])
        val_loss.append(H.history['val_loss'][len(H.history['val_loss'])-1])
        val_accuracy.append(H.history['val_accuracy'][len(H.history['val_accuracy']) - 1])

    if current_class >= 2:
            print('[Info] training network for '+str(current_class+1)+' classes....')
            data_list = np.array(data_list, dtype="float") / 255.0
            labels_list = np.array(labels_list)
            # split data and labels for the class
            (trainX, testX, trainY, testY) = train_test_split(data_list, labels_list, test_size=0.25, random_state=42)
            data_list, labels_list = [], []
            # combination of previous and new data
            trainX = np.append(np.array(previous_testX), trainX, axis=0)
            trainY = np.append(np.array(previous_testY), trainY, axis=0)
            previous_testX += np.array(testX).tolist()
            previous_testY += np.array(testY).tolist()

            testX = np.array(previous_testX)
            trainY = lb.fit_transform(trainY)
            testY = lb.transform(np.array(previous_testY))

            # Shuffle the order
            shuf = list(zip(trainX, trainY))
            random.shuffle(shuf)
            trainX, trainY = zip(*shuf)
            shuf = []
            trainX = np.asarray(trainX)
            trainY = np.asarray(trainY)

            # update model
            model = update_model(model=model, classes_size=len(lb.classes_), new_weights_initializer='average')
            # train the neural network
            model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
            H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=batch_size, verbose=2)
            # evaluate the network
            print("[INFO] evaluating network...")
            predictions = model.predict(np.array(testX), batch_size=32)
            rp = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
            print(rp)
            if current_class < num_labels-1:
                previous_data, previous_labels = reduce_class_data(trainX=trainX, trainY=trainY, current_label=current_class,
                                                                   labels=folders, model=model)
                previous_trainX += previous_data
                previous_trainY += previous_labels
                previous_data, previous_labels = [], []

            loss.append(H.history['loss'][len(H.history['loss'])-1])
            accuracy.append(H.history['accuracy'][len(H.history['accuracy']) - 1])
            val_loss.append(H.history['val_loss'][len(H.history['val_loss'])-1])
            val_accuracy.append(H.history['val_accuracy'][len(H.history['val_accuracy']) - 1])

# create new dir
day = dt.datetime.now().strftime('%Y-%m-%d_%H-%M')
dirName = "output_in/"+str(day)
os.mkdir(dirName)
copyfile("train_in.py", dirName+"/train_in.py")

# plot the training loss and accuracy
print(loss)
print(val_loss)
print(accuracy)
print(val_loss)
plt.style.use("ggplot")
plt.figure()
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.plot(accuracy, label="train_accuracy")
plt.plot(val_accuracy, label="val_accuracy")
plt.title("Training/Validation Loss and Accuracy")
plt.xlabel("N+2 classes")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(dirName+"/plot.png")
