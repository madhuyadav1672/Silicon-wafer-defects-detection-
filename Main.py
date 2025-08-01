from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import cv2
import pickle
import os
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model, Model
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.applications import VGG16
from keras.layers import Convolution2D
from keras.layers import  MaxPooling2D

main = tkinter.Tk()
main.title("Silicon Wafer Defect Detection using Machine Learning Techniques")
main.geometry("1300x1200")

global filename, labels
global X_train, y_train, X_test, y_test, labels, X, Y, vgg_model

def getLabel(name):
    global labels
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    text.delete('1.0', END)
    global filename, dataset, labels, X, Y
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):        
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))
                    X.append(img)
                    label = getLabel(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)                    
    text.insert(END,"Dataset Loading Completed\n")
    text.insert(END,"Total images found in dataset = "+str(X.shape[0])+"\n")
    text.insert(END,"Various Silicon Wafer Defects found in Dataset : "+str(labels)+"\n\n")
    label, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (6, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def imagePreprocessing():
    global X, Y
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Dataset Shuffling & Normalization Completed")

def splitDataset():
    global X, Y
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"Dataset Train & Test Split Details\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    
    
def runModel():
    global X_train, y_train, X_test, y_test
    global vgg_model
    text.delete('1.0', END)
    vgg_model = VGG16(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in vgg_model.layers:
        layer.trainable = False
    vgg_model = Sequential()
    vgg_model.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    vgg_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    vgg_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg_model.add(Flatten())
    vgg_model.add(Dense(units = 256, activation = 'relu'))
    vgg_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    vgg_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vgg_model.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vgg_model.load_weights("model/cnn_weights.hdf5")
    predict = vgg_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("CNN with Pre-Trained VGG16", y_test1, predict)

def graph():
    f = open('model/cnn_history.pckl', 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values['accuracy']
    loss_value = train_values['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(accuracy_value, 'ro-', color = 'green')
    plt.plot(loss_value, 'ro-', color = 'blue')
    plt.legend(['Training Accuracy', 'Training Loss'], loc='upper left')
    plt.title('Transfer Learning VGG16 with CNN Training Accuracy & Loss Graph')
    plt.show()

def predict():
    global vgg_model, labels
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32, 32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = vgg_model.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (400,300))
    cv2.putText(img, 'Defect Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow('Defect Predicted As : '+labels[predict], img)
    cv2.waitKey(0)            

font = ('times', 16, 'bold')
title = Label(main, text='Silicon Wafer Defect Detection using Machine Learning Techniques',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Silicon Wafer Defect Dataset", command=uploadDataset)
upload.place(x=10,y=500)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=400,y=500)

preprocessButton = Button(main, text="Preprocess Dataset", command=imagePreprocessing)
preprocessButton.place(x=10,y=550)
preprocessButton.config(font=font1)

splitButton = Button(main, text="Train & Test Split", command=splitDataset)
splitButton.place(x=330,y=550)
splitButton.config(font=font1)

vggButton = Button(main, text="Run CNN Transfer Learning with VGG16", command=runModel)
vggButton.place(x=660,y=550)
vggButton.config(font=font1)

graphButton = Button(main, text="Training Graph", command=graph)
graphButton.place(x=10,y=600)
graphButton.config(font=font1)

predictButton = Button(main, text="Defect Detection from Test Image", command=predict)
predictButton.place(x=330,y=600)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='light salmon')
main.mainloop()
