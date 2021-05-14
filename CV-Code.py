
#Image size
im_size = 40

#Number of iterations
iterations = 20


#Percentage of testing data
test_data = 0.3


print("Started!")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
import tensorflow as tf
import cv2



path=os.getcwd()
path=path+"/Files/"
path=path.replace('\\', '/')
print (path)
CATEGORIES = next(os.walk(path))[1]
print(CATEGORIES)


training_data = []
label_data = []
for inx,ima in enumerate(CATEGORIES):
    files=next(os.walk(path+ima))[2]

    for file in files:

        this_image = path+""+ima+"/"+file

        print("Image: ",this_image)
        try:
            #im=Image.open(this_image)
            im_array = cv2.imread(this_image,cv2.IMREAD_GRAYSCALE)
            #im_array = cv2.imread(this_image)
            im=cv2.resize(im_array,(im_size, im_size))
            im=np.array(im)
            #print(im)
            training_data.append([im, inx])
            label_data.append([inx,ima])
        except Exception as e:
            pass



print("#############\n\nCompleted.\n\n\n")

import pandas as pd
label_data = pd.DataFrame(label_data)
label_data = label_data[1].unique()
import random
random.shuffle(training_data)
X = []
Y = []
for features,label in training_data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1, im_size, im_size,1)
X=X.astype('float32')
X=X/255.0
########################################

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

#Build the CNN model
model=Sequential()
model.add(Conv2D(64,(3,3), strides=1,input_shape=[im_size, im_size, 1], activation='relu'))

model.add(Dropout(0.1))
model.add(Conv2D(im_size,(2,2),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(300,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.1))

model.add(Dense(300,activation='relu',kernel_constraint=maxnorm(3)))

model.add(Dense(100,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dense(len(label_data), activation='sigmoid'))
##########################################################


#Set the parameters of the CNN model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


#Reshape the features and labels to fit into the CNN model
X = np.asarray(X).astype('float32').reshape((-1,im_size,im_size,1))
Y = np.asarray(Y).astype('float32').reshape((-1,1))


#Store the best model in a file
file_path= "model3.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                           monitor='val_accuracy',
                                                           mode='max',
                                                           save_best_only=True)



#Start the model training
model.fit(X, Y, epochs=iterations, callbacks=[model_checkpoint_callback],verbose=1, validation_split=test_data)
##############################################################





#GUI code
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy

#load the trained model to classify the images

from keras.models import load_model
model = load_model('model3.h5')

#initialise GUI

top=tk.Tk()
top.geometry('800x600')
top.title('Image Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed


    im = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    im=cv2.resize(im,(im_size, im_size))
    #print(im.shape)
    im = np.asarray(im).astype('float32').reshape((im_size,im_size,1))

    im=im/255.0
    im = np.array(im).reshape(-1, im_size, im_size,1)
    pred=model.predict_classes(im)

    pred  = pred.item()
    sign  = label_data[pred]
    #proba = np.array(100*model.predict_proba(im)).astype("float64")
    #proba = np.around(proba, decimals=2)
    print("\n-----------------------\nPredicted image: ",sign)
    #print("Probability of other classes: ", proba)
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",
   command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)



def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Choose an image",command=upload_image,
  padx=10,pady=5)

upload.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Image Classification",pady=20, font=('arial',20,'bold'))

heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
