# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt

# load (downloaded if needed) the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()




#IMPORT CLASSES
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

#INITIALIZE VARIABLES
# set batch and epoch sizes
batch_size = 200
epochs = 10

# fix random seed for reproducibility
seed = 93
numpy.random.seed(seed)

# input image dimensions
img_rows, img_cols = 28, 28

print("Variables intialized")



#RESHAPE DATA
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols).astype('float32')
input_shape = (1, img_rows, img_cols)

# normalize inputs from 0-255 to 0-1
X_train /= 255
X_test  /= 255

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

from keras import models
from keras.models import load_model

model = load_model('w1_data.hdf5')

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(scores)


print('Test Loss:     %.2f%%' % (scores[0]))
print('Test Accuracy: {:.2%}'.format(scores[1]))
print('Test Error:    {:.2%}'.format(1-scores[1]))

from keras.preprocessing import image
from tkinter import *
import tkinter.scrolledtext as st
import tkinter.filedialog as fd
import tkinter.messagebox as mbox
root = Tk()
root.title('MNIST')
root.resizable(0,0)
root.configure(background = 'light green')

var1 = IntVar()
photo = PhotoImage(file = 'back_image.gif')
w = Label(root, image = photo)
w.grid(row = 0, column = 0, columnspan = 7, padx = 4, pady = 4)

def Predict():
    text1.delete(1.0, END)
    root.filename = fd.askopenfilename(initialdir="/", title="Select file")
    test_image = image.load_img(root.filename,grayscale=True, target_size = (28, 28))
    test_image = image.img_to_array(test_image)
    test_image = numpy.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    a = result[0]
    a = a.tolist()
    a = a.index(1)
    text1.insert(1.0, a)
    




def Exit():
    m = mbox.askyesno(title = 'Quit', message = 'Are you sure?')
    if m:
        root.destroy()

def about_file():
    mbox._show(title = 'OJO', message = 'This program is used to predict number from an image.')



#MENU
mymenu = Menu(root)
file_menu = Menu(mymenu, tearoff = 0)
mymenu.add_cascade(label = 'File', menu = file_menu)
file_menu.add_command(label = 'Exit', command = Exit)
about_menu = Menu(mymenu, tearoff = 0)
mymenu.add_cascade(label = 'About', menu = about_menu)
about_menu.add_command(label = 'About', command = about_file)
root.config(menu = mymenu)



n = 'helvetica', 15
label = Label(text = '   Click on Predict to load image:', font = n, bg = 'light green').grid(row = 1, column = 0, sticky = 'e')
predict_button = Button(text = 'Predict', width = 10, height = 5, command = Predict, font = n ).grid(row = 2, column = 0, padx = 10, pady = 10, sticky = 'w')
text1 = st.ScrolledText(root, width = 4, height = 2, wrap = 'word', font = n, bg = 'white')
text1.grid(row = 2, column = 2, padx = 8)
root.mainloop()
