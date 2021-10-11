import numpy as np
import pandas as pd
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Dropout
import tensorflow as tf
import datetime
import sklearn as sk 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sys import platform


# ============================ IMPORTING AND FORMATING DATA =======================================


#importing data
dir_path = r"/home/kdudesmith/Dropbox/College/Jackman_Research/data_prep/"
df = pd.read_csv(rdir_path+"discontinusous_tansitionless_class_balanced_random_sample.csv")

# creating test/train split and removing unnessecary features
X_train, X_test, y_train, y_test = train_test_split( 
    df.drop(["Location", "OrbitNo", "Date", "X_MSO", "Y_MSO", "Z_MSO"], axis = 1), 
    df["Location"], 
    test_size=0.33, random_state=42)
del df

X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')




#====================== CREATING NEURAL NETWORK =================================



# Neural Network Architecture Specification 

layer_sizes = [128]
learning_rates = [0.0001]

#creating neural network with 1 hidden layer of size 128
model = Sequential() 
model.add(Flatten())

model.add(Dense(layer_size, activation = tf.nn.relu))
model.add(Dropout(0.2))

model.add(Dense(3, activation=tf.nn.softmax))

opt = tf.keras.optimizers.Adam(lr = learning_rate, decay = 1e-5)


model.compile(optimizer=opt,
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])


# ======================================== TRAINING MODEL ======================================
model.fit(X_train, y_train, batch_size = 32, validation_split = 0.15, epochs = 10)



# ================================================== EVALUATING MODEL ===============================

# printing overall acuracy/loss
X_test = np.asarray(X_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

val_loss, val_acc = model.evaluate(X_test, y_test)

print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)



#Creating Confusion Matrix
y_pred = model.predict(X_test) 
y_pred = np.argmax(y_pred, axis = 1)


messenger_confussion_matrix = confusion_matrix(y_pred, y_test)



cmd_obj = ConfusionMatrixDisplay(messenger_confussion_matrix, display_labels=["Magnetosphere", "Magnetosheath", "Solar Wind"])


cmd_obj.plot()

cmd_obj.ax_.set(
                title='Confusion Matrix for Feed Forward Neural Network Classifier', 
                xlabel='Predicted Location', 
                ylabel='Actual Location')


plt.show()

