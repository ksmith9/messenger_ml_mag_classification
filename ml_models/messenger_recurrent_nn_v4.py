import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
from pathlib import Path
import random
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import os
import datetime
from tqdm import tqdm
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import pickle

tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(10)

print("Starting...")

start_time = datetime.datetime.now()

def data_import( data_path, dimensions_path):
    data = pd.read_parquet(data_path).to_numpy()
    with open(dimensions_path, 'rb') as f: 
            return data.reshape(pickle.load(f))


print("Importing training data...\n")
data_directory = Path("../data")

training_features = data_import(data_directory / Path("training_features_5.parquet"), data_directory / Path("training_features_5_dimensions.pickle"))
training_targets = pd.read_parquet( data_directory / Path("training_targets_5.parquet")).to_numpy().astype(np.uint8)

print("Importing validation data...\n")
validation_features = data_import(data_directory / Path("validation_features_5.parquet"), data_directory / Path("validation_features_5_dimensions.pickle"))
validation_targets = pd.read_parquet( data_directory / Path("validation_targets_5.parquet")).to_numpy().astype(np.uint8)



#model parameters
number_nodes = [64, 128, 256]
learning_rates = [1e-3, 1e-2,1e-1]
dropouts = [0.1, 0.2, 0.3]
batch_size = 4096
output_size = 3
epochs = 100 
batches_per_epoch = int(len(training_targets)/batch_size)

class_weight = {0:np.count_nonzero(training_targets == 0) / len(training_targets), 
        1:np.count_nonzero(training_targets == 1) / len(training_targets),
        2:np.count_nonzero(training_targets == 2) / len(training_targets)}
print(class_weight)


print("Building model...\n")
#Defining Constants/building model

def build_model(nodes, lr, drop):
    model = Sequential()
    
    name = f"no_buffer_MESSENGER_LSTM_{nodes}_{lr}_{drop}"
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    
    tensorboard = TensorBoard(log_dir = f'rnnlogs/{time}_{name}')
    for dropout in drop: 
        for learning_rate in lr:
            for node_size in nodes: 
                
                model.add(keras.layers.RNN(
                    keras.layers.LSTMCell(256),
                        return_sequences = True,
                    input_shape=(300, 3)))
                
                
                model.add(keras.layers.Dropout(dropout))
                
                model.add(keras.layers.RNN(
                    keras.layers.LSTMCell(256),
                        return_sequences = False,
                    input_shape=(300, 3)))
                model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(output_size, activation = tf.nn.softmax))
    
    lr_decay = 0.01*(1/0.95 - 1)/batches_per_epoch
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-4, decay = lr_decay, clipnorm = 5.0)#, global_clipnorm = 5.0)
    checkpoint_path = f"{time}_{name}_(no_buffer)/cp.ckpt"
    

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = 10)

    my_callbacks = [tensorboard, 
            cp_callback, 
            es_callback]
    
    model.compile(
     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     optimizer=opt,
     metrics=["accuracy"],
    )
    return model, my_callbacks, name

model, my_callbacks, name = build_model([number_nodes[0]], [learning_rates[1]], [dropouts[0]])




print(f"Training Model {name} \n")

model.fit(training_features, 
    training_targets, 
     batch_size = batch_size, 
     validation_data = (validation_features, validation_targets), 
     class_weight = class_weight,
     epochs = epochs,
     callbacks = my_callbacks)

print("Saving trained model...\n")
model.save(f"LSTM_1024_{time}")


print("Finished! Time taken:", datetime.datetime.now()-start_time)
