#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from itertools import combinations_with_replacement

#USER DEFINIED VARIABLES

#input files 
data_directory = Path("../data")

data_file_train = "train_list.parquet"
random_time_samples_train = "train_list_transitionless_class_balanced_random_sample.parquet"

data_file_validation = "validation_list.parquet"
random_time_samples_validation = "validation_list_transitionless_class_balanced_random_sample.parquet"

#neural netork parameters
possible_layer_sizes = [32, 64, 128]

number_of_regions = 3 

training_sample_sizes = 0.1
batch_size = 64

                

#FUNCTIONS

#Data import and prepocessing function, including normalisation and removal of unessecary columns

def data_import_and_processor(data_path): 
    df = pd.read_parquet(data_path)
    
    df.drop(df.index.where(df.index > "2014-03-16").dropna(), inplace = True) #THIS LINE IS ONLY NESSECARY WHILE WEIJIE'S LIST IS INCOMPLETE
    
    df.drop(["X_MSO", "Y_MSO","Z_MSO", "OrbitNo" ], axis = 1, inplace = True)
    
    #NORMALISING DATA
    transformer = RobustScaler().fit(df["B_X_MSO"].to_numpy().reshape(-1, 1))
    df["B_X_MSO"] = transformer.transform(df["B_X_MSO"].to_numpy().reshape(-1, 1))

    transformer = RobustScaler().fit(df["B_Y_MSO"].to_numpy().reshape(-1, 1))
    df["B_Y_MSO"] = transformer.transform(df["B_Y_MSO"].to_numpy().reshape(-1, 1))

    transformer = RobustScaler().fit(df["B_Z_MSO"].to_numpy().reshape(-1, 1))
    df["B_Z_MSO"] = transformer.transform(df["B_Z_MSO"].to_numpy().reshape(-1, 1))
    
    #Preparing output variables
    df_index = df.index
    
    df_arr = df[["B_X_MSO","B_Y_MSO","B_Z_MSO"]].to_numpy()

    df_target = df["Location"].astype("uint8").to_numpy()

    total_data_size = len(df)

    return df_index, df_arr, df_target, total_data_size


#function to create sequences to be fed into the RNN

def sequencer(time_series, df_index, df_arr, df_target, total_data_size, x):
    time_series = time_series.index
    seq = 0
    features = []
    targets = []
    
    while x < total_data_size-150 and seq < len(time_series):
        if time_series[seq] == df_index[x]:
            features.append(df_arr[x-150:x+150])
            targets.append(df_target[x+150])
            seq += 1
            x += 1
    
            if df_target[x+150] == df_target[x]: 
                pass
            else: 
                exit("Sequencing Failed, for training the region classifier all sequences must be located within the same region.")
                break
        else:
            x += 1
            
    return features, targets, x
    


#function to initialise rnn model architecture 
def build_model(number_of_lstm_layers, layer_sizes):
    model = keras.models.Sequential()
    for layer_number in range(0, number_of_lstm_layers):
        
        if layer_number != number_of_lstm_layers -1 :  
            lstm_layer = keras.layers.RNN(
                keras.layers.LSTMCell(layer_sizes[layer_number]),
                input_shape=(300, 3), return_sequences = True)
            
            model.add(lstm_layer)
            model.add(keras.layers.Dropout(0.25))

        else: 
            lstm_layer = keras.layers.RNN(
                keras.layers.LSTMCell(layer_sizes[layer_number]),
                input_shape=(300, 3), return_sequences = False)
            
            model.add(lstm_layer)
            model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Dense(number_of_regions))
    return model
    
    


# Function to train rnn

def train_model(batch, bookmark_train, bookmark_validation): 
    X_train, y_train, bookmark_train = sequencer(
            train_times.iloc[int(len(train_times)*batch):int(len(train_times)*(batch+training_sample_sizes))], 
            train_df_index,
            train_df_arr,
            train_df_target, 
            train_total_data_size,
            bookmark_train)     

    y_train = np.array(y_train).astype('uint8')
    X_train = np.array(X_train)

    X_validation, y_validation, bookmark_validation = sequencer(
            validation_times.iloc[int(len(validation_times)*batch):int(len(validation_times)*(batch+training_sample_sizes))],
            validation_df_index, 
            validation_df_arr, 
            validation_df_target, 
            validation_total_data_size,
            bookmark_validation)

    y_validation = np.array(y_validation).astype('uint8')
    X_validation = np.array(X_validation)

    model.fit(X_train, y_train,
       validation_data = (X_validation, y_validation), 
        batch_size=batch_size,
        epochs=5,
        callbacks=[cp_callback])

    return bookmark_train, bookmark_validation
            
            


#data import:



print("Importing Training Data...")
train_df_index, train_df_arr, train_df_target, train_total_data_size = data_import_and_processor(data_directory / data_file_train)

train_times = pd.read_parquet(data_directory / random_time_samples_train )

train_times.index = train_times.Date
train_times.drop("Date", axis=1, inplace = True)



print("Importing Validation Data")
validation_df_index, validation_df_arr, validation_df_target, validation_total_data_size = data_import_and_processor(data_directory / data_file_validation)

validation_times = pd.read_parquet(data_directory / random_time_samples_validation )
validation_times.index = validation_times.Date
validation_times.drop("Date", axis=1, inplace = True)





# training rnn and saving output 
for size in range(1, len(possible_layer_sizes)+1): 
    for unit in list(combinations_with_replacement(possible_layer_sizes, size)):
    
        #creating rnn 
        
        name = f"MESSENGER_RNN_{size}x{unit}"
        tensorboard = TensorBoard(log_dir=f"rnn_logs_/{name}")


        model = build_model(size, unit)
        
        opt = tf.keras.optimizers.Adam(learning_rate = 1e-3, decay = 1e-5)
        
        
        checkpoint_path = f"training_1/cp_{unit}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=opt,
            metrics=["accuracy"],
        )

        initial = True
        

        print("Model initialised. Commencing training.")
        for batch in tqdm(np.arange(0,1, training_sample_sizes)):               
                
                if initial == False: 
                    np.savetxt(f"training_1/arr_checkpoint_{unit}.txt", [bookmark_train, bookmark_validation, batch])
                
                else: 
                    bookmark_train, bookmark_validation = 0,0
                
                bookmark_train, bookmark_validation = train_model(batch, int(bookmark_train), int(bookmark_validation))
                initial = False

