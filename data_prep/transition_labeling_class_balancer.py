import random
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime

#============================================= DATA IMPORT AND FORMATING =======================================================
print("Importing Crossing Data...")


#Path to directory where the boundary crossing list is located. 

dir_path = ""

crossing_list = pd.read_csv(dir_path + "Mercury_Boundary_Crossing_List.csv", sep = ',',skipinitialspace = True, index_col = False)



#Converting transitions and Messenger data to datetime format

stopgap_minutes = 10


crossing_list["Start_Date"] = pd.to_datetime(crossing_list["Start_Date"], format = "%Y-%m-%d %H:%M:%S") - datetime.timedelta(minutes = stopgap_minutes)
crossing_list["End_Date"] = pd.to_datetime(crossing_list["End_Date"], format = "%Y-%m-%d %H:%M:%S") + datetime.timedelta(minutes = stopgap_minutes)

crossing_list = crossing_list.drop("Type", axis = 1).to_numpy()


print("Importing Messenger Data...")

#Importing Messenger Data

df = pd.read_csv(dir_path +"validation_list.csv")



#Converting first column to datetime format 
df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%d %H:%M:%S")




#Initial column for transitions

df["Transition"] = 0

df["Transition"]= df["Transition"].astype('bool')




#========================================== TRANSITION + STOP GAP LABELING ======================================

print("Labeling Data...")

#Labeling date ranges where transitions occurred (with a 10 minute buffer either side)

for tran in range(0, len(crossing_list)):
    
    if df.loc[df["Date"] == crossing_list[tran][0]].empty:
        
        if df.loc[df["Date"] == crossing_list[tran][1]].empty:
            pass
        
        else: 
            df["Transition"][df["Date"].between(
            crossing_list[tran][0],
            crossing_list[tran][1])
        ] = 1
            
    else: 
        df["Transition"][df["Date"].between(
            crossing_list[tran][0],
            crossing_list[tran][1])
        ] = 1

del crossing_list



#==================================== CLASS BALANCING ================================================================
All_0s = np.where(df.loc[df["Transition"] == 0]["Location"] == 0)[0]

All_1s = np.where(df.loc[df["Transition"] == 0]["Location"] == 1)[0]

All_2s = np.where(df.loc[df["Transition"] == 0]["Location"] == 2)[0]



classBalance = np.min([len(All_0s),len(All_1s),len(All_2s)]) #Must be smaller than smallest class size





#Creating 3 randomly sampled indicies of equal size corresponding to the Magnetosheath, Magnetosphere and Solar Wind.  

if len(All_0s) != classBalance:

    train_0s_index = random.sample(list(All_0s),classBalance)

else:

    train_0s_index = All_0s



if len(All_1s) != classBalance:

    train_1s_index = random.sample(list(All_1s),classBalance)

else:

    train_1s_index = All_1s

    

if len(All_2s) != classBalance:

    train_2s_index = random.sample(list(All_2s),classBalance) 

else:

    train_2s_index = All_2s





#======================================= EXPORTING TRANSITION LABELED, CLASS BALANED DATASET TO CSV ==================================

df.iloc[np.concatenate([np.array(train_0s_index) , 
			np.array(train_1s_index),
			np.array(train_2s_index)])].sort_values(by=['Date']).to_csv("discontinusous_tansitionless_class_balanced_random_sample.csv", index = False)


