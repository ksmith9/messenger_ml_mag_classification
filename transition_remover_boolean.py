import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import csv

csv.field_size_limit(sys.maxsize)
print("Importing Crossings Data...")
#Importing crossing data
MPIcrossings = pd.read_csv("MagPause_In.txt", sep = ' ',skipinitialspace = True, index_col = False)
MPOcrossings = pd.read_csv("MagPause_Out.txt", sep = ' ',skipinitialspace = True, index_col = False)
BSIcrossings = pd.read_csv("Bow_Shock_In.txt", sep = ' ',skipinitialspace = True, index_col = False)
BSOcrossings = pd.read_csv("Bow_Shock_Out.txt", sep = ' ',skipinitialspace = True, index_col = False)

print("Formating Data...")
#Stitching lists together 
crossing_list = pd.concat([MPIcrossings, MPOcrossings, BSIcrossings, BSOcrossings])

del MPIcrossings
del MPOcrossings
del BSIcrossings
del BSOcrossings


print("Importing Messenger Data...")

#Importing Messenger Data
df = pd.read_csv("validation_list.csv")




#Converting crossing list to strings and formating text to match the Messenger date fomat 
crossing_list.loc[:, "S_Yr":"E_Sec"] = crossing_list.loc[:, "S_Yr":"E_Sec"].astype(str)
crossing_list["S_Sec"], crossing_list["E_Sec"] = crossing_list["S_Sec"].str.rstrip(".0"), crossing_list["E_Sec"].str.rstrip(".0")

for col in crossing_list.loc[:, "S_Yr":"E_Sec"].columns:
    crossing_list[col] = crossing_list[col].str.rjust(2,"0")

#Compiling date columns 
crossing_list['Start_Date'] = crossing_list["S_Yr"]+"-"+crossing_list["S_Mon"]+"-"+crossing_list["S_Day"]+" "+crossing_list["S_Hr"]+":"+crossing_list["S_Min"]+":"+crossing_list["S_Sec"]
crossing_list['End_Date'] = crossing_list["E_Yr"]+"-"+crossing_list["E_Mon"]+"-"+crossing_list["E_Day"]+" "+crossing_list["E_Hr"]+":"+crossing_list["E_Min"]+":"+crossing_list["E_Sec"]


#Dropping superfluous columns 
crossing_list.drop(crossing_list.loc[:, "S_Yr":"E_Sec"].columns, axis = 1, inplace=True)
crossing_list.drop(["Inst", "Type"], axis = 1, inplace = True)


#Converting transitions and Messenger data to datetime format
crossing_list["Start_Date"] = pd.to_datetime(crossing_list["Start_Date"], format = "%Y-%m-%d %H:%M:%S") - datetime.timedelta(minutes = 10)
crossing_list["End_Date"] = pd.to_datetime(crossing_list["End_Date"], format = "%Y-%m-%d %H:%M:%S") + datetime.timedelta(minutes = 10)

df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%d %H:%M:%S")


#Initial column for transitions (to be changed later)
df["Transition"] = 0
df["Transition"]= df["Transition"].astype('bool')

print("Labeling Data...")
#Labeling date ranges where transitions occurred (with a 10 minute buffer either side)
for tran in range(0, len(crossing_list)):
    if df.loc[df["Date"] == crossing_list.iloc[tran]["Start_Date"]].empty:
        if df.loc[df["Date"] == crossing_list.iloc[tran]["End_Date"]].empty:
            pass
        else: 
            df["Transition"][df["Date"].between(
            crossing_list.iloc[tran]["Start_Date"],
            crossing_list.iloc[tran]["End_Date"])
        ] = 1
    else: 
        df["Transition"][df["Date"].between(
            crossing_list.iloc[tran]["Start_Date"],
            crossing_list.iloc[tran]["End_Date"])
        ] = 1

del crossing_list

print("Transition value counts: ", df["Transition"].value_counts())

print("Removing stop gaps and exporting...")
#Dataframe without transitions and stop gap:
df.loc[df["Transition"] == 0].drop("Transition", axis =1).to_csv("validation_list_no_transition.csv")
del df


