from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import datetime
from tqdm import tqdm
from matplotlib.dates import DateFormatter
from scipy import stats
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import pyarrow as pa
import pickle
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import MinMaxScaler

start_time = datetime.datetime.now()
print("Starting")

np.random.seed(51225)

data_directory = Path("../data")
crossing_list_csv = Path("mercury_boundary_crossing_list.csv")

# Reading WJS' the crossing list CSV file into a pandas DataFrame
crossing_list = pd.read_csv(
	data_directory / crossing_list_csv,
	sep=',', skipinitialspace=True, index_col=False,
	usecols=["Start_Date", "End_Date"],
	parse_dates=["Start_Date", "End_Date"], infer_datetime_format=True
)

# Creating a crossing list from WJS' Magnetopause and Bow Shock crossing list
MPIcrossings = pd.read_csv(data_directory / Path("MagPause_In_Time_Duration__public_version_WeijieSun_20230829.txt"), sep = ' ',skipinitialspace = True, index_col = False)
MPOcrossings = pd.read_csv(data_directory / Path("MagPause_Out_Time_Duration_public_version_WeijieSun_20230829.txt"), sep = ' ',skipinitialspace = True, index_col = False)

BSIcrossings = pd.read_csv(data_directory / Path("Bow_Shock_In_Time_Duration__public_version_WeijieSun_20230829.txt"), sep = ' ',skipinitialspace = True, index_col = False)
BSOcrossings = pd.read_csv(data_directory / Path("Bow_Shock_Out_Time_Duration_public_version_WeijieSun_20230829.txt"), sep = ' ',skipinitialspace = True, index_col = False)

#Combining and cleaning up files
cross_list = [MPIcrossings,MPOcrossings, BSIcrossings, BSOcrossings]

del MPIcrossings
del MPOcrossings
del BSIcrossings
del BSOcrossings

crossings = pd.concat(cross_list)
del cross_list

#Converting date and time components to zero padded strings
crossings["S_Yr"] = crossings["S_Yr"].astype(str).str.zfill(4)
crossings["S_Mon"] = crossings["S_Mon"].astype(str).str.zfill(2)
crossings["S_Day"] = crossings["S_Day"].astype(str).str.zfill(2)
crossings["S_Hr"] = crossings["S_Hr"].astype(str).str.zfill(2)
crossings["S_Min"] = crossings["S_Min"].astype(str).str.zfill(2)
crossings["S_Sec"] = crossings["S_Sec"].astype(int).astype(str).str.zfill(2)

crossings["E_Yr"] = crossings["E_Yr"].astype(str).str.zfill(4)
crossings["E_Mon"] = crossings["E_Mon"].astype(str).str.zfill(2)
crossings["E_Day"] = crossings["E_Day"].astype(str).str.zfill(2)
crossings["E_Hr"] = crossings["E_Hr"].astype(str).str.zfill(2)
crossings["E_Min"] = crossings["E_Min"].astype(str).str.zfill(2)
crossings["E_Sec"] = crossings["E_Sec"].astype(int).astype(str).str.zfill(2)

dates = crossings["S_Yr"] + crossings["S_Mon"]+ crossings["S_Day"] +crossings["S_Hr"] +crossings["S_Min"] +crossings["S_Sec"]

#Creating a single start date column
crossings["S_Date"] = dates

crossings["S_Date"] = pd.to_datetime(crossings["S_Date"],format = "%Y%m%d%H%M%S")

#Creating a single end date column
dates = crossings["E_Yr"] + crossings["E_Mon"]+ crossings["E_Day"] +crossings["E_Hr"] +crossings["E_Min"] +crossings["E_Sec"]

crossings["E_Date"] = dates
del dates

crossings["E_Date"] = pd.to_datetime(crossings["E_Date"],format = "%Y%m%d%H%M%S")

#Sorting crossing list by start date
crosssort = crossings.sort_values("S_Date")
del crossings

#Setting start and end date based on available data from WJS
start_date = datetime.date(2011,6,1)
strdate = datetime.datetime(2011,3,24,0,0,0)
enddate = datetime.datetime(2014,3,16,0,0,0) # <------------------- Keep this unless Weijie Sun's list is updated
#enddate = datetime.datetime(2015,4,30,0,0,0)


#Filename of raw MESSENGER data
Filename = "Messenger_"+strdate.strftime("%Y%j")+".csv"
#Data column names
CNames = ['Year', 'DOY', 'Hour', 'Minute', 'Second', 'X_MSO', 'Y_MSO', 'Z_MSO', 'B_X_MSO', 'B_Y_MSO', 'B_Z_MSO']
#Initial import of first file
Data = pd.read_csv( data_directory / Path("MESSENGER") / Path(Filename), sep = ' ', skipinitialspace = True, header = None, names = CNames)


#Data assimilation loop

n_days = ((enddate -strdate).days)

EMPTY_FILE_THRESHOLD = 1024  # Set the threshold for an empty file (in bytes)

print("Assimilating MESSENGER data...")
for day in tqdm(range(1, n_days)):
	next_day = strdate + datetime.timedelta(days=day)
	next_day_str = next_day.strftime("%Y") + next_day.strftime("%j")
	Filename = "Messenger_" + next_day_str + ".csv"
	
	file_path = data_directory / Path("MESSENGER") / Path(Filename)
	#Skip over next file if it is empty and raise a warning
	if not os.path.exists(file_path) or os.path.getsize(file_path) < EMPTY_FILE_THRESHOLD:
		print(f"Warning: {Filename} is empty or doesn't exist.")
		continue
	
	Data2 = pd.read_csv(file_path, sep=' ', skipinitialspace=True, header=None, names=CNames)
	Data = pd.concat([Data,Data2], ignore_index = True)


#cleaning up files
del n_days
del next_day
del next_day_str
del Data2

#Converting date and time in the MESSENGER DataFrame to zero padded strings
Data["Year"] = Data["Year"].astype(str).str.zfill(4)
Data["DOY"] = Data["DOY"].astype(str).str.zfill(3)
Data["Hour"] = Data["Hour"].astype(str).str.zfill(2)
Data["Minute"] = Data["Minute"].astype(str).str.zfill(2)
Data["Second"] = pd.to_numeric(Data["Second"], errors = 'coerce').fillna(0).astype(int).astype(str).str.zfill(2)

#Creating a single date column
Dates = Data["Year"] + Data["DOY"] +Data["Hour"] +Data["Minute"] +Data["Second"]
Data["Date"] = Dates
del Dates

Data["Date"] = pd.to_datetime(Data["Date"],format = "%Y%j%H%M%S", errors = 'coerce')

print("Datetimes Created")

# Dropping rows with missing values in the "Date" column and setting date as index
Data.dropna(subset = ["Date"], inplace=True)

Data = Data.set_index("Date")

# Resampling the data to a 1-second interval and interpolating missing values
Data = Data.resample("1S").mean().interpolate()

# Calculating the azimuthal angle in degrees based on the X_MSO and Y_MSO columns
p = np.degrees(np.arctan2(Data["Y_MSO"], Data["X_MSO"]))+180.
# Calculating the difference in consecutive angles
dp = np.array(p[1:])- np.array(p[:-1])
# Finding the indices where the angle difference is less than -180 (indicating the start of a new orbit)
orbitStartIndex = np.where(dp  < -180)[0]
lenp = len(p)

del p
del dp

# Creating a location array of zeros with the same shape as the MESSENGER Data
loc = np.zeros(Data.shape[0]).astype(int)

# Calculating the time difference in seconds between each date in "crosssort" and the first date in Data
xxx = (crosssort["S_Date"] - Data.index[0])
xxx = xxx.dt.total_seconds()

print("Created Location Array")

if np.nanmin(xxx) < 0:
	# If the minimum value in 'xxx' is less than 0
	i = np.argmax(np.array(xxx.iloc[np.where(xxx < 0)]))
	# Find the index of the maximum value within the subset of 'xxx' where values are less than 0
		# If the crossing at the index is MPO or BSI, set all elements to 1
	if crosssort["Type"].iloc[i] == "BSI":
		loc[:] = 1
	if crosssort["Type"].iloc[i] == "MPO":
		loc[:] = 1
	# If the crossing at the index is MPI, set all elements to 0
	if crosssort["Type"].iloc[i] == "MPI":
		loc[:] = 0
	# If the crossing at the index is BSO, set all elements to 2	
	if crosssort["Type"].iloc[i] == "BSO":
		loc[:] = 2

else:
	loc[:] = 2
    # If the minimum value in 'xxx' is not less than 0, set all elements to 2
del xxx

for i in range(len(crosssort)):
	# Find the MESSENGER Data index where a crossing starts
	index = np.where((Data.index - crosssort["S_Date"].iloc[i]).total_seconds() == 0)[0]
	if len(index) == 1:
		index = index[0]
		# Depending on the crossing type, set all remaining locations accordingly, when we reach the end of the crossing list, all elements up to the last documented crossing will be correctly classified
		if crosssort["Type"].iloc[i] == "BSI":
			loc[index:] = -2
			
		if crosssort["Type"].iloc[i] == "MPO":
			loc[index:] = -1
			
		if crosssort["Type"].iloc[i] == "MPI":
			loc[index:] = -1
			
		if crosssort["Type"].iloc[i] == "BSO":
			loc[index:] = -2

	index = np.where((Data.index - crosssort["E_Date"].iloc[i]).total_seconds() == 0)[0]
	if len(index) == 1:
		index = index[0]
		if crosssort["Type"].iloc[i] == "BSI":
			loc[index:] = 1
			
		if crosssort["Type"].iloc[i] == "MPO":
			loc[index:] = 1
			
		if crosssort["Type"].iloc[i] == "MPI":
			loc[index:] = 0
			
		if crosssort["Type"].iloc[i] == "BSO":
			loc[index:] = 2

del crosssort
print("Filled Location Array")
Data["Location"] = loc.tolist()

#Creating an array counting which orbit the spacecraft is on
orbit = np.zeros(lenp)
for i in range(len(orbitStartIndex)):
	orbit[orbitStartIndex[i]:] = i

del lenp
del orbitStartIndex

Data["OrbitNo"] = orbit.tolist()
print("Filled Orbit List")
df = Data
del Data



def outlier_removal_normalisation(col, verbose = False):

	"""
	Perform outlier removal and normalization on a given column.

	Parameters:
		col (Series): The column to process.
		verbose (bool): If True, display additional information and generate plots.

	Returns:
		Series: The processed column with outliers removed and normalized values.

	"""
	if verbose == True:
		print("Old max: ", col.max())
		print("Old min: ", col.min())
		hist1 = sns.histplot(col)
		fig1 = hist1.get_figure()
		fig1.savefig(f"{col.name}_Before_Hard_Robust3_MinMax.png")
	# Outlier removal using median absolute deviation (MAD)
	b = 1/np.quantile(col, 0.75, axis = None)
	MAD = median_abs_deviation(col, scale = b)
	M = np.median(col)
	col.mask( (col-M)/MAD <-2, inplace = True )
	col.mask( (col-M)/MAD > 2, inplace = True )
	
	if verbose == True: 
		print("Amount removed: ", col.isna().value_counts())

    	# Interpolate missing values
	normal = col.interpolate(method = "pad")
	# Apply robust scaling
	transformer = RobustScaler(quantile_range = (1,99)).fit(normal.to_numpy().reshape(-1,1))
	scaled = transformer.transform(normal.to_numpy().reshape(-1,1))
	# Apply min-max scaling
	transformer = MinMaxScaler(feature_range = (-1,1)).fit(normal.to_numpy().reshape(-1,1))
	scaled = transformer.transform(normal.to_numpy().reshape(-1,1))
	if verbose == True:
		print("New max: ", scaled.max())
		print("New min: ", scaled.min())
		hist2 = sns.histplot(scaled)
		fig2 = hist2.get_figure()
		fig2.savefig(f"{col.name}_After_Hard_Robust3_MinMax.png")
	return scaled

#Create normalised columns with outliers removed 
df["B_X_MSO_Normalised"] = outlier_removal_normalisation(df["B_X_MSO"])
df["B_Y_MSO_Normalised"] = outlier_removal_normalisation(df["B_Y_MSO"])
df["B_Z_MSO_Normalised"] = outlier_removal_normalisation(df["B_Z_MSO"])

df = df[df.index < "March 2014"] # <------------------- Keep this unless Weijie Sun's list is updated


def batch_assignment(buffer_size):
	    """
	    Assigns data to batches of 5 minute duration, excluding transition regions. This can then be fed into an RNN to perform region classification. A buffer can be added to remove data near the transition regions. 

	    Parameters:
		buffer_size (int): Buffer size in minutes.

	    Returns:
		None

	    """
	cl = crossing_list.copy()
	if buffer_size > 0: 
		#Labeling Transition Regions
		stopgap_minutes_delta = datetime.timedelta(minutes=buffer_size)
		cl["Start_Date"] -= stopgap_minutes_delta
		cl["End_Date"] += stopgap_minutes_delta

	# Initial column for transitions (to be changed later)
	df["Transition"] = False


	tqdm().pandas()


	# Labeling date ranges where transitions occurred (with a buffer_size minute buffer either side)

	print(f"Labeling transition regions for buffer size {buffer_size}...")


	for idx, row in tqdm(cl.iterrows(), total=len(cl), position = 0, leave = True):
		df.loc[row.Start_Date:row.End_Date, "Transition"] = True

	print("Elapsed:", datetime.datetime.now() - start_time)
	print(df.Transition.value_counts())



	print("Labeling sequences for buffer size {buffer_size}...")
	batch = 0
	x = 0
	total_data_size = len(df)
	if buffer_size > 0:
		df[f"Batch_{buffer_size}"] = -1

		pbar = tqdm(total=len(df), position=0, leave=True)
		# If both the start and end of a 5 minute window are outside a transition region, proceed to add the data to a batch. Otherwise, move on a second and check again. 
		while x < total_data_size - 300:
			if x + 150 < total_data_size:
				if df.loc[df.index[x - 150], "Transition"] == False and df.loc[df.index[x + 150], "Transition"] == False:
					df.loc[df.index[x - 150]:df.index[x + 149], f"Batch_{buffer_size}"] = batch  # Adjusted indexing range
					batch += 1
					x += 300
					pbar.update(300)
				else:
					x += 1
					pbar.update(1)
			else:
				x += 1
				pbar.update(1)

	else:
		df["Batch"] = -1

		pbar = tqdm(total=len(df), position=0, leave=True)

		while x < total_data_size - 300:
			if x + 150 < total_data_size:
				if df.loc[df.index[x - 150], "Transition"] == False and df.loc[df.index[x + 150], "Transition"] == False:
					df.loc[df.index[x - 150]:df.index[x + 149], "Batch"] = batch  # Adjusted indexing range
					batch += 1
					x += 300
					pbar.update(300)
				else:
					x += 1
					pbar.update(1)
			else:
				x += 1
				pbar.update(1)
	return 



batch_assignment(10)
batch_assignment(5)
batch_assignment(0)

del crossing_list

#Test/Train/Validation split
print("Constructing TTV datasets...")
def export_np_to_parquet(data, filename):
	"""
	Export NumPy array to Parquet file format.

	Parameters:
		data (ndarray): NumPy array to export.
		filename (str): Name of the file (without extension).

	Returns:
		None
	"""
		with open(data_directory /Path( filename+"_dimensions.pickle"), "wb") as f:
			pickle.dump(data.shape, f)
		table = pa.table({"data": data.reshape(-1)})
		pa.parquet.write_table(table, data_directory / Path(filename+".parquet"))
		return

def ttv_split(size):
	"""
	Perform test/train/validation split on data and export as Parquet files.

	Parameters:
		size (str): Size of the buffer (for file nameing and column accessing purposes).

	Returns:
		None

	"""
	batch = df["Batch"+size].max()	
	#Test/Train/Validation split
	# 60:20:20 split
	train_features = np.empty(shape = [int(batch*0.6), 300, 3])
	test_features = np.empty(shape = [int(batch*0.2), 300, 3])
	validation_features = np.empty(shape = [int(batch*0.2), 300, 3])

	train_targets = np.empty(shape = [int(batch*0.6)])
	test_targets = np.empty(shape = [int(batch*0.2)])
	validation_targets = np.empty(shape = [int(batch*0.2)])

	train_len = len(train_targets)
	test_len = len(test_targets)


	df_features = df[["B_X_MSO_Normalised", "B_Y_MSO_Normalised", "B_Z_MSO_Normalised"]]
	df_targets = df["Location"]
	df_batches = df["Batch"+size]



	train_index = 0
	test_index = 0
	validation_index = 0
	print("Creating test/train/validation split...")
	pbar = tqdm(total=batch , position = 0, leave = True)
	b = 1
	while b < batch -1: 
		if b == 0: 
			print(df_features.loc[df_batches == b])
		dataset_det = np.random.uniform(0,1)
		if dataset_det < 0.6 and train_index < train_len: 
			train_features[train_index] = df_features.loc[df_batches == b]
			train_targets[train_index] = df_targets.loc[df_batches == b].unique()[0]
			train_index +=1
			b += 1
			pbar.update(1)
		elif dataset_det < 0.8 and test_index < test_len: 
			test_features[test_index] = df_features.loc[df_batches == b]
			test_targets[test_index] = df_targets.loc[df_batches == b].unique()[0]
			test_index +=1
			b += 1
			pbar.update(1)
		elif validation_index < test_len:
			validation_features[validation_index] = df_features.loc[df_batches == b]
			validation_targets[validation_index] = df_targets.loc[df_batches == b].unique()[0]		
			validation_index +=1
			b +=1
			pbar.update(1)
	# Exporting datasets	
	export_np_to_parquet(train_features, "training_features"+size)
	export_np_to_parquet(train_targets, "training_targets"+size)

	export_np_to_parquet(test_features, "test_features"+size)
	export_np_to_parquet(test_targets, "test_targets"+size)

	export_np_to_parquet(validation_features, "validation_features"+size)
	export_np_to_parquet(validation_targets, "validation_targets"+size)
	return
#Also export the whole dataset with relevant batch/transition information. 
df.to_parquet(data_directory  / Path("full_sequenced_MESSENGER_data_combined_batches.parquet"))

ttv_split("")
print("\n Finished No Buffer! \n")
ttv_split("_5")
print("\n Finished 5 Minute Buffer! \n")
ttv_split("_10")
print("\n Finished 10 Minute Buffer! \n")
