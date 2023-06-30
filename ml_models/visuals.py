import pandas as pd
import datetime
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tqdm.notebook import tqdm
import numpy as np
from matplotlib.dates import DateFormatter

# Precomputed results
results = pd.read_csv("2022-12-07-11_50_40test_results.csv")

# Compute and print classification report
cr = classification_report(results["y_true"], 
                     results["y_pred"])
print(cr)

# Compute confusion matrix
messenger_confusion_matrix = confusion_matrix(results["y_true"], results["y_pred"])

# Plot confusion matrix
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)
cmd_obj = ConfusionMatrixDisplay(messenger_confusion_matrix, display_labels=["Magnetosphere", "Magnetosheath", "Solar Wind"]) 
cmd_obj.plot()

# Set plot title and labels
cmd_obj.ax_.set(
    title='Confusion Matrix for LSTM Classifier', 
    xlabel='Predicted Location', 
    ylabel='Manually Labelled Location'
)

plt.tight_layout()

# Get the current date and time
date_time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

# Save the figure
plt.savefig(f"LSTM_confusion_matrix_{date_time_now}.png")
plt.close(fig)

df_traj = pd.read_parquet("../data/2012-04-19.parquet")

checkpoint_path = glob("../data_prep/22_07_24_LSTM_1_weights/cp.ckpt")[0]


#Defining Constants/building model

batch_size = 64
units = 1024
output_size = 3
training_sample_sizes = 0.1

def build_model():
    lstm_layer = keras.layers.RNN(
        keras.layers.LSTMCell(units), input_shape=(300, 3)
    )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.Dropout(0.2),
            keras.layers.Dense(output_size),
        ]
    )
    return model
    
model = build_model()

# load model weights
model.load_weights(checkpoint_path)

# Calculate prediction probabilities for each time step
probabilities = np.empty(shape = [len(df_traj)- 150, 3])
for x in tqdm(range(150, len(df_traj)-150), position = 0, leave = False, display = True):

    probabilities[x] = tf.nn.sigmoid(
        model.predict(
            np.array(
                [np.array(df_traj[[ "B_X_MSO",
                                    "B_Y_MSO",
                                    "B_Z_MSO"]
                                 ].iloc[x-150:x+150])]))).numpy()[0]

probabilities = pd.DataFrame(np.argmax(probabilities, axis = 1))

# Define colour mapping for regions
mapping = {'0': '#b80000', '1': '#1273de', '2': '#fccb00'}

# Replace region labels with hex 
probabilities = pd.DataFrame(probabilities).astype("str")[0].replace(mapping)

probabilities = np.array(probabilities)

probabilities = np.char.mod('%s', probabilities)

# Save predictions to a text file
np.savetxt('probabilities_LSTM_traj_04_19_2012.txt', probabilities, fmt='%s')

mapping = {"0": "#b80000",  "1":"#1273de", "2":"#fccb00"}

## Create custom lines
r_line = plt.Line2D((0,1),(0,0), color='#b80000')
b_line = plt.Line2D((0,1),(0,0), color='#1273de')
y_line = plt.Line2D((0,1),(0,0), color='#fccb00')

# Remove initial 150 rows from df_traj
df_traj.drop(df_traj.index[:150], inplace = True)

# Filter df_traj by start and end time
df_traj[(df_traj.index > "2012-04-19 14:00:00") & (df_traj.index < "2012-04-19 23:00:00")]

df_traj["predicted_location"] = np.array(probabilities)

start_time = "2012-04-19 14:00:00"
end_time = "2012-04-19 23:00:00"

df_traj_subwindow = df_traj[(df_traj.index > start_time) & (df_traj.index < end_time)]

# Create plot
fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
fig.set_dpi(200)

def gen_repeating(s):
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        yield (s[i], i, j-1)
        i = j

# Add lines for each location segment
for color, start, end in gen_repeating(df_traj_subwindow["predicted_location"]):
    if start > 0: # make sure lines connect
        start -= 1
    idx = df_traj_subwindow.index[start:end+1]
    ax.plot(np.sqrt(df_traj_subwindow.loc[idx, 'B_X_MSO']**2+
                   df_traj_subwindow.loc[idx, 'B_Y_MSO']**2+
                   df_traj_subwindow.loc[idx, 'B_Z_MSO']**2),
            color=color,
            label='',
            linewidth = 2)

# Get handles and labels for legend
handles, labels = ax.get_legend_handles_labels()

# Create legend label list
ax.legend(
    handles + [r_line, 
               b_line, 
               y_line],
    labels + [
        'Magnetosphere',
        'Magnetosheath',
        'Solar Wind'
    ],
    fontsize=20,
    loc='upper right',
    bbox_to_anchor=(1.2, 1.0175)
)

#Formating x-axis dates
ax.xaxis_date()
fig.autofmt_xdate()
formatter =  DateFormatter('%Y-%h-%d %H:%M')
ax.xaxis.set_major_formatter(formatter)
plt.ylabel("|B| (nT)")

# Save plot
plt.savefig(f"LSTM_line_plot_{date_time_now}.png")
plt.close(fig)

# Line plot for WJS's manually created labels
wj_mapping = {'0': '#b80000', '-1': '#5300eb', '1': '#1273de', '-2': '#e65100', '2': '#fccb00'}

# Create custom lines for legend
r_line = plt.Line2D((0,1),(0,0), color='#b80000')
o_line = plt.Line2D((0,1),(0,0), color='#5300eb')
b_line = plt.Line2D((0,1),(0,0), color='#1273de')
p_line = plt.Line2D((0,1),(0,0), color='#e65100')
y_line = plt.Line2D((0,1),(0,0), color='#fccb00')

start_time = "2012-04-19 14:00:00"
end_time = "2012-04-19 23:00:00"

# Filter df_traj by start and end time
df_traj_subwindow = df_traj[(df_traj.index > start_time) & (df_traj.index < end_time)]

# Create plot
fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
fig.set_dpi(200)


# Add lines for each location segment 
for color, start, end in gen_repeating(df_traj_subwindow["Location"].replace(wj_mapping)):
    if start > 0: # make sure lines connect
        start -= 1
    idx = df_traj_subwindow.index[start:end+1]
    ax.plot(np.sqrt(df_traj_subwindow.loc[idx, 'B_X_MSO']**2+
                   df_traj_subwindow.loc[idx, 'B_Y_MSO']**2+
                   df_traj_subwindow.loc[idx, 'B_Z_MSO']**2),
            color=color,
            label='',
            linewidth = 2)

# get handles and labels for legend
handles, labels = ax.get_legend_handles_labels()

# Create legend from label list
ax.legend(
    handles + [r_line, 
               o_line,
               b_line,
               p_line,
               y_line],
    labels + [
        'Magnetosphere',
        'Magnetopause',
        'Magnetosheath',
        'Bow Shock',
        'Solar Wind'
    ],
    fontsize=20,
    loc='upper right',
    bbox_to_anchor=(1.2, 1.0175)
)

#Formating x-axis dates
ax.xaxis_date()
fig.autofmt_xdate()
formatter =  DateFormatter('%Y-%h-%d %H:%M')
ax.xaxis.set_major_formatter(formatter)
plt.ylabel("|B| (nT)")

# save figure
plt.savefig(f"WJS_line_plot_{date_time_now}.png")
plt.close(fig)




