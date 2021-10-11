import numpy as np
import pandas as pd
import datetime
import sklearn as sk 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


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






# =================================== CREATING RANDOM FOREST CLASSIFIER ==========================


clf = RandomForestClassifier(max_depth=7, random_state=42)





#===================================== TRAINING MODEL =============================================


clf.fit(X_train, y_train)





#===================================== EVALUATING MODEL ===========================================


#printing model accuracy
print("Model accuracy: ", clf.score(X_test, y_test))



#Creating Confusion Matrix
y_test_pred = clf.predict(X_test)




cm = confusion_matrix(y_test, y_test_pred)




cmd_obj = ConfusionMatrixDisplay(cm, display_labels=["Magnetosphere", 
							"Magnetosheath", 
							"Solar Wind"])
cmd_obj.plot()

cmd_obj.ax_.set(title='Confusion Matrix for Random Forest Classifier', 
                xlabel='Predicted Location', 
                ylabel='Actual Location')


plt.show()
