"""David Hernán García Fernández - A01173130"""
"""Import libraries"""
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

"""Obtain the datasets"""
data_B = pd.read_csv('B.csv')
data_BC = pd.read_csv('BC.csv')
data_C = pd.read_csv('C.csv')
data_E = pd.read_csv('E.csv')
data_EB = pd.read_csv('EB.csv')

"""Concatenate the datasets to make one"""
data_set = pd.concat([data_B, data_C, data_BC, data_E, data_EB], axis=0)

"""Mix the dataset content randomly"""
data_set = shuffle(data_set, random_state=50)

"""Separate the target (y)"""
data_set_x = data_set[["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"]]
data_set_y = data_set[["y"]]

"""Create the training and testing sets"""
X_training, X_testing, y_training, y_testing =  train_test_split(data_set_x, data_set_y, test_size=0.15)
X_training = np.squeeze(X_training)
X_testing = np.squeeze(X_testing)
y_training = np.squeeze(y_training)
y_testing = np.squeeze(y_testing)

"""Create the Random Forest Classifier by defining its parameters"""
rnd_full = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42, bootstrap=False)
"""Fit the model"""
rnd_full.fit(X_training, y_training)
"""Make a prediction"""
y_pred= rnd_full.predict(X_testing)
"""Print the efficiency of the model"""
print("The efficiency of this model is: ", accuracy_score(y_testing, y_pred))


def resp_invalida(answer):
    """Function that cycles the program until the user 
    provides a valid dataset name"""
    ciclar = True
    while ciclar == True:
      try:
        pruebas = pd.read_excel(answer)
        return pruebas
        ciclar = False
      except:
        print("\nInvalid Name, try again")
        answer=input("\nEnter the name of the file and its path: ")

def resp_mala(answer):
    """Function that cycles the program until the user 
    provides a valid answer"""
    ciclar = True
    while ciclar == True:
        if answer in ('y', 'Y', 'yes', 'Yes', 'YES'):
            return True
            ciclar = False
        elif answer in ('N', 'n', 'no', 'No', 'NO'):   
            return False
            ciclar = False
        else:
            print("\nInvalid answer, try again")
            answer=input("\nDo you want to analyze another file? [Y/N]: ")
            
#Arch_Name="/content/drive/MyDrive/Sistemas inteligentes/FP/Test/TestDataset.csv"

"""Initialize variables"""
resp_bool=True
respuesta=None

while resp_bool == True:
  """Request file name"""
  answer=input("\nEnter the name of the file: ")
  """Validate the file name"""
  pruebas=resp_invalida(answer)
  pruebas = pruebas[["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"]]
  
  """Make a prediction"""
  y_pred= rnd_full.predict(pruebas)
  
  """Initialize variables"""
  count_B=count_BC=count_C=count_E=count_EB=0

  """Count the types of cardboard sheets"""
  for i in range(len(y_pred)):
    if y_pred[i] == 1:
      count_E+=1
    elif y_pred[i] == 2:
      count_EB+=1
    elif y_pred[i] == 3:
      count_B+=1
    elif y_pred[i] == 4:
      count_C+=1
    elif y_pred[i] == 5:
      count_BC+=1
    else:
      print("\nERROR: Unidentified cardboard sheet")
  
  """Print the accounts"""
  print("\nThere are ",count_B," type B cardboard sheets")
  print("\nThere are ",count_BC," type BC cardboard sheets")
  print("\nThere are ",count_C," type C cardboard sheets")
  print("\nThere are ",count_E," type E cardboard sheets")
  print("\nThere are ",count_EB," type EB cardboard sheets")

  """Another round?"""
  respuesta=input("\nDo you want to analyze another file? [Y/N]: ")
  """Validate the answer"""
  resp_bool=resp_mala(respuesta)

print("Have a nice day!")