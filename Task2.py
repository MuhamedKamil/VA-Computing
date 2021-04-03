import pandas as pd
import numpy as np

import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score


data = {'y_Actual':    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        'y_Predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
        }

data2 = {
         'y_Actual':['yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes'],
         'y_Predicted':['yes', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes']
        }

"""
df['y_Actual'] = df['y_Actual'].map({'yes': 1, 'no': 0})
df['y_Predicted'] = df['y_Predicted'].map({'yes': 1, 'no': 0})
"""

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
df['y_Actual'] = df['y_Actual']
df['y_Predicted'] = df['y_Predicted']
confusion = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

TN = confusion[0].tolist()[0] #D
FN = confusion[0].tolist()[1] #B
FP = confusion[1].tolist()[0] #C
TP = confusion[1].tolist()[1] # A
#---------------------------------------------------------
TotalSamples = TN+FN+FP+TP
PO = (TN+TP)/TotalSamples
Pcorrect   = ((TP+FN)/TotalSamples)*((TP+FP)/TotalSamples)
Pincorrect = ((FP+TN)/TotalSamples)*((FN+TN)/TotalSamples)
Pe = Pcorrect + Pincorrect
K = 1-(1-PO)/(1-Pe)
print("Result = " , K)
#---------------------------------------------------------
#Build in function result
rater1 = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
rater2 = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
print("Built in function Result = "  ,cohen_kappa_score(rater1,rater2))

