import pandas as pd
import numpy as np

def ReadCSV (path):
	data = pd.read_csv(path)
	return data

def merge(list1, list2):
	merged_list = [[list1[i], list2[i]] for i in range(0, len(list1))]
	return merged_list

def PrepareBatches (BatchSize ,TrainingData,  TrainingLabels,modAvailable = True):
	batchlistforData    = []
	batchlistforLabels  = []
	numOfsamples        = TrainingData.shape[0]
	numberOfbatches , mod = divmod(numOfsamples , BatchSize)
	DataBatch           = np.zeros((BatchSize, TrainingData.shape[1]))
	LabelsBatch         = np.zeros((BatchSize, TrainingLabels.shape[1]))

	for b in range (0,numberOfbatches):
		DataBatch   = TrainingData[b*BatchSize:(b+1)*BatchSize,:]
		LabelsBatch = TrainingLabels[b*BatchSize:(b+1)*BatchSize,:]
		batchlistforData.append(DataBatch)
		batchlistforLabels.append(LabelsBatch)

	if modAvailable and mod > 0:
		DataBatch   = TrainingData[numberOfbatches*BatchSize:,:]
		LabelsBatch = TrainingLabels[numberOfbatches*BatchSize:,:]
		batchlistforData.append(DataBatch)
		batchlistforLabels.append(LabelsBatch)

	return batchlistforData, batchlistforLabels




def DataPipeline(PathForFirstDataset , PathForSecondDataset , BatchSize):
	FirstDataset           = ReadCSV(PathForFirstDataset)
	SecondDataset          = ReadCSV(PathForSecondDataset)
	# ----------------------------------------------------------------------------------------
	# Get the labels from each dataset and convert it to list and merge them
	LabelsForFirstDataset  = FirstDataset['label'].tolist()
	LabelsForSecondDataset = SecondDataset['label'].tolist()
	Labels                 = merge(LabelsForFirstDataset, LabelsForSecondDataset)
	TrainingLabels         = np.array(Labels)
	# ----------------------------------------------------------------------------------------
	# drop the labels columns from each dataset to create Training data and Training labels
	FirstDataset           = FirstDataset.drop('label', axis=1)
	SecondDataset          = SecondDataset.drop('label', axis=1)
	TrainingData           = pd.concat([FirstDataset, SecondDataset], axis=1).to_numpy()
	# ----------------------------------------------------------------------------------------
	ListOFTrainingBatches_X, ListOFTrainingBatches_Y = PrepareBatches(BatchSize, TrainingData, TrainingLabels)

	return ListOFTrainingBatches_X,ListOFTrainingBatches_Y

DataSetPaths = ['data1.csv' ,'data2.csv']


X , Y = DataPipeline(DataSetPaths[0],DataSetPaths[1] , 2)
print(len(X))
print(len(Y))
