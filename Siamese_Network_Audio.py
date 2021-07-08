#!/usr/bin/env python
# coding: utf-8

# # 3. Making the distance matrix
# 

# ## Mount google drive



from google.colab import drive
drive.mount('/content/drive')


# ## Import libraries




import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations
import tensorflow.keras.backend as K


# ## Import the model and data



model_path = '/'
data_path_raw = ''
data_path_test = ''

# model_path = 'featureextractor'
# data_path_raw = 'raw_data.pkl'
# data_path_test = 'test_data.pkl'

## Import the model
model = keras.models.load_model(model_path, compile=False)

## Import the test data
with open(data_path_test, 'rb') as file:
    test_data = pickle.loads(file.read())

## Import the train data
with open(data_path_raw, 'rb') as file:
    x, y = pickle.loads(file.read())

## Set the train data size to the same length as the test data
test_size = len(test_data[0])
# test_size = 2**6 ## For testing
print(test_size)

x = x[:test_size] # subset the data to  the same size
y = y[:test_size]
print(len(x))


# ## Define the audio to mel spectogram function, the distance function and the performance function




def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))




def audio_to_spectro(vectors, hop_length=175, sr=11025, n_mels=64, n_fft=256): 
    """Takes 1D audio vectors and converts to spectrogramm"""
    ## Convert audio vectors of data set to mel_spectrogram
    records_MelSpec = []
    for audio_vec in vectors: 
        ## Normalize the vector
        # audio_vec = (audio_vec - audio_vec.mean()) / audio_vec.std()
        ## Convert to spectrogram
        S = librosa.feature.melspectrogram(
            np.array(audio_vec),
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        ## Add small number to avoid log(0)
        S = np.log(S + 1e-9) 
        records_MelSpec.append(S)
    records_MelSpec_arr = np.array(records_MelSpec)

    return records_MelSpec_arr




def distance_matrix(test_data, model):
    """ This function takes a test data set of 
    length (n) predict the variables and outputs
    them into a distance matrix """

    #prep
    N = len(test_data)
    M = np.zeros((N,N)) 
    w, h = test_data[0].shape

    featuremaps = [model.predict(image.reshape(1,int(w), int(h), 1)) for image in test_data]

    combs = list(combinations(range(N), 2))
    print(f'Amount of combinations to predict: {len(combs)}')
    for i, comb in enumerate(combs):
        # if i < 100: break ## For testing
        #reshape
        """Note: the reshape is necessary as the prev. came as pairs 
        - i.e. (1,x,x,1) vs (x,x,1)"""
        a = featuremaps[comb[0]]
        b = featuremaps[comb[1]]

        distance = euclidean_distance([a,b])

        ## Enter into Matrix, chanmge the prediction from simularity to distance
        M[comb[0], comb[1]] = M[comb[1], comb[0]] = distance
        if i % 1000 == 0:
            print(f"Processed: {i}")

    print('Done!')
    return M




def calculate_performance_numpy(distances_matrix, labels):
    """
    For a given distance matrix and labels of all samples, this function calculates two performance measures:
     - The mean CMC scores for n = [1, 3, 5, 10]
     - A mean accuracy metric. This metric calculates how many of the k samples that belong to the same class are among
       the first k ranked elements.

    For N samples, the arguments to this function are:
    :param distances_matrix: A NumPy array defining a distance matrix of floats of size [N, N].
    :param labels: An array of integers of size N.

    """
    assert distances_matrix.shape[0] == distances_matrix.shape[1], "The distance matrix must be a square matrix"
    assert len(labels) == distances_matrix.shape[0], "The size of the matrix should be equal to number of labels"

    # Create a bool matrix (mask) where all the elements are True, except for the diagonal.
    mask = np.logical_not(np.eye(labels.shape[0], dtype=np.bool))

    # Create a bool matrix (label_equal) with value True in the position where the row and column (i, j)
    # belong to the same label, except for i = j.
    label_equal = labels[np.newaxis, :] == labels[:, np.newaxis]
    label_equal = np.logical_and(label_equal, mask)

    # Add the maximum distance to the diagonal.
    distances_matrix = distances_matrix + np.logical_not(mask) * np.max(distances_matrix.flatten(), axis=-1)

    # Get the sorted indices of the distance matrix for each sample.
    sorted_indices = np.argsort(distances_matrix, axis=1)

    # Get a bool matrix where the bool values in label_equal are sorted according to sorted_indices
    sorted_equal_labels_all = np.zeros(label_equal.shape, dtype=bool)
    for i, ri in enumerate(sorted_indices):
        sorted_equal_labels_all[i] = label_equal[i][ri]

    # Calculate the mean CMC scores for k=[1, 3, 5, 10] over all samples
    # The score is 1 if a sample j with the same label as i is in the first k ranked positions. It i s 0 otherwise.
    cmc_scores = np.zeros([4])
    for sorted_equal_labels in sorted_equal_labels_all:
        # CMC scores for a sample
        score = np.asarray([np.sum(sorted_equal_labels[:n]) > 0 for n in [1, 3, 5, 10]])
        # Update running average
        cmc_scores = cmc_scores + score
    cmc_scores /= len(sorted_equal_labels_all)

    # Calculate the accuracy metric

    # Calculate how many samples are there with the same label as any sample i.
    num_positives = np.sum(label_equal, axis=1, dtype=np.int)
    num_samples = len(sorted_equal_labels_all)

    # Calculate the average metric by adding up how many labels correspond to sample i in the first n elements of the
    # ranked row. So, if all the first n elements belong to the same labels the sum is n (perfect score).
    acc = 0
    for i, n in enumerate(num_positives):
        acc = acc + np.sum(sorted_equal_labels_all[i, :n], dtype=np.float32) / (n * num_samples)

    return cmc_scores, acc


# ## Training set
# ### Convert the training data to mel spectogram and make a distance matrix


train_spectro = audio_to_spectro(x)
train_matrix = distance_matrix(train_spectro, model)

#Check if we are predicting the distance 
#Store Word doc in correct format ~ see https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html


## Print a the dataframe for a sanity check
matrix_df = pd.DataFrame(train_matrix)
matrix_df



import matplotlib.pylab as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(20,16))
ax = sns.heatmap(train_matrix, linewidth=0)
plt.show()


# ### Calculate the performance on the training set



calculate_performance_numpy(train_matrix, y)


# ## Test set
# ### Convert to spectogram, predict and make a matrix on the final test set, save to answer.txt



test_spectro = audio_to_spectro(test_data[0])
test_matrix = distance_matrix(test_spectro, model)

np.savetxt('answer.txt', test_matrix, fmt='%.18e', delimiter=';')







