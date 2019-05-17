'''
The code we used for extracting the PCA representations. It can be run, provided that you have the pre-extracted ResNet50 features saved locally in path/features_resnet/clean
, path/features_resnet/imagenet/ and path/features_resnet/attack_type_attack_level, for the clean Animals with attributes, ImageNet and the corrupted data respectively.
'''

import numpy as np
from sklearn.decomposition import PCA

np.random.seed(0)

# Load classes and attributes:
a = open('files/classes.txt').read()
a = "%r"%a
b = a.split('\\')
animals = [b[j] for j in [2*i + 1 for i in range(50)]]
for ind in range(len(animals)):
    animals[ind] = animals[ind][1:]
    
binary_matrix = np.loadtxt('files/predicate-matrix-binary.txt')

# Load the data:
# - features is a n_points x n_dimensions data matrix
# - labels is a n_points x 85 matrix, containing the 85 features for each data 
#   point
# - labels_animals indicates the animal class of each input

features_clean = np.zeros((1,2048))
labels = np.zeros((1, 85))
an_indexes = np.zeros(1)

for ind, animal in enumerate(animals):
    data = np.load('features_resnet/clean/_' + animal + '.npz')
    lst = data.files
    for item in lst:
        features_clean = np.concatenate((features_clean, data[item]), axis = 0)
        n = data[item].shape[0]
        labels = np.concatenate((labels, np.tile(binary_matrix[ind,:], (n,1))), axis = 0)
        an_indexes = np.concatenate((an_indexes, np.repeat(ind, n)), axis = 0)
       
features_clean = features_clean[1:, :]
labels = labels[1:,:]
an_indexes = an_indexes[1:]

# Shuffle the data points randomly

shuffle = np.random.permutation(features_clean.shape[0])
features_clean = features_clean[shuffle,:]
labels = labels[shuffle,:]
labels_animals = an_indexes[shuffle]

#################################################################################
        
# Do PCA on the IMAGENET features

data = np.load('features_resnet/imagenet/features.npz')
lst = data.files
for item in lst:
    features_imagenet = data[item]   

wanted_dim = 100

pca = PCA(n_components = wanted_dim)
pca.fit(features_imagenet)

###################################################################################

# Project data according to the projection discovered on the imagenet data.
# Save the data

features_clean = pca.transform(features_clean)
features_clean = np.concatenate((features_clean, np.ones((features_clean.shape[0],1))), axis = 1)

# Transform and save clean data:

np.save('files/all_features.npy', features_clean)
np.save('files/all_labels.npy', labels)
np.save('files/all_labels_animals.npy', labels_animals)


#################################################################################
# Load poisoned data - blured:

for sigma in [2,4,6,8]:
    # Load data
    
    features_blured = np.zeros((1,2048))
    for ind, animal in enumerate(animals):
        data = np.load('features_resnet/blured_' + str(sigma) + '/_' + animal + '.npz')
        lst = data.files
        for item in lst:
            features_blured = np.concatenate((features_blured, data[item]), axis = 0)
            
    features_blured = features_blured[1:, :]
    features_blured = features_blured[shuffle,:]
    
    # Project data as before
    features_blured = pca.transform(features_blured)
    features_blured = np.concatenate((features_blured, np.ones((features_blured.shape[0],1))), axis = 1)
    
    # Save data
    np.save('files/all_features_blured_' + str(sigma) + '.npy', features_blured)


# Load poisoned data - pixels:

for sigma in [10, 30, 50]:
    # Load data
    
    features_pixels = np.zeros((1,2048))
    for ind, animal in enumerate(animals):
        data = np.load('features_resnet/pixels_' + str(sigma) + '/_' + animal + '.npz')
        lst = data.files
        for item in lst:
            features_pixels = np.concatenate((features_pixels, data[item]), axis = 0)
            
    features_pixels = features_pixels[1:, :]
    features_pixels = features_pixels[shuffle,:]
    
    # Project data as before
    features_pixels = pca.transform(features_pixels)
    features_pixels = np.concatenate((features_pixels, np.ones((features_pixels.shape[0],1))), axis = 1)
    
    # Save data
    np.save('files/all_features_pixels_' + str(sigma) + '.npy', features_pixels)

# Load RGB data:

features_rgb = np.zeros((1,2048))
for ind, animal in enumerate(animals):
    data = np.load('features_resnet/RGB/_' + animal + '.npz')
    lst = data.files
    for item in lst:
        features_rgb = np.concatenate((features_rgb, data[item]), axis = 0)
        
features_rgb = features_rgb[1:, :]
features_rgb = features_rgb[shuffle,:]

# Project data as before
features_rgb = pca.transform(features_rgb)
features_rgb = np.concatenate((features_rgb, np.ones((features_rgb.shape[0],1))), axis = 1)

# Save data
sigma = 0
np.save('files/all_features_RGB_' + str(sigma) + '.npy', features_rgb)
