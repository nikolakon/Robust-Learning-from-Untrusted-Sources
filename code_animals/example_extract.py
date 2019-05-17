#!/usr/bin/python

'''
This is an example script that shows how to extract features for an image dataset, using the Tensornets package. Optionally, various types of
image corruptions can be applied.
'''

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import tensornets as nets
from tensornets import utils


import sys

# uses the fabulous: https://github.com/taehoonlee/tensornets

import numpy as np

# Load the animal classes

a = open('/fs3/group/chlgrp/datasets/Animals_with_Attributes2/classes.txt').read()
a = "%r"%a
b = a.split('\\')
c = [b[j] for j in [2*i + 1 for i in range(50)]]
for ind in range(len(c)):
    c[ind] = c[ind][1:]

# We used ResNet50

try:
    modelname = sys.argv[1]
except IndexError:
    modelname = "ResNet50"

if 'large' in modelname:
    target_size, crop_size = 331, 331
    inputs = tf.placeholder(tf.float32, [None, crop_size, crop_size, 3]) # extra large
    model = nets.NASNetAlarge(inputs)
else:
    target_size, crop_size = 256, 224
    inputs = tf.placeholder(tf.float32, [None, crop_size, crop_size, 3]) # normal size
    #model = nets.ResNet50(inputs)

if modelname == 'NASNetAlarge':
    model = nets.NASNetAlarge(inputs)
    featlayer = model.get_outputs()[-4] # ?x4032
    bs = 250
elif modelname == 'VGG19':
    model = nets.VGG19(inputs)
    featlayer = model.get_outputs()[-4] # ?x4096
    bs = 100
elif modelname == 'MobileNet25':
    model = nets.MobileNet25(inputs)
    featlayer = model.get_outputs()[-4]
    bs = 2500
elif modelname == 'SqueezeNet':
    model = nets.SqueezeNet(inputs)
    featlayer = model.get_outputs()[-2]
    bs = 1000
elif modelname == 'ResNet50':
    model = nets.ResNet50(inputs)
    featlayer = model.get_outputs()[-3] # 'avgpool:0', ?x2048
    bs = 500
elif modelname == 'InceptionResNet2':
    model = nets.InceptionResNet2(inputs)
    featlayer = model.get_outputs()[-4]
    bs = 250
else:
    print('Unknown model: ',modelname)
    raise SystemExit
 
model_pretrained = model.pretrained()

# ind selects which animal to extract features for
try:
    ind = int(sys.argv[2])
    animal = c[ind]
    IMAGE_PATTERN = "/path/to/data/Animals_with_Attributes2/JPEGImages/" + animal + "/*.jpg"
except IndexError:
    IMAGE_PATH = '/something/' 
    IMAGE_PATTERN = '%something_*.JPEG' % IMAGE_PATH

try:
    outputname = sys.argv[3]
except IndexError:
    outputname = modelname 

# Blur and noise level are used to select a poisoning type and level; extras is used for the RGB swap.
try:
    blur = int(sys.argv[4])
except IndexError:
    blur = 0

if blur>0:
    from scipy.ndimage.filters import gaussian_filter

try:
    noise_level = int(sys.argv[5])
except IndexError:
    noise_level = 0

try:
    extras = int(sys.argv[6])
except IndexError:
    extras = 0

import glob
FILENAMES = sorted(glob.glob(IMAGE_PATTERN))
nimages = len(FILENAMES)
nchunks = (nimages+bs-1)//bs

# Choose what to save - we only needed the features 
save_ascii = False
save_binary = False
save_max = False
save_feats = True

PREDS,FEATS = [],[]
with tf.Session() as sess:
    sess.run(model_pretrained)  # equivalent to nets.pretrained(model)
    for i in xrange(nchunks):
        #if i > 0:
        #    break
        images = []
	# Load an image and potentially corrupt it
        for filename in FILENAMES[i*bs:(i+1)*bs]:
            img = utils.load_img(filename, target_size=target_size, crop_size=crop_size)
            if blur>0:
                img = gaussian_filter(img, sigma=blur)
            if noise_level>0:
                noise = np.random.standard_normal(img.shape) # noise same size as image
                img = np.clip( img+noise_level*noise, 0., 255.) # img is [0.,255.]
            elif noise_level < 0 and noise_level >= -100: # specify percentage of hot/cold pixels
                orig_shape = img.shape
                img = img.reshape(-1, img.shape[-1])  # flatten into 1D list of BGR
                num_pixels = len(img)
                num_defects = (-noise_level*num_pixels)//100 
                idx = np.random.choice(num_pixels, num_defects, replace=False)
                img[idx[::2]] = [0,0,0] # half pixels black
                img[idx[1::2]] = [255,255,255] # half pixels white
                img = img.reshape(orig_shape)
                #plt.imsave('deadpixels.png',img[0].astype(np.uint8))
            
            # 1: flip channels,  2: flip horizontal, 3: flip vertical, 4: rotate 180
            if extras == 1: # flip channels
                img = img[:,:,:,::-1] # RGB <-> BGR
            if extras == 2 or extras == 4: # flip horizontally
                img = img[:,:,::-1,:]  
            if extras == 3 or extras == 4: # flip vertically 
                img = img[:,::-1,:,:]
                
            images.append( img.squeeze() )
        
        images = np.asarray(images)
        images = model.preprocess(images)  # equivalent to img = nets.preprocess(model, img)

        if save_feats:
            preds,feats = sess.run([model,featlayer], {inputs: images})
            FEATS.extend(feats)
        PREDS.extend(preds)

        print('Processed chunk %04d of %04d' % (i, nchunks))
        print(utils.decode_predictions(preds, top=1)[0])

        if save_ascii:
            ascii_outputname = outputname
            if nchunks>1:
                ascii_outputname = ascii_outputname.replace('.txt','')
                ascii_outputname = ascii_outputname+'-%04d_of_%04d.txt'%(i, nchunks)

            print("Saving ", ascii_outputname)
            np.savetxt(ascii_outputname,  preds)  # 00_of_12
    
    PREDS = np.asarray(PREDS)
    if save_binary:
        binary_outputname = outputname.replace('.txt','')+'.npz'
        np.savez_compressed(binary_outputname, PREDS) # binary

    if save_max:
        PREDSmax = PREDS.max(axis=1)
        PREDSclass = PREDS.argmax(axis=1)
        np.savetxt(outputname.replace('.txt','')+'-max.txt', PREDSmax, fmt='%.20f')
        np.savetxt(outputname.replace('.txt','')+'-cls.txt', PREDSclass, fmt='%d')
    
    # Save features
    FEATS = np.asarray(FEATS)
    if save_feats:
        binary_outputname = outputname.replace('.txt','') + '_' + animal + '.npz'
        np.savez_compressed(binary_outputname, FEATS)

