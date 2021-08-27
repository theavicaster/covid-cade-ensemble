import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sympy.solvers import solve
from sympy import Symbol
import matplotlib.pyplot as plt

train_dir = '/content/data-folders/train'
test_dir = '/content/data-folders/test'

data_format = K.image_data_format()
K.set_image_data_format(data_format)
np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_of_classes = 3

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                                   #samplewise_center=True,
                                   #samplewise_std_normalization=True
                                   )
test_datagen = ImageDataGenerator(
                                   #samplewise_center=True,
                                   #samplewise_std_normalization=True
                                 )


training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    seed = 42
   )

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle = False, 
    seed = 42
   )

inc_v3 = load_model('inc_v3.h5')
inc_rs = load_model('inc_rs.h5')
den = load_model('den.h5')

Ypred_inc_rs = inc_rs.predict_generator(test_set, verbose=1)
Ypred_inc_v3 = inc_v3.predict_generator(test_set, verbose=1)
Ypred_den = den.predict_generator(test_set, verbose=1)

ypred_inc_rs = np.argmax(Ypred_inc_rs, axis=1)
ypred_inc_v3 = np.argmax(Ypred_inc_v3, axis=1)
ypred_den = np.argmax(Ypred_den, axis=1)

num_classifiers = 3

from sympy.solvers import solve
from sympy import Symbol

from itertools import product
from numpy.linalg import norm

fuzzyMeasures = np.array([ 0.038 , 0.015, 0.074 ])

l = Symbol('l', real = True)
lam = solve(  ( 1 + l* fuzzyMeasures[0]) * ( 1 + l* fuzzyMeasures[1]) *( 1 + l* fuzzyMeasures[2])  - (l+1), l )
lam = lam[1]

Ypred_fuzzy = np.zeros(shape = Ypred_val_ang.shape, dtype = float)

for sample in range(0,Ypred_val_angvel.shape[0]):
  for classes in range(0,3):
    
    scores = np.array([ Ypred_inc_v3[sample][classes],Ypred_inc_rs[sample][classes],Ypred_den[sample][classes] ])
    permutedidx = np.flip(np.argsort(scores))
    scoreslambda = scores[permutedidx]
    fmlambda = fuzzyMeasures[permutedidx]

    ge_prev = fmlambda[0]
    fuzzyprediction = scoreslambda[0] * fmlambda[0]

    for i in range(1,2):
      ge_curr = ge_prev + fmlambda[i] + lam * fmlambda[i] * ge_prev
      fuzzyprediction = fuzzyprediction + scoreslambda[i] *(ge_curr - ge_prev)
      ge_prev = ge_curr

    fuzzyprediction = fuzzyprediction + scoreslambda[3] * ( 1 - ge_prev)

    Ypred_fuzzy[sample][classes] = fuzzyprediction

ypred_fuzzy = np.argmax(Ypred_fuzzy, axis=1)

from pycm import *
test_labels = test_set.labels
cm = ConfusionMatrix(actual_vector=test_labels, predict_vector=ypred_fuzzy)
