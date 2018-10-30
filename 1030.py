import pandas as pd

# The Dataset comes from:
# https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits



def load(path_test, path_train):
  with open(path_test, 'r')  as f: testing  = pd.read_csv(f)
  with open(path_train, 'r') as f: training = pd.read_csv(f)
  n_features = testing.shape[1]
  X_test  = testing.ix[:,:n_features-1]
  X_train = training.ix[:,:n_features-1]
  y_test  = testing.ix[:,n_features-1:].values.ravel()
  y_train = training.ix[:,n_features-1:].values.ravel()
  return X_train, X_test, y_train, y_test

def peekData(X_train):
  print ("Peeking your data...")
  fig = plt.figure()
  cnt = 0
  for col in range(5):
    for row in range(10):
      plt.subplot(5, 10, cnt + 1)
      plt.imshow(X_train.ix[cnt,:].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
      plt.axis('off')
      cnt += 1
  fig.set_tight_layout(True)
  plt.show()


def drawPredictions(model, X_train, X_test, y_train, y_test):
  fig = plt.figure()
  y_guess = model.predict(X_test)
  num_rows = 10
  num_cols = 5
  index = 0
  for col in range(num_cols):
    for row in range(num_rows):
      plt.subplot(num_cols, num_rows, index + 1)
      plt.imshow(X_test.ix[index,:].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
      fontcolor = 'g' if y_test[index] == y_guess[index] else 'r'
      plt.title('Label: %i' % y_guess[index], fontsize=6, color=fontcolor)
      plt.axis('off')
      index += 1
  fig.set_tight_layout(True)
  plt.show()

X_train, X_test, y_train, y_test = load('Datasets/optdigits.tes', 'Datasets/optdigits.tra')
import matplotlib.pyplot as plt
from sklearn import svm
peekData(X_train)
print ("Training SVC Classifier...")
svc = svm.SVC(kernel='linear', C=1, gamma=0.001)
svc.fit(X_train, y_train) 




print ("Scoring SVC Classifier...")
score = svc.score(X_test, y_test)
print ("Score:\n", score)
drawPredictions(svc, X_train, X_test, y_train, y_test)
true_1000th_test_value = y_test[999]
print ("1000th test label: ", true_1000th_test_value)
guess_1000th_test_value = svc.predict(X_test[999:1000])
print ("1000th test prediction: ", guess_1000th_test_value)
plt.imshow(X_test.ix[999,:].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
svc = svm.SVC(kernel='poly', C=1, gamma=0.001)
svc.fit(X_train, y_train) 
print ("Scoring SVC poly Classifier...")
score = svc.score(X_test, y_test)
print ("Score:\n", score)
svc = svm.SVC(kernel='rbf', C=1, gamma=0.001)
svc.fit(X_train, y_train) 
print ("Scoring SVC rbf Classifier...")
score = svc.score(X_test, y_test)
print ("Score:\n", score)




def load(path_img, path_lbl):
  import numpy as np
  from array import array
  import struct
  with open(path_lbl, 'rb') as file:
    magic, size = struct.unpack(">II", file.read(8))
    if magic != 2049:
      raise ValueError('Magic number mismatch, expected 2049, got {0}'.format(magic))
    labels = array("B", file.read())
  with open(path_img, 'rb') as file:
    magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
    if magic != 2051:
      raise ValueError('Magic number mismatch, expected 2051, got {0}'.format(magic))
    image_data = array("B", file.read())

  images = []
  for i in range(size): 
      images.append([0] * rows * cols)
  divisor = 1
  for i in range(size): 
      images[i] = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28,28)[::divisor,::divisor].reshape(-1)
  return pd.DataFrame(images), pd.Series(labels)