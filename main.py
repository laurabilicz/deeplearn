from scipy import misc
import numpy as np
import glob

tmp_x = []
tmp_y = []
for image in glob.glob("./training/0/*.png"):
    tmp_x.append(misc.imread(image).reshape(784))
    tmp_y.append(0)
for image in glob.glob("./training/8/*.png"):
    tmp_x.append(misc.imread(image).reshape(784))
    tmp_y.append(1)

X = np.asarray(tmp_x, dtype=np.float32)
y = np.asarray(tmp_y, dtype=np.float32)

w = np.zeros((784), dtype=np.float32)

b = 0.;

num_of_cases = X.shape[0];

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))

def loss(w, X, y):
    w = np.matrix(w)
    X = np.matrix(X)
    y = np.matrix(y)
    y_ = sigmoid(np.dot(X, w.T))
    return np.power(y_,2) / 2

def update(w, X, y):
    w = np.matrix(w)
    X = np.matrix(X)
    y = np.matrix(y)
    y_ = sigmoid(np.dot(X, w.T))
    
    w_new = w - ( ( 0.1 * (y_-y) )* X)
    
    return w_new


for i in np.random.random_integers(num_of_cases-1, size=(num_of_cases,)):
    w = update(w,X[i],y[i])
    #print(w)


"""
score = 0
for i in range(num_of_cases):
    _w = np.matrix(w)
    _X = np.matrix(X[i])
    y_ = sigmoid(np.dot(_X, _w.T))
    #print (y[i] , y_)
    if y_ < 0.5:
        result = 0
    else:
        result = 1
    if result == y[i]:
        score += 1

print (score/X_test.shape[0]*100)
"""

##TESTING

tmp_x = []
tmp_y = []
for image in glob.glob("./test/0/*.png"):
    tmp_x.append(misc.imread(image).reshape(784))
    tmp_y.append(0)
for image in glob.glob("./test/8/*.png"):
    tmp_x.append(misc.imread(image).reshape(784))
    tmp_y.append(1)
    
X_test = np.asarray(tmp_x, dtype=np.float32)
y_test = np.asarray(tmp_y, dtype=np.float32)

score = 0
for i in range(X_test.shape[0]):
    _w = np.matrix(w)
    _X = np.matrix(X_test[i])
    y_ = sigmoid(np.dot(_X, _w.T))
    #print (y_test[i] , y_)
    if y_ < 0.5:
        result = 0
    else:
        result = 1
    if result == y_test[i]:
        score += 1

print (score/X_test.shape[0]*100)
