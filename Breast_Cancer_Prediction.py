#This program uses data of characteristics of cancer tumors to build a predictive model.
#The algorithm can predict any input characteristic based on data of other chosen characteristics.
#Two methods are used, SVD least squares regression, and Newtons method of convergence.


import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time

#all_headers is a list of all different characteristics of a tumor
all_headers = ['patient ID', 'Malignant/Benign', 'radius (mean)', 'radius (stderr)', 'radius (worst)', 'texture (mean)', 'texture (stderr)', 'texture (worst)', 'perimeter (mean)', 'perimeter (stderr)', 'perimeter (worst)', 'area (mean)', 'area (stderr)', 'area (worst)', 'smoothness (mean)', 'smoothness (stderr)', 'smoothness (worst)', 'compactness (mean)', 'compactness (stderr)', 'compactness (worst)', 'concavity (mean)', 'concavity (stderr)', 'concavity (worst)', 'concave points (mean)', 'concave points (stderr)', 'concave points (worst)', 'symmetry (mean)', 'symmetry (stderr)', 'symmetry (worst)', 'fractal dimension (mean)', 'fractal dimension (stderr)', 'fractal dimension (worst)']

# Load the data
tumor_data = pd.io.parsers.read_csv("breast-cancer-train.dat", header=None, names=all_headers)


#subset_headers is a list of the characteristics we wish to use when making predictions
subset_headers = []
#random_number_parameters randomizes the number of characteristics we wish to use when building subset_headers
random_number_parameters = np.random.randint(len(all_headers)-3)
for i in range(random_number_parameters):
    subset_headers.append(all_headers[i+3])
    
#b_header is the characteristic we wish to predict
b_header = 'radius (mean)'


#construct matrix A--data from wanted relevant tumor features
A= np.zeros(len(subset_headers)*len(tumor_data)).reshape(len(tumor_data),len(subset_headers))
for i in range(len(subset_headers)):
    A[:,i] = tumor_data[subset_headers[i]]
    
#construct vector b--data from one specific tumor feature
b= np.zeros(len(tumor_data[b_header]))
for i in range(len(tumor_data[b_header])):
    b[i] = tumor_data[b_header][i]



#SVD Least Squares Regression Algorithm for system Ax=b
def weights_SVD(A,b):
#find x vector solution, which is the weights vector of the predictive model, such that Ax=b using SVD linear least squares
    u,s,vt = la.svd(A,full_matrices=False)
    s_inverse = np.diag(1/s)
    x = vt.T@s_inverse@u.T@b
    return x

svd1 = time.time()
x = weights_SVD(A,b)
svd2 = time.time()

#relative error of the predicted b_header vs the true b_header when using SVD least squares regression
SVD_relative_error = la.norm(A@x-b)/la.norm(b)
print('SVD least squares regression relative error:',SVD_relative_error)
print("time taken for SVD algorithm (seconds):",svd2-svd1)


#construct the objective function, gradient, and Hessian to be used in Newtons Algorithm


#the gradient in the Newtons method: gradient(x,A,b)
def gradient(x,A,b):
    grad = -1*A.T@(b-A@x)
    return grad
#the hessian in Newtons method: hessian(x,A,b)
def hessian(x,A,b):
    H = A.T@A
    return H

#convergence tolerance for Newtons method
tol = .001
x0 = np.random.rand(len(x))
#solve for 
def newtons_method(x0,A,b,x):
    x_new = x0
    x_prev = np.random.randn(x0.shape[0])
    niter = 0
    while(la.norm(x_new-x)>tol):
        x_prev = x_new
        s = -la.solve(hessian(x_prev,A,b), gradient(x_prev,A,b))
        x_new = x_prev + s
        niter += 1
    return x_new,niter

n1 = time.time()    
x_n,niter = newtons_method(x0,A,b,x)
n2 = time.time()

#relative error of the predicted b_header vs the true b_header when using Newtons method
Newtons_method_relative_error = la.norm(A@x_n-b)/la.norm(b)
print('Newtons Method relative error:',Newtons_method_relative_error)
print("time taken for newton algorithm (seconds):",n2-n1)
