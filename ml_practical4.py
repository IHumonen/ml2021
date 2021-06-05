#!/usr/bin/env python
# coding: utf-8

# ## Task 1

# Read the documentation for scipy.optimize.minimize, paying special attention to the Jacobian argument jac. Who computes the gradient, the minimize function itself, or the developer using it?

# The developer computes the gradient through argument jac if it is callable: "If it is a callable, it should be a function that returns the gradient vector: jac(x, *args) -> array_like, shape (n,)"

# Run the following two examples; which performs better?

# In[1]:


import scipy.optimize, numpy.random


# In[2]:


def f(x):
  return x**2

def df(x):
  return 2*x

print(scipy.optimize.minimize(f, numpy.random.randint(-1000, 1000), jac=df))


# In[3]:


def f(x):
  return x**2

print(scipy.optimize.minimize(f, numpy.random.randint(-1000, 1000), jac=False))


# The first example performs better

# ## Task 2
# 
# Write in python the loss function for support vector machines from equation (7.48) of Daumé. You can use the following hinge loss surrogate:

# In[4]:


def hinge_loss_surrogate(y_gold, y_pred):
  return numpy.max(0, 1 - y_gold * y_pred)

def svm_loss(w, b, C, D):
    #D = [x, y]
    x = D[0]
    y_gold = D[1]
    y_pred = numpy.dot(w*x) + b
    l = 0.5*numpy.norm(w)**2 + C*numpy.sum(hinge_loss_surrogate(y_gold, y_pred))
    return l


# ## Task 3
# 
# Use scipy.optimize.minimize with jac=False to implement support vector machines.
# 

# In[5]:


def svm(D):
    # compute w and b with scipy.optimize.minimize and return them
    result = scipy.optimize.minimize(lambda x: svm_loss(x[:-1], x[-1], 1, D), numpy.random.randint(-10, 10), jac=False)
    return result['x'][:-1], result['x'][-1]


# ## Task 4
# Implement the gradient of svm_loss, and add an optional flag to svm to use it:

# gradient of $f$ = svm_loss:   
#         
# $\frac{\partial \xi_n}{\partial w} = 0 $ or $ -y_nx_n$      
#      
# $\frac{\partial \xi_n}{\partial b} = 0 $ or $ -y_n$      
#         
# $\frac{\partial f}{\partial w} = w + C \sum\limits_{n}{(\frac{\partial \xi_n}{\partial w}x_n)}$    
#      
# $\frac{\partial f}{\partial b} = C \sum\limits_{n}{\frac{\partial \xi_n}{\partial b}}$

# In[6]:


def gradient_hinge_loss_surrogate(y_gold, y_pred):
  if hinge_loss_surrogate(y_gold, y_pred) == 0:
    return [0, 0]
  else:
    return [-y_pred, -y_gold]

def gradient_hinge_loss_surrogate(y_gold, x, w, b):
  if hinge_loss_surrogate(y_gold, y_pred) == 0:
    return [0, 0]
  else:
    return [-y_gold*x, -y_gold]

def gradient_svm_loss(w, b, C, D):
    #D = [x, y]
    x = D[0]
    y_gold = D[1]
    y_pred = numpy.dot(w*x) + b
    l_w = w + C*numpy.sum(gradient_hinge_loss_surrogate(y_gold, x, w, b))
    l_b = C*numpy.sum(gradient_hinge_loss_surrogate(y_gold, x, w, b))
    return numpy.concatenate((l_w, l_b))

def svm(D, use_gradient=False):
    if use_gradient != False:
        use_gradient = lambda x: gradient_svm_loss(x[:-1], x[-1], 1, D)
    result = scipy.optimize.minimize(lambda x: svm_loss(x[:-1], x[-1], 1, D), numpy.random.randint(-10, 10), jac=use_gradient)
    return result['x'][:-1], result['x'][-1]


# In[ ]:




