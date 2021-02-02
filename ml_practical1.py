#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# Task 1

# In[3]:


class Tree:
  '''Create a binary tree; keyword-only arguments `data`, `left`, `right`.

  Examples:
    l1 = Tree.leaf("leaf1")
    l2 = Tree.leaf("leaf2")
    tree = Tree(data="root", left=l1, right=Tree(right=l2))
  '''

  def leaf(data):
    '''Create a leaf tree
    '''
    return Tree(data=data)

  # pretty-print trees
  def __repr__(self):
    if self.is_leaf():
      return "Leaf(%r)" % self.data
    else:
      return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right) 

  # all arguments after `*` are *keyword-only*!
  def __init__(self, *, data = None, left = None, right = None):
    self.data = data
    self.left = left
    self.right = right

  def is_leaf(self):
    '''Check if this tree is a leaf tree
    '''
    return self.left == None and self.right == None

  def children(self):
    '''List of child subtrees
    '''
    return [x for x in [self.left, self.right] if x]

  def depth(self):
    '''Compute the depth of a tree
    A leaf is depth-1, and a child is one deeper than the parent.
    '''
    return max([x.depth() for x in self.children()], default=0) + 1


# In[4]:


l11 = Tree(data='like')
l12 = Tree(data='nah')
l13 = Tree(data='nah')
l14 = Tree(data='like')
l21 = Tree(data='mornig', left=l11, right=l12)
l22 = Tree(data='likedOtherSys', left=l13, right=l14)
l31 = Tree(data='takenOtherSys', left=l21, right=l22)
l32 = Tree(data='like')
l41 = Tree(data='isSystems', left=l31, right=l32)


# In[5]:


print(l41)


# Task 2

# In[6]:


data = pd.read_csv(r'C:\Users\Иннокентий\Documents\машинка2021\data.csv')


# In[7]:


print(data)


# In[8]:


def is_ok(x):
    if  x >= 0:
        return True
    else:
        return False


# In[9]:


data['ok'] = data['rating'].apply(is_ok)


# In[10]:


data.head()


# Task 3

# In[11]:


import numpy as np


# In[12]:


def single_feature_score(data, goal, feature):
    y = data[goal]
    X_direсt = data[feature]
    X_inverse = -data[feature]
    direct_score = X_direсt == y
    inverse_score = X_inverse == y
    return max(np.mean(direct_score), np.mean(inverse_score))


# In[14]:


print(single_feature_score(data, 'ok', 'morning'))


# In[15]:


def best_feature(data, goal, features):
  # optional: avoid the lambda using `functools.partial`
  return max(features, key=lambda f: single_feature_score(data, goal, f))


# In[16]:


print('The best feature is \"' + best_feature(data, 'ok', data.columns[1:-1]) + '\"')


# In[17]:


def wost_feature(data, goal, features):
  # optional: avoid the lambda using `functools.partial`
  return min(features, key=lambda f: single_feature_score(data, goal, f))


# In[18]:


print('The worst feature is \"' + wost_feature(data, 'ok', data.columns[1:-1]) + '\"')


# Task 4

# In[19]:


data = data[data.columns[1:]] #remove "rating" column


# In[20]:


def decision_tree_train(data, remaining_features):
    remaining_features = list(remaining_features)
    y = data['ok']
#    X = data[data.columns[:-1]]
    number_of_true = np.sum(y)
    if number_of_true < len(y) - number_of_true: 
        guess = False  
    else:
        guess = True  
    if len(set(y)) == 1: 
        return Tree(data=guess) 
    else:
        if len(remaining_features) == 0:  
            return Tree(data=guess) 
        else:
            scores = {}
            for f in remaining_features:
                scores[f] = single_feature_score(data, 'ok', f)
            current_feature = max(remaining_features, key=lambda x: scores[x])            
            no = data[data[current_feature] == False]
            yes = data[data[current_feature] == True]
            remaining_features.remove(current_feature)
            left = decision_tree_train(no, remaining_features)
            right = decision_tree_train(yes, remaining_features)
            return Tree(data=current_feature, left=left, right=right)


# In[21]:


tree_t4 = decision_tree_train(data, data.columns[:-1])


# In[22]:


print(tree_t4)


# In[23]:


def decision_tree_test(tree, test_point):
    if tree.is_leaf():
        return tree.data
    else:
        feature = tree.data
        if test_point[feature] == False:
            return decision_tree_test(tree.left, test_point)
        else:
            return decision_tree_test(tree.right, test_point)


# In[24]:


def tree_score(tree, data):
    y_test = data['ok']
    predicted = data.apply(lambda x: decision_tree_test(tree, x), axis=1)
    return np.mean(y_test == predicted)


# In[25]:


print(tree_score(tree_t4, data))


# In[26]:


feature_scores = {}
for feature in data.columns[:-1]:
    feature_scores[feature] = single_feature_score(data, 'ok', feature)


# In[27]:


for feature in sorted(data.columns[:-1], key=lambda f: feature_scores[f]):
    print(feature, feature_scores[feature])


# Task 5

# In[28]:


def decision_tree_train_maxdepth(data, remaining_features, maxdepth=np.inf):
    remaining_features = list(remaining_features)
    y = data['ok']
    number_of_true = np.sum(y)
    if number_of_true < len(y) - number_of_true: 
        guess = False  
    else:
        guess = True  
    if len(set(y)) == 1: 
        return Tree(data=guess) 
    else:
        if len(remaining_features) == 0 or maxdepth <= 0:  
            return Tree(data=guess) 
        else:
            scores = {}
            for f in remaining_features:
                scores[f] = single_feature_score(data, 'ok', f)
            current_feature = max(remaining_features, key=lambda x: scores[x])            
            no = data[data[current_feature] == False]
            yes = data[data[current_feature] == True]
            remaining_features.remove(current_feature)
            left = decision_tree_train_maxdepth(no, remaining_features, maxdepth=maxdepth-1)
            right = decision_tree_train_maxdepth(yes, remaining_features, maxdepth=maxdepth-1)
            return Tree(data=current_feature, left=left, right=right)


# In[29]:


depths = range(0, tree_t4.depth())
depth_scores = []
for depth in depths:
    tree = decision_tree_train_maxdepth(data, data.columns[:-1], maxdepth=depth)
    depth_scores.append(tree_score(tree, data))


# In[31]:


pl = pd.DataFrame({'depth' : depths, 'score' : depth_scores})
print(pl)


# In[32]:


pl.plot(x='depth', y='score')

