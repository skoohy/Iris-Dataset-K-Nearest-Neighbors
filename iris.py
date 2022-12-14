import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from confidence import confidence1 as confidence1 # Confidence Calculator 1
from confidence import confidence2 as confidence2 # Confidence Calculator 1
from accuracy import accuracy as accuracy         # Accuracy Calculator
plt.style.use(['science', 'notebook'])
plt.rcParams['axes.linewidth'] = 2
plt.rc('axes', edgecolor='black')
random.seed(87)

iris_data  = np.loadtxt("iris.data", delimiter=',', usecols=[0,1,2,3])
# [sepal length  |  sepal width  |  petal length  |  petal width] 
iris_class = np.loadtxt("iris.data", delimiter=',', usecols=[4], dtype=str)
# [setosa  |  versicolor  |  virginica]

iris_data = iris_data.tolist() # Turn iris_data into a list

# Create list containing iris data and the class/label for each observation 
label = np.concatenate((np.zeros(50), np.ones(50), np.full(50, 2)))
for i in range(len(iris_data)):
    iris_data[i].append(label[i])
    
random.shuffle(iris_data)

test_size  = 0.25 # Percentage that 'iris_data' will be used for testing

# First 75% of the data
train_data = iris_data[:-int(test_size*len(iris_data))] 

# Last 25% of the data
test_data  = iris_data[-int(test_size*len(iris_data)):] 

train_set  = {0: [], 1: [], 2: []}
test_set   = {0: [], 1: [], 2: []} 

# Append data to the above dicts except the last point (the label)
for i in train_data:
    train_set[i[-1]].append(i[:-1])  
for i in test_data:
    test_set[i[-1]].append(i[:-1]) 

k = list(range(4, 26))

# Confidence is calcurted only when the algortithsm guess was wrong
# Calculated and append confidence for different k (k is stored here)
confidence_k = []
for ki in k:
    confidence_k.append(confidence1(train_set, test_set, ki))

# Calculated and append confidence for different k (k is not stored here)
confidence = []
for ki in k:
    confidence.append(confidence2(train_set, test_set, ki))    
confid = list(np.concatenate(confidence).flat) # elimated list of lists

# Count the number of times knn was wrong for different k 
occurances = []
for x in confidence_k:
    occurances.append(len(x))
print(occurances)    
  
# Data used to produce histogram for the frequency for the number of times 
# knn was iinccorect for different k example:
# if for k=5 the alogirithm was wrong 2 times then we get [5, 5]    
hist_data = []
for ki in k:
    hist_data += [ki] * occurances[ki-4]
print(hist_data)


#################################### Plots ####################################
# These plots are built off this specific random seed, changes may need to 
# be made for different seeds

figure(figsize=(6, 4), dpi=200)
plt.hist(hist_data, bins = 22, edgecolor='black', color='green', alpha=0.65, linewidth=2)
plt.xticks([x for x in range(min(hist_data), max(hist_data)+1, 2)])
plt.xlabel("k")
plt.grid(False)
plt.ylabel("Occurances for a wrong guess", fontsize=15)
plt.show()


figure(figsize=(6, 4), dpi=200)
plt.plot(k, [accuracy(train_set, test_set, ki) for ki in k], linewidth=3, color='black')
plt.xticks([x for x in range(min(k), max(k)+1, 2)])
plt.xlim(min(k), max(k))
plt.xlabel("k")
plt.grid(True)
plt.ylabel("Accuracy")
plt.show()

figure(figsize=(6, 4), dpi=200)
plt.scatter(hist_data, confid, linewidth=1.5, color='green')
plt.xlim(3.5, 25.5)
plt.ylim(0.45, 1.05)
for ki in range(4, 25):
    plt.plot([ki+0.5, ki+0.5], [min(confid)-.05, max(confid)+.05], color='black', linewidth=2, alpha=0.4, linestyle='--')
plt.scatter(25, 0.52, linewidth=1.5, color='red', label='Overlapping Points')
plt.scatter(23, 0.5217391304347826,  linewidth=1.5, color='red')
plt.xticks([x for x in range(min(k), max(k)+1, 2)])
plt.legend(loc="upper right", fontsize=12, frameon=True)
plt.grid(False)
plt.ylabel("Confidence for inccorect guesses", fontsize=15)
plt.xlabel("k")
plt.show()
