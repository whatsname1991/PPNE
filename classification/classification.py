import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn import preprocessing

from collections import defaultdict

dir = ""
# featurereader = open(dir + "embedding.citeseer.TADW.txt")
# samples = []
# for sample in featurereader:
#     ss = sample.strip().split('  ')
#     feature = []
#     for s in ss:
#         feature.append(float(s))
#     samples.append(feature)
# featurereader.close()
# print "Feature Loaded"
# samples = np.array(samples)

featurereader = open(dir + "embedding.citeseer.inequation.constraints.txt")
print 'constraints'
samples = []
nodeDict = {}
nodeCount = 0
dimension = 0
for sample in featurereader:
    ss = sample.strip().split(' ')
    if len(ss) <= 2:
        nodeCount = int(ss[0]) - 1
        dimension = int(ss[1])
        continue;
    if ss[0] == "</s>":
        continue
    nodeid = int(ss[0])
    feature = []
    for i in range(1,len(ss)):
        feature.append(float(ss[i]))
    nodeDict[nodeid] = feature
featurereader.close()
for i in range(0,nodeCount):
    samples.append(nodeDict[i])
print "Feature Loaded"
print dimension
samples = np.array(samples)


# featurereader = open(dir + "embedding.LINE.citeseer.txt")
# print "DeepWalk or LINE"
# samples = []
# nodeDict = {}
# nodeCount = 0
# dimension = 0
# for sample in featurereader:
#     ss = sample.strip().split(' ')
#     if len(ss) <= 2:
#         nodeCount = int(ss[0])
#         dimension = int(ss[1])
#         continue;
#     nodeid = int(ss[0])
#     feature = []
#     for i in range(1,len(ss)):
#         feature.append(float(ss[i]))
#     nodeDict[nodeid] = feature
# featurereader.close()
# for i in range(0,nodeCount):
#     samples.append(nodeDict[i])
# print "Feature Loaded"
# samples = np.array(samples)

labels = []
labelreader = open(dir + "group.txt")
for line in labelreader:
    ss = line.split('\t')
    label = ss[1]
    labels.append(float(label))
labels = np.array(labels)
print "Label Loaded"

shuffles = []
number_shuffles = 5
for x in range(number_shuffles):
    shuffles.append(skshuffle(samples, labels))

all_results = defaultdict(list)
training_percents = [0.1,0.2,0.3]

# uncomment for all training percents
#training_percents = numpy.asarray(range(1,10))*.1
for train_percent in training_percents:
    print str(train_percent) + " Begin!"
    for shuf in shuffles:
        X, y = shuf
        training_size = int(train_percent * X.shape[0])
        X_train = X[:training_size, :]
        y_train = y[:training_size]
        X_test = X[training_size:, :]
        y_test = y[training_size:]
        print "training size:" + str(training_size)

        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                  max_iter=-1, probability=False, random_state=None, shrinking=False,
                  tol=0.0001, verbose=False)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        results = {}
        average = "accuracy"
        #results[average] = f1_score(y_test,preds, average="micro")
        results[average] = accuracy_score(y_test,preds)
        all_results[train_percent].append(results)

print 'Results, feature count', samples.shape[1]
print '-------------------'
for train_percent in sorted(all_results.keys()):
    print 'Train percent:', train_percent
    sum = 0.0
    for x in all_results[train_percent]:
        print  x
        sum += x["accuracy"]
    sum = sum / number_shuffles
    print "avg_accuracy: " + str(sum)
    print '-------------------'