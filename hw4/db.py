import sys
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from time import time
from itertools import cycle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.manifold import TSNE
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords

data_repo = sys.argv[1]
prediction = sys.argv[2]
stemmed = True

# TF-idf settings
max_features = None
max_df = 0.39
min_df = 2

# lsa settings
n_components = 20

# db settings
eps = 0.2
min_samples = 10

st = PorterStemmer()

all_titles = []
docs = []
stoplist = set('use definitly my you '.split())

print("Loading titles")
with open(data_repo + "/title_StackOverflow.txt", 'r') as f:
    for line in f:
        l = line.lower()
        l = l.translate(str.maketrans({key: ' ' for key in string.punctuation}))
        tokens = nltk.word_tokenize(l)
        #tokens = [word for word in tokens if word not in stoplist]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tmp = ''
        if stemmed:
            for token in tokens:
                tmp += st.stem(token) + ' '
        else:
            for token in tokens:
                tmp += token + ' '
        all_titles.append(tmp)

print("Loading docs")
with open(data_repo + "/docs.txt", 'r') as f:
    for line in f:
        l = line.lower()
        l = l.translate(str.maketrans({key: ' ' for key in string.punctuation}))
        tokens = nltk.word_tokenize(l)
        #tokens = [word for word in tokens if word not in stoplist]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tmp = ''
        if stemmed:
            for token in tokens:
                tmp += st.stem(token) + ' '
        else:
            for token in tokens:
                tmp += token + ' '
        docs.append(tmp)
all = list(np.concatenate((all_titles, docs), axis = 0))


print(all_titles[0])

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(max_df = max_df, max_features = max_features,
                             min_df = min_df, stop_words = 'english',
                             use_idf = True
                             )
print("done in %fs" % (time() - t0))
print()
vectorizer.fit(all)
X = vectorizer.transform(all_titles)
print("n_samples: %d, n_features: %d" % X.shape)

if n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy = False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)
    print("done in %fs" % (time() - t0))

    explained_varience = svd.explained_variance_ratio_.sum()
    print("Explained varience of the SVD step: {}%".format(
        int(explained_varience * 100)
    ))
    print()


db = DBSCAN(eps = eps, min_samples = min_samples, n_jobs = -1).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors_ = colors.cnames.keys()
for k, col in zip(unique_labels, colors_):
    if k == -1:
        col = 'k'
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markersize=4, markerfacecolor = col)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', marker = '.', markersize=1, markerfacecolor = col)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


test = []
with open(data_repo + "/check_index.csv", 'r') as f:
    for line in f:
        tokens = line.split(',')
        test.append(tokens)

ans = []
for i in range(1, len(test)):
    if labels[int(test[i][1])] != -1 and labels[int(test[i][2])] != -1:
        if labels[int(test[i][1])] == labels[int(test[i][2])]:
            ans.append(1)
        else:
            ans.append(0)
    else:
        ans.append(0)

ofile = open(prediction, 'w')
ofile.write("ID,Ans\n")
for i in range(len(ans)):
    ofile.write(str(i) + ',')
    ofile.write(str(ans[i]) + '\n')
