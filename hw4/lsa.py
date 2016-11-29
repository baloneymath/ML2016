import sys
import numpy as np
import string
import nltk
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
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

data_repo = sys.argv[1]
prediction = sys.argv[2]
stemmed = True

# Tf-Idf settings
max_features = None
max_df = 0.4
min_df = 2

# lsa settings
n_components = 20

# k-means settings
n_clusters = 100
n_init = 100
minibatch = False
batch_size = 2000
max_iter = 1000
verbose = False

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

if minibatch:
    km = MiniBatchKMeans(n_clusters = n_clusters, init = 'k-means++',
                         n_init = n_init, init_size = 1000, n_jobs = -1,
                         batch_size = batch_size, verbose = verbose)
else:
    km = KMeans(n_clusters = n_clusters, init = 'k-means++', n_jobs = -1,
                max_iter = max_iter, n_init = n_init, verbose = verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %.3fs" % (time() - t0))
print()

print("Top terms per cluster:")
if n_components:
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
else:
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(n_clusters):
    print("Cluster %d:" % i, end = '')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end = '')
    print()


colors_ = colors.cnames.keys()
fig = plt.figure(figsize = (10, 10))
fig.subplots_adjust(left = 0.01, right = 0.99, bottom = 0.02, top = 0.9)
k_means_cluster_centers = np.sort(km.cluster_centers_, axis = 0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_clusters), colors_):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'o', markerfacecolor = col,
            markersize = 4)
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor = col,
             markeredgecolor = 'k', markersize = 8)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.show()

'''
result = km.predict(X)
test = []
ans = []
with open(data_repo + "/check_index.csv", 'r') as f:
    for line in f:
        tokens = line.split(',')
        test.append(tokens)
for i in range(1, len(test)):
    #print(result[int(test[i][1])], result[int(test[i][2])])
    if result[int(test[i][1])] == result[int(test[i][2])]:
        ans.append(1)
    else:
        ans.append(0)
'''
test = []
with open(data_repo + "/check_index.csv", 'r') as f:
    for line in f:
        tokens = line.split(',')
        test.append(tokens)
labels = np.array(km.labels_)

ans = []
for i in range(1, len(test)):
    if labels[int(test[i][1])] == labels[int(test[i][2])]:
        ans.append(1)
    else:
        ans.append(0)

ofile = open(prediction, 'w')
ofile.write("ID,Ans\n")
for i in range(len(ans)):
    ofile.write(str(i) + ',')
    ofile.write(str(ans[i]) + '\n')
