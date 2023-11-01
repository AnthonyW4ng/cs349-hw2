import numpy as np

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    sum_ = 0
    for i in range(len(a)):
        sum_ += (a[i] - b[i]) ** 2
    dist = sum_ ** (1/2)
    return(dist)
        
# returns Cosine Similarity between vectors a dn b
def cosim(a,b):
    sum_ = 0
    a_mag = 0
    b_mag = 0
    for i in range(len(a)):
        sum_ += a[i] * b[i]
        a_mag += a[i] ** 2
        b_mag += b[i] ** 2

    a_mag = a_mag ** (1/2)
    b_mag = b_mag ** (1/2)

    dist = sum_ / (a_mag * b_mag)
    return(dist)

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    K = 5
    nth_obs = 4
    
    train = np.array(train, dtype=object)
    train_labels = train[:, 0]
    train_features = train[:, 1]
    train_features = reduce_dimension(train_features)
    train_features = grayscale(train_features)

    query = np.array(query, dtype=object)
    test_labels = query[:, 0]
    test_features = query[:, 1]
    test_features = reduce_dimension(test_features)
    test_features = grayscale(test_features)

    labels = np.zeros(np.size(test_labels), dtype=int)

    for i in range(len(test_features)):
        test_f = test_features[i]
        neighbours_dist = np.ones(K) * np.inf
        neighbours = np.zeros(K)

        for j in range(i % nth_obs, len(train_features), nth_obs):
            train_f = train_features[j]
            if metric == 'euclidean':
                dist = euclidean(test_f, train_f)
            else:
                dist = 1 - cosim(test_f, train_f)
            for k in range(K):
                if dist < neighbours_dist[k]:
                    neighbours_dist = np.insert(neighbours_dist, k, dist)
                    neighbours_dist = np.delete(neighbours_dist, -1)

                    neighbours = np.insert(neighbours, k, j)
                    neighbours = np.delete(neighbours, -1)
                    break


        nearest_labels = [train_labels[int(n)] for n in neighbours]
        unique, counts = np.unique(nearest_labels, return_counts=True)
        idx = np.argmax(counts)
        labels[i] = unique[idx]

    confusion_matrix = np.zeros((10, 10))
    for i in range(len(test_labels)):
        predicted = labels[i] 
        real = test_labels[i]
        confusion_matrix[predicted][real] += 1

    print(confusion_matrix)
    return(labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    n_clusters = 50

    train = np.array(train, dtype=object)
    train_labels = train[:, 0]
    train_features = train[:, 1]
    train_features = reduce_dimension(train_features)
    train_features = grayscale(train_features)

    
    query = np.array(query, dtype=object)
    test_labels = query[:, 0]
    test_features = query[:, 1]
    test_features = reduce_dimension(test_features)
    test_features = grayscale(test_features)

    # randomize the centroids
    centroids = []
    for i in range(n_clusters):
        centroids.append(np.random.randint(2, size=len(train_features[0])))

    converged = False
    # while not converged
    while not converged:
        # intializes each cluster
        clusters = []
        clusters_idx = []
        for i in range(n_clusters):
            clusters.append([])
            clusters_idx.append([])

        # for each obs find the closest centroid
        for i in range(len(train_features)):
            obs = train_features[i]
            count = 0
            cluster_num = 0
            best_dist = float('inf')
            for j in range(len(centroids)):
                c = centroids[j]
                if metric == 'euclidean':
                    dist = euclidean(obs, c)
                else:
                    dist = 1 - cosim(obs, c)
                if dist < best_dist:
                    cluster_num = count
                    best_dist = dist
                count += 1
            clusters[cluster_num].append(obs)
            clusters_idx[cluster_num].append(i)
            
        # using those clusters calculate new centroid
        old_centroids = centroids
        centroids = []
        for i in range(n_clusters):
            if len(clusters[i]) == 0:
                n_clusters -= 1
            else:
                centroids.append(np.round(np.sum(clusters[i], 0)/len(clusters[i])))

        # check if converged
        for i in range(n_clusters):
            converged = True
            if not all(np.equal(old_centroids[i], centroids[i])):
                converged = False

    # assign a label to those groups
    cluster_labels = []
    for i in range(n_clusters):
        c_labels = []
        for idx in clusters_idx[i]:
            c_labels.append(train_labels[idx])
        unique, counts = np.unique(c_labels, return_counts=True)
        index = np.argmax(counts)
        cluster_labels.append(unique[index])
    print('labels:\n', cluster_labels)

    labels = np.zeros(np.size(test_labels), dtype=int)
    for i in range(len(test_features)):
        test_obs = test_features[i]
        count = 0
        cluster_num = 0
        best_dist = float('inf')
        for c in centroids:
            if metric == 'euclidean':
                dist = euclidean(test_obs, c)
            else:
                dist = 1 - cosim(test_obs, c)
            if dist < best_dist:
                cluster_num = count
                best_dist = dist
            count += 1
        labels[i] = cluster_labels[cluster_num]

    confusion_matrix = np.zeros((10, 10))
    count = 0
    for i in range(len(test_labels)):
        predicted = labels[i] 
        real = test_labels[i]
        confusion_matrix[predicted][real] += 1
        if predicted == real:
            count += 1
    print(count/len(test_labels))

    print(confusion_matrix)
    return(labels)

def reduce_dimension(features):
    for i in range(len(features)):
        features[i] = features[i][0::2]
    return features


def grayscale(features):
    for i in range(len(features)):
        for j in range(len(features[i])):
            if features[i][j] > 0:
                features[i][j] = 1
    return features


def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = int(tokens[0])
            attribs = []
            for i in range(784):
                attribs.append(int(tokens[i+1]))
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    # show('valid.csv','pixels')

    test_data = read_data('test.csv')
    train_data = read_data('train.csv')
    # print('KNN: Euclidean')
    # labels = knn(train_data, test_data, 'euclidean')
    # print('\nKNN: Cosim')
    # labels = knn(train_data, test_data, 'cosim')

    print('KMeans: Euclidean')
    labels = kmeans(train_data, test_data, 'euclidean')
    print('\nKMeans: Cosim')
    labels = kmeans(train_data, test_data, 'cosim')
    
if __name__ == "__main__":
    main()
    