import random
import numpy as np
class KMeans:
    def __init__(self,n_cluster=2,max_iter=100):
        self.n_cluster=n_cluster
        self.max_iter=max_iter
        self.centroids=None

    def fit_predict(self,X):
        random_index=random.sample(range(0,X.shape[0]),self.n_cluster)
        self.centroids=X[random_index]
        

        for i in range(self.max_iter):
            # assign clusters
           cluster_group=self.assign_clusters(X)
           old_centroid=self.centroids
              #move centroids
           self.centroids=self.move_centroid(X,cluster_group)
            #check finish
           if(old_centroid==self.centroids).all():
               break
        return cluster_group   
    
    def assign_clusters(self,X):
        cluster_group=[]
        ditances=[]
        for row in X:
            for centroid in self.centroids:
                ditances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distance=min(ditances)
            index_pos=ditances.index(min_distance)
            cluster_group.append(index_pos)
            # print(index_pos)
            ditances.clear()    
        return np.array(cluster_group)

    def move_centroid(self,X,cluster_group):
        new_centroid=[]

        cluster_type=np.unique(cluster_group) 
        for type in cluster_type:
            
            new_centroid.append(X[cluster_group==type].mean(axis=0))
        
        return np.array(new_centroid)    