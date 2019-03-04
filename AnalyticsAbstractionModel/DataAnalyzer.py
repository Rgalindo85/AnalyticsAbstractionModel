#DataAnalyzer.py
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import ast
from sklearn.cluster import KMeans


def AnalyzeGraphs(filename):
    """
    Try to generate a graph adjancy matrix, using the edges in each path. Then
    use the matrix to obtain its eigenvectors and then calculate the sum vector
    of the eigenvectors, as a codifier for the graph, Fill an array of the
    resultant vector
    """

    pd_database = pd.read_csv(filename)
    nodes_list = ['1','2','3','4','5','6','7','8','9','10']

    eigenvectors = []

    for path in pd_database['Steps']:
        path = ast.literal_eval(path)

        eig_vec = GetVectorID(path, nodes_list)
        eig_vec_flat = np.array(eig_vec).flatten()
        #if len(eig_vec) < 10:
        #print(eig_vec_flat)

        eigenvectors.append(eig_vec_flat)

    print('*** Clusteriing %d graphs by K-Means ***' % len(eigenvectors))

    np_eigenvectors = np.array(eigenvectors)
    #print(np_eigenvectors.shape())
    #print('Size:', np_eigenvectors.size())
    kmeans_2 = KMeans(n_clusters = 2, random_state = 0).fit(np_eigenvectors)
    kmeans_3 = KMeans(n_clusters = 3, random_state = 0).fit(np_eigenvectors)
    kmeans_4 = KMeans(n_clusters = 4, random_state = 0).fit(np_eigenvectors)
    kmeans_5 = KMeans(n_clusters = 5, random_state = 0).fit(np_eigenvectors)

    data_new_2 = kmeans_2.transform(np_eigenvectors)
    data_new_3 = kmeans_3.transform(np_eigenvectors)
    data_new_4 = kmeans_4.transform(np_eigenvectors)
    data_new_5 = kmeans_5.transform(np_eigenvectors)

    with open('k-means2_out.txt', 'w') as outfile2:
        np.savetxt(outfile2, data_new_2, fmt='%4.1f')
    with open('k-means3_out.txt', 'w') as outfile3:
        np.savetxt(outfile3, data_new_3, fmt='%4.1f')
    with open('k-means4_out.txt', 'w') as outfile4:
        np.savetxt(outfile4, data_new_4, fmt='%4.1f')
    with open('k-means5_out.txt', 'w') as outfile5:
        np.savetxt(outfile5, data_new_5, fmt='%4.1f')

    outfile2.close()
    outfile3.close()
    outfile4.close()
    outfile5.close()


    PlotDistanceAngle(np_eigenvectors, kmeans_2.cluster_centers_, data_new_2)

    #print('k-Means Labels:', kmeans.labels_)
    #print('k-Means Cluster Centers:', kmeans.cluster_centers_)

def PlotDistanceAngle(eig_vectors, centroids, distances):
    mag_c1 = np.linalg.norm(centroids[0])
    mag_c2 = np.linalg.norm(centroids[1])
    # print(eig_vectors.shape)
    # print(centroids.shape)
    # print(distances.shape)

    cos_theta_array_c1 = []
    cos_theta_array_c2 = []

    for vec in eig_vectors:
        mag_vec = np.linalg.norm(vec)
        cos_theta_c1 = np.dot(vec, centroids[0])/(mag_c1*mag_vec)
        cos_theta_c2 = np.dot(vec, centroids[1])/(mag_c2*mag_vec)

        cos_theta_array_c1.append(cos_theta_c1)
        cos_theta_array_c2.append(cos_theta_c2)

    print(len(distances.T[0]))
    print(len(cos_theta_array_c1))

    fig, ax = plt.subplots(ncols = 2, nrows = 1, sharex=True, sharey=True)
    ax[0].scatter(cos_theta_array_c1[:100], distances.T[0][:100], color='r')
    ax[0].scatter(cos_theta_array_c2[:100], distances.T[1][:100], color='b')

    ax[1].scatter(cos_theta_array_c1[:100], distances.T[1][:100])
    ax[1].scatter(cos_theta_array_c2[:100], distances.T[0][:100])
    plt.show()


def AnalyzeDistancesToClusters(input_file):


    pd_distances = pd.read_csv(input_file, sep='\s+', header=None, usecols=[0,1], names = ['c1', 'c2'])
    #pd_distances = pd.read_csv(input_file, sep='\s+', header=None, usecols=[0,1,2], names = ['c1', 'c2', 'c3'])
    #pd_distances = pd.read_csv(input_file, sep='\s+', header=None, usecols=[0,1,2,3], names = ['c1', 'c2', 'c3', 'c4'])
    print(pd_distances.head())
    print(pd_distances.tail())

    plt.figure(figsize = (5,5))
    plt.hist2d(pd_distances['c1'], pd_distances['c2'], bins=50, cmap='RdBu_r', norm=LogNorm())
    cb = plt.colorbar()
    plt.xlabel('Distance to Cluster 1')
    plt.ylabel('Distance to Cluster 2')
    cb.set_label('Number of Graphs')
    plt.xlim((0,6))
    plt.ylim((0,6))
    plt.title('Distance to Clusters')

    plt.show()
    # 
    # plt.scatter(pd_distances['c1'], pd_distances['c2'])
    # 
    
    # plt.savefig('Plots/2-means-scatter.png')


    # fig, axes = plt.subplots(2,2, figsize = (10,10), sharex=True, sharey=True)
    # axes[0,0].hist(pd_distances['c1'], bins=30, range=(0, 6))
    # axes[0,1].hist(pd_distances['c2'], bins=30, range=(0, 6))
    # axes[1,0].hist(pd_distances['c3'], bins=30, range=(0, 6))
    # axes[1,1].hist(pd_distances['c4'], bins=30, range=(0, 6))

    # axes[0,0].set_title('Distance To Centroid of Cluster 1')
    # axes[0,1].set_title('Distance To Centroid of Cluster 2')
    # axes[1,0].set_title('Distance To Centroid of Cluster 3')
    # axes[1,1].set_title('Distance To Centroid of Cluster 4')

    # axes[0,0].set_xlabel('Distance')
    # axes[0,1].set_xlabel('Distance')
    # axes[1,0].set_xlabel('Distance')
    # axes[1,1].set_xlabel('Distance')

    # axes[0,0].set_ylabel('Number of Graphs')
    # axes[0,1].set_ylabel('Number of Graphs')
    # axes[1,0].set_ylabel('Number of Graphs')
    # axes[1,1].set_ylabel('Number of Graphs')

    # #fig.tight_layout()
    # plt.savefig('Plots/4-means-hist.png')

    # plt.figure(figsize = (5,5))
    # plt.scatter(pd_distances['c1'], pd_distances['c2'])
    # plt.title('Distance to Clusters')
    # plt.xlabel('Distance to Cluster 1')
    # plt.ylabel('Distance to Cluster 2')
    # plt.xlim((0,6))
    # plt.ylim((0,6))
    # plt.savefig('Plots/2-means-scatter.png')

def GetVectorID(path, nodes_list):
    """
    Try to generate a graph adjancy matrix, using the edges in the path. Then
    use the matrix to obtain its eigenvectors and then calculate the sum vector
    of the eigenvectors, as a codifier for the graph, returns the vector
    """

    gr = nx.MultiGraph()
    gr.add_nodes_from(nodes_list)

    for step in path:
        gr.add_edge(step[0], step[1])

    adj_matrix = nx.to_numpy_matrix(gr, nodelist=nodes_list)
    eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
    total_vector = eigenvectors.sum(axis=0)

    #print(total_vector.shape)

    #nx.draw_networkx(gr, with_labels=True)
    # plt.savefig('{}{}'.format(name, '.png'))
    #
    # plt.clf()
    # plt.cla()
    # plt.close()

    return total_vector


def ExploreDemographicData():
    """
    Aqui se hace un analisis simple de las variables en la base de datos,
    algunos graficos unidimensionales, 2D y 3D...
    """
    csvFile_name = 'test_data.csv'
    pd_database = pd.read_csv(csvFile_name, delimiter=',', header=0)

    print(pd_database.head())

    plt.style.use('ggplot')
    #plt.figure()
    # pd_database.plot.hist(color='b', alpha=0.5, bins=50,
    #                         columns=['Age', 'Social Level', 'Salary'])
    #pd_database['Total Time'].plot.hist(color='b', alpha=0.5, bins=30)

    # fig, axes = plt.subplots(nrows=2, ncols=3)
    # pd_database['Salary'].plot.hist(ax=axes[0,0]);       axes[0,0].set_title('Salary')
    # pd_database['Social Level'].plot.hist(ax=axes[0,1]); axes[0,1].set_title('Social Level')
    # pd_database['Age'].plot.hist(ax=axes[0,2]);          axes[0,2].set_title('Age')
    # pd_database['N Steps'].plot.hist(ax=axes[1,0]);      axes[1,0].set_title('N Steps')
    # pd_database['Total Time'].plot.hist(ax=axes[1,1]);   axes[1,1].set_title('Total Time')
    # plt.show()


    # Scatter Mattrix
    scat_db = pd.DataFrame(pd_database, columns=['Salary', 'Gender', 'Age', 'Social Level', 'N Steps', 'Total Time'] )
    # pd.scatter_matrix(scat_db, c=pd_database['Social Level'], alpha = 0.5, figsize = [10, 10])
    # plt.show()

    # g = sns.PairGrid(scat_db, hue='Social Level')
    # g.map_diag(plt.hist)
    # g.map_offdiag(plt.scatter)
    # g.add_legend();

    g = sns.pairplot(scat_db, hue='Gender', palette='Set3')#, diag_kind= 'kde')
    plt.show()


def test():
    path1 = [('1', '7', {'time': '2019-02-27 13:46:59'}),
             ('3', '6', {'time': '2019-02-27 13:46:57'}),
             ('3', '2', {'time': '2019-02-27 13:48:42'}),
             ('6', '1', {'time': '2019-02-27 13:46:57'}),
             ('7', '3', {'time': '2019-02-27 13:47:07'})]

    path2 = [('1', '9', {'time': '2019-02-27 13:51:08'}),
             ('1', '8', {'time': '2019-02-27 13:55:27'}),
             ('2', '3', {'time': '2019-02-27 13:53:48'}),
             ('3', '1', {'time': '2019-02-27 13:54:46'}),
             ('3', '7', {'time': '2019-02-27 13:56:31'}),
             ('4', '7', {'time': '2019-02-27 13:49:18'}),
             ('5', '4', {'time': '2019-02-27 13:48:47'}),
             ('7', '10', {'time': '2019-02-27 13:48:17'}),
             ('7', '10', {'time': '2019-02-27 13:52:33'}),
             ('7', '1', {'time': '2019-02-27 13:50:25'}),
             ('7', '9', {'time': '2019-02-27 13:57:15'}),
             ('7', '2', {'time': '2019-02-27 13:57:40'}),
             ('8', '7', {'time': '2019-02-27 13:52:06'}),
             ('8', '3', {'time': '2019-02-27 13:56:25'}),
             ('9', '8', {'time': '2019-02-27 13:51:37'}),
             ('9', '7', {'time': '2019-02-27 13:57:20'}),
             ('10', '5', {'time': '2019-02-27 13:48:20'}),
             ('10', '2', {'time': '2019-02-27 13:52:50'})
            ]

    path3 = [('2', '3', {'time': '2019-02-27 13:52:34'}),
             ('2', '10', {'time': '2019-02-27 13:54:48'}),
             ('3', '4', {'time': '2019-02-27 13:49:16'}),
             ('3', '5', {'time': '2019-02-27 13:52:40'}),
             ('4', '9', {'time': '2019-02-27 13:47:38'}),
             ('4', '9', {'time': '2019-02-27 13:52:58'}),
             ('4', '7', {'time': '2019-02-27 13:49:42'}),
             ('4', '1', {'time': '2019-02-27 13:54:52'}),
             ('5', '4', {'time': '2019-02-27 13:52:51'}),
             ('5', '2', {'time': '2019-02-27 13:54:18'}),
             ('6', '3', {'time': '2019-02-27 13:47:54'}),
             ('7', '2', {'time': '2019-02-27 13:51:21'}),
             ('8', '4', {'time': '2019-02-27 13:46:44'}),
             ('9', '10', {'time': '2019-02-27 13:47:41'}),
             ('9', '5', {'time': '2019-02-27 13:54:08'}),
             ('10', '6', {'time': '2019-02-27 13:47:47'}),
             ('10', '4', {'time': '2019-02-27 13:54:52'})
            ]
