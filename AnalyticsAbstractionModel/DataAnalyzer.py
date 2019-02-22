#DataAnalyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ExploreData():
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
