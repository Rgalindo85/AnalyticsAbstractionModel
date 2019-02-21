#DataAnalyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ExploreData():
    """
    Aqui se hace un analisis simple de las variables en la base de datos,
    algunos graficos unidimensionales, 2D y 3D...
    """
    csvFile_name = 'test_data.csv'
    pd_database = pd.read_csv(csvFile_name, delimiter=',', header=0)

    print(pd_database.head())

    plt.figure()
    # pd_database.plot.hist(color='b', alpha=0.5, bins=50,
    #                         columns=['Age', 'Social Level', 'Salary'])
    pd_database['Salary'].plot.hist(color='b', alpha=0.5, bins=30)                        
    plt.show()
