from itertools import combinations
from DataAnalyzer import AnalyzeGraphs, AnalyzeDistancesToClusters
import UserDataGenerator as data_gen
import datetime


def run():
    """
    Este codigo intenta generar un toy model, para la representacion de un grafo
    de probabilidades con nodos fijos, asi crear una representacion por usuario.
    """
    #GetDataBase()
    #ExploreData()
    #graphDB_FileName = '2019-03-01_app_data.csv'
    #AnalyzeGraphs(filename = graphDB_FileName)
    AnalyzeDistancesToClusters(input_file='k-means2_out.txt')
    #AnalyzeDistancesToClusters(input_file='k-means3_out.txt')
    #AnalyzeDistancesToClusters(input_file='k-means4_out.txt')
    

def GetDataBase():
    """
    Al no tener una db, se crea un Toy model, este escibe un archivo csv con la
    informacion de cada usuario
    """
    n_vertices = 10
    n_edges = len(list(combinations(range(1, n_vertices+1), 2) ) )
    day = str(datetime.datetime.today().strftime('%Y-%m-%d'))
    print(day)


    #ToyModel(n_vertices, n_edges)
    data_gen.FillUserInteractions(n_vertices, n_edges, day)


if __name__ == '__main__':
    run()
