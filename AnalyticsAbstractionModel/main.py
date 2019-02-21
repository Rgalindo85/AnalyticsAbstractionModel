from DataGenerator import ToyModel
from itertools import combinations
from DataAnalyzer import ExploreData


def run():
    """
    Este codigo intenta generar un toy model, para la representacion de un grafo
    de probabilidades con nodos fijos, asi crear una representacion por usuario.
    """
    #GetDataBase()
    ExploreData()

def GetDataBase():
    """
    Al no tener una db, se crea un Toy model, este escibe un archivo csv con la
    informacion de cada usuario
    """
    n_vertices = 5
    print('Todas las combinaciones: ', list(combinations(range(1, n_vertices+1), 2) ) )
    n_edges = len(list(combinations(range(1, n_vertices+1), 2) ) )
    ToyModel(n_vertices, n_edges)


if __name__ == '__main__':
    run()
