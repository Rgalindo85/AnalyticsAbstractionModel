import GraphGenerator as grafo_maker
import random
import datetime
import csv

def ToyModel(n_vertices, n_edges):
    """
    Crea un grafo aleatorio, con pesos aleatorios
    Regresa un dictionario con la representacion
    """

    with open('test_data.csv', 'w') as csvFile:
        Step_Labels = ['User', 'Age', 'Social Level', 'Salary', 'Step', 'In', 'Out', 'Time']
        fields = Step_Labels
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        writer.writeheader()
        for user in range(1, 1000):
            gr1 = grafo_maker.CreateGraph(n_vertices, n_edges)
            user_ID = 'U_%d' % random.randint(1, 1000)
            ObtainPath(gr1, user_ID, csvFile, writer)

    csvFile.close()
def ObtainPath(gr, user_ID, csvFile, writer):
    """
    Aqui al camino seleccionado se le adicionara un timestamp aleatorio a cada
    paso, y escribe los datos en un archivo CSV
    """
    User_Steps = []
    user_demogra = [random.randint(18, 100),
                    random.randint(1, 6),
                    random.randint(800, 100000)]

    i = 0
    start_time = datetime.datetime.now()
    for list in gr:
        end_time = start_time + datetime.timedelta(seconds=random.randint(1, 120))
        time = random_date(start_time, end_time)
        user_step = [i, list[0], list[1], time.strftime('%Y-%m-%d %H:%M:%S')]
        User_Steps.append(user_step)
        start_time = time
        i += 1

    data = []
    for steps in User_Steps:
        data.append({'User': user_ID,
                     'Age':  user_demogra[0],
                     'Social Level' : user_demogra[1],
                     'Salary': user_demogra[2], # in 800K to 100000K pesos
                     'Step': steps[0],
                     'In':   steps[1],
                     'Out':  steps[2],
                     'Time': steps[3]
                     })

    #print('%s =', user_ID, data)

    #with open('test.csv', 'w') as csvFile:

    writer.writerows(data)

    #csvFile.close()

def random_date(start, end):
    """
    Esta funcion regresa el tiempo entre un intervalo definido
    """
    delta = end - start
    int_delta = (delta.days*24*60*60) + delta.seconds
    random_second = random.randrange(int_delta)

    return start + datetime.timedelta(seconds=random_second)
