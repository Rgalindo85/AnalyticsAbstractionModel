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
        Step_Labels = ['User', 'Gender', 'Age', 'Social Level', 'Salary', 'N Steps', 'Steps', 'Total Time', 'Time']
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

    range_ages         = [x for x in range(18, 35)]*45 + [x for x in range(35, 60)]*35 + [x for x in range(60, 100)]*20
    range_social_level = [1]*10 +[2]*20 +[3]*30 +[4]*25 +[5]*15 +[6]*10
    range_salary       = [x for x in range(800, 1500)]*40 + [x for x in range(1500, 5500)]*45 + [x for x in range(5500, 20000)]*15
    gender_list        = ['F']*55 + ['M']*45
    user_demogra = [random.choice(range_ages),
                    random.choice(range_social_level),
                    random.choice(range_salary)]

    i = 0
    start_time = datetime.datetime.now()
    for list in gr:
        end_time = start_time + datetime.timedelta(seconds=random.randint(1, 120))
        time = random_date(start_time, end_time)
        user_step = [i, list[0], list[1], time.strftime('%Y-%m-%d %H:%M:%S')]
        User_Steps.append(user_step)
        start_time = time
        i += 1

    data_steps = []
    steps_time = []
    data = []
    for steps in User_Steps:
        move = '%s-%s' % (steps[1], steps[2])
        steps_time.append(steps[3])
        data_steps.append(move)

    first_time = datetime.datetime.strptime(steps_time[0], '%Y-%m-%d %H:%M:%S').timestamp()
    last_time  = datetime.datetime.strptime(steps_time[-1], '%Y-%m-%d %H:%M:%S').timestamp()
    #total_time = (datetime.datetime.timestamp(steps_time[-1]) - datetime.datetime.timestamp(steps_time[0])) + 10
    total_time = last_time - first_time
    total_time_in_min = (total_time + 10)/60.0   # seconds to minutes

    data.append({'User':          user_ID,
                 'Gender':        random.choice(gender_list),
                 'Age':           user_demogra[0],
                 'Social Level' : user_demogra[1],
                 'Salary':        user_demogra[2], # in 800K to 100000K pesos
                 'N Steps':       len(data_steps),
                 'Steps':         data_steps,
                 'Total Time' :   total_time_in_min,
                 'Time':          steps_time,
                 })

    #print(user_ID, data_steps, steps_time)

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
