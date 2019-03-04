import csv, random, datetime
import pandas as pd
import GraphGenerator as gr_maker
import matplotlib.pyplot as plt
import networkx as nx


def FillUserInteractions(n_vertices, n_edges, day):
    """
    Create a csv file with the information of the interactions of the user with the platform,
    it selects the user ID from the demographic_data.csv file, fill the info of the data
    """
    pd_usersID = pd.read_csv('demographic_data.csv', usecols=['User'])

    #print(pd_usersID.head())
    with open('{}_app_data.csv'.format(day), 'w') as csvFile:
        print('***** Creating the data base for user interactions in skynetapp ******')
        interaction_Labels = ['User', 'N Steps', 'Total Time', 'Steps']
        writer = csv.DictWriter(csvFile, fieldnames=interaction_Labels)
        writer.writeheader()

        index = 0
        total_users = len(pd_usersID['User'])
        for user in pd_usersID['User']:
            index +=1
            #index = pd.Index(pd_usersID['User']).get_loc(user)
            #print(user, index)
            if index % 1000 == 0.0: print('{}/{}'.format(index, total_users) )
            if index % 2 == 0:
                continue

            n_visits = random.randint(1, 10)

            for visit in range(1, n_visits):
                gr = nx.MultiDiGraph(gr_maker.CreateRandomGraph(n_vertices, n_edges))
                interactions = gr.edges(data=True)
                total_time = GetTotalTime(interactions)

                data = [{'User':       user,
                         'N Steps':    len(interactions),
                         'Total Time': total_time,
                         'Steps':      interactions
                        }]
                writer.writerows(data)

    csvFile.close()


def GetTotalTime(steps):
    """
    From the times that are recorded by each edge, find the total time,
    converting them to unix timestamps and finding the min and max value to set the range
    """

    times = []
    for step in steps:
        times.append(datetime.datetime.strptime(step[2]['time'], '%Y-%m-%d %H:%M:%S').timestamp())

    sorted_times = sorted(times)
    total_time = sorted_times[-1] - sorted_times[0]

    return total_time/60.0       #return time in minutes



def FillUserInfo():
    """
    Generate random info about users to fill a csv file that contains some demographic info
    the filename is demographic_data.csv
    """
    with open('demographic_data.csv', 'w') as csvFile:
        demo_Labels = ['User', 'Gender', 'Age', 'Social Level', 'Salary', 'City']
        writer = csv.DictWriter(csvFile, fieldnames=demo_Labels)
        writer.writeheader()

        for user in range(0, int(1e6) ):
            user_ID = 'U_{}{}{}{}{}{}'.format(random.randint(0,9), random.randint(0,9), random.randint(0,9), random.randint(0,9), random.randint(0,9), random.randint(0,9) )

            range_ages         = [x for x in range(18, 35)]*45 + [x for x in range(35, 60)]*35 + [x for x in range(60, 100)]*20
            range_social_level = [1]*10 +[2]*20 +[3]*30 +[4]*25 +[5]*15 +[6]*10
            range_salary       = [x for x in range(800, 1500)]*40 + [x for x in range(1500, 5500)]*45 + [x for x in range(5500, 20000)]*15
            gender_list        = ['F']*55 + ['M']*45
            city_list          = ['BOG', 'MED', 'CAL', 'BAR', 'CAR', 'NEI', 'IBA']*50 + ['PER', 'ARM', 'PAS', 'TUN', 'VIL', 'ARA', 'BUC', 'CUC']*30

            user_demogra = [random.choice(gender_list),
                            random.choice(range_ages),
                            random.choice(range_social_level),
                            random.choice(range_salary),
                            random.choice(city_list)]

            data = [{'User':          user_ID,
                     'Gender':        user_demogra[0],
                     'Age':           user_demogra[1],
                     'Social Level' : user_demogra[2],
                     'Salary':        user_demogra[3], # in 800K to 100000K pesos
                     'City':          user_demogra[4],
                    }]

            writer.writerows(data)

    csvFile.close()
