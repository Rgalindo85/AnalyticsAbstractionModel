import random
import datetime
import networkx as nx


def CreatePath(n_vertex, n_edges):
    """
    According with a defined number of nodes and edges, it creates a random path,
    storing the edges (in, out, {time:the time})
    """

    rnd_edges = random.randint(1, n_edges)
    vertex_in = random.randint(1, n_vertex)

    Steps = []
    start_time = datetime.datetime.now()

    for edge in range(1, rnd_edges+1):
        vertex_out = random.randint(1, n_vertex)
        if vertex_out == vertex_in:
            vertex_out = random.randint(1, n_vertex)
            
        end_time = start_time + datetime.timedelta(seconds=random.randint(1, 120))
        time = random_date(start_time, end_time)

        #wgt = random.randint(0, 100)/100.0
        #gr.add_edge(str(vertex_in), str(vertex_out), wgt)
        step = (str(vertex_in), str(vertex_out), {'time': time.strftime('%Y-%m-%d %H:%M:%S')})
        Steps.append(step)
        start_time = time

        vertex_in = vertex_out

    return Steps


def CreateRandomGraph(n_vertex, n_edges):
    """
    Using graphs from Networkx, with a defined path for fixed nodes,
    it return a graph with the info of the edges and the time when it happens
    to allow to build the path
    """

    G = nx.MultiDiGraph()
    path = CreatePath(n_vertex, n_edges)
    nodes_list = [str(i) for i in range(1, n_vertex+1)]

    G.add_nodes_from(nodes_list)
    G.add_edges_from(path)
    #G.nodes(data=True)

    #nx.draw_networkx(G, with_labels=True)
    return G


def random_date(start, end):
    """
    Esta funcion regresa el tiempo entre un intervalo definido
    """
    delta = end - start
    int_delta = (delta.days*24*60*60) + delta.seconds
    random_second = random.randrange(int_delta)

    return start + datetime.timedelta(seconds=random_second)
