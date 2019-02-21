import random


class Vertex:
    """
    Define una clase para los vertices, que se agregaran al grafo,
    Las propiedades son, agregar vecino, obtener las conexiones, obtener el ID
    y obenter el peso
    """
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __srt__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]


class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()


def CreateGraph(n_vertex, n_edges):

    gr = Graph()
    for vertex in range(1, n_vertex+1):
        gr.add_vertex(str(vertex))


    rnd_edges = random.randint(1, n_edges)
    vertex_in  = random.randint(1, n_vertex+1)
    # print('Numero de Aristas: ', rnd_edges)
    # print('Vertice Inicial: ', vertex_in)

    Steps = []

    for edge in range(1, rnd_edges+1):
        vertex_out = random.randint(1, n_vertex+1)
        #print(vertex_in, vertex_out)
        # if vertex_in == vertex_out:
        #     random.seed(30)
        #     vertex_out = random.randint(1, n_vertex)

        wgt = random.randint(0, 100)/100.0
        gr.add_edge(str(vertex_in), str(vertex_out), wgt)
        step = [str(vertex_in), str(vertex_out), wgt]
        Steps.append(step)

        vertex_in = vertex_out



    return Steps
