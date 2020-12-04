import time
import math
from queue import PriorityQueue
from threading import Thread, Lock

class DesignatedRouter:
    def __init__(self):
        self.senders = []
        self.receivers = []
        self.disabled_routers = []
        self.adjacency_list = {}
        self.lock = Lock()

    def receive_topology(self, receiver, node):
        while True:
            node_adjacency_list = set(receiver.receive())
            self.lock.acquire()
            if node not in self.adjacency_list:
                self.adjacency_list[node] = set()
            self.adjacency_list[node].update(node_adjacency_list)
            self.lock.release()

    def send_topology(self, node):
        self.senders[node].send([self.adjacency_list])

    def run(self):
        receive = []
        for id, receiver in enumerate(self.receivers):
            receive.append(Thread(target=self.receive_topology, args=(receiver, id)))
        broadcast = []
        for id, receiver in enumerate(self.receivers):
            broadcast.append(Thread(target=self.send_topology, args=(id,)))
        for process in receive:
            process.start()
        time.sleep(5)
        for process in broadcast:
            process.start()
        time.sleep(5)

class Router:
    def __init__(self, id):
        self.id = id
        self.des_receiver = None
        self.des_sender = None
        self.senders = {}
        self.receivers = {}
        self.active_neighbours = []
        self.paths = {}
        self.lock = Lock()

    def send_hello(self, sender):
        #if self.disabled == True:
            #return
        sender.send(["hello"])

    def receive_hello(self, receiver, node):
        #if self.disabled == True:
            #return
        while True:
            hello = receiver.receive()
            if hello is not None:
                # print("node #", self.id, ":", "Discovered node ", node)
                self.active_neighbours.append(node)
                break

    def send_topology(self):
        self.des_sender.send(self.active_neighbours)

    def receive_topology(self):
        while True:
            topology_update = self.des_receiver.receive()
            if topology_update is not None:
                self.lock.acquire()
                adjacency_list = topology_update[0]
                self.update_paths(adjacency_list, self.id)
                self.lock.release()

    def update_paths(self, adjacency_list, start):
        distances = {}
        is_visited = {}
        for i in adjacency_list.keys():
            is_visited[i] = False
            distances[i] = math.inf
        distances[start] = 0
        self.paths = {start: []}
        q = PriorityQueue()
        q.put((0, start))

        while not q.empty():
            cur_node = q.get()[1]
            is_visited[cur_node] = True
            for neighbor in adjacency_list[cur_node]:
                if not is_visited[neighbor]:
                    if distances[neighbor] > distances[cur_node] + 1:
                        distances[neighbor] = distances[cur_node] + 1
                        self.paths[neighbor] = self.paths[cur_node] + [neighbor]
                    q.put((distances[neighbor], neighbor))
        return

    def run(self):
        hello_processes = []
        for _, sender in self.senders.items():
            hello_processes.append(Thread(target=self.send_hello, args=(sender,)))
        for id, receiver in self.receivers.items():
            hello_processes.append(Thread(target=self.receive_hello, args=(receiver, id)))
        for process in hello_processes:
            process.start()
        for process in hello_processes:
            process.join()

        self.send_topology()
        self.receive_topology()