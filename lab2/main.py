import time
from threading import Thread
from functions import print_path, ring_topology, \
    star_topology, bus_topology
from other_classes import Transmitter
from sender import Sender
from receiver import Receiver
from routers import DesignatedRouter, Router


def make_network(nodes, edges, num_nodes, protocol, wind, show_comm):
    routers = []
    #print(nodes)
    #print(edges)
    des_router = DesignatedRouter()
    for id in num_nodes:
        router = Router(id)
        transmitter = Transmitter()
        router.des_sender = Sender(transmitter, protocol, wind, show_comments=show_comm)
        des_router.receivers.append(Receiver(transmitter, protocol, wind, show_comments=show_comm))
        transmitter = Transmitter()
        router.des_receiver = Receiver(transmitter, protocol, wind, show_comments=show_comm)
        des_router.senders.append(Sender(transmitter, protocol, wind, show_comments=show_comm))
        routers.append(router)
    for edge in edges:
        transmitter = Transmitter()
        ind1 = edge[0]
        ind2 = edge[1]
        routers[ind1].senders[edge[1]] = Sender(transmitter, protocol, wind, show_comments=show_comm)
        routers[ind2].receivers[edge[0]] = Receiver(transmitter, protocol, wind, show_comments=show_comm)
    return des_router, routers

if __name__ == "__main__":
    protocol = 0  # GO BACK N
    wind = 10 #size of window
    show_comm = False
    p = 0.3 #probability of router failure
    num = 5 #number of nodes

    nodes, edges, num_nodes = star_topology(num, p)
    des_router, routers = make_network(nodes, edges, num_nodes, protocol, wind, show_comm)
    processes = [Thread(target=des_router.run)]
    for router in routers:
        processes.append(Thread(target=router.run))
    for process in processes:
        process.start()
    time.sleep(10)
    for node in routers:
        print("Paths from node №", nodes[node.id], ":")
        for id, path in node.paths.items():
            if id != node.id:
                print_path(path, node.id, nodes)
    for process in processes:
        process.join()
