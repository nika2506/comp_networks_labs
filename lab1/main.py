import time
from multiprocessing import Process, Queue
from functions import make_plot_packages, make_plot_wind
from classes import Receiver, Sender, Transmitter

if __name__ == "__main__":
    sender_buf = Queue()
    receiver_buf = Queue()
    transmitter = Transmitter(sender_buf, receiver_buf)

    wind = 10 #size of window
    wind_array = [10 * i for i in range(1, 7)] #array of different size of window
    p = 0.1 #probability of loss or corrupt package
    num_packages_array = [500 * i for i in range(1, 7)] #array of different number of packages
    num_packages = 3000 #number of packages
    time_gbn = [] #performance of Go back N
    time_sr = [] #performance of Selective repeat

    for num in num_packages_array:
        print(num)
        sender = Sender(transmitter, 0, wind, False)
        receiver = Receiver(transmitter, 0, wind, False)
        sender_proc = Process(target=sender.send, args=(num, p))
        receiver_proc = Process(target=receiver.receive, args=(num,))
        start = time.time()
        sender_proc.start()
        receiver_proc.start()
        receiver_proc.join()
        sender_proc.join()
        end = time.time()
        time_gbn.append(end - start)

        sender = Sender(transmitter, 1, wind, False)
        receiver = Receiver(transmitter, 1, wind, False)
        sender_proc = Process(target=sender.send, args=(num, p))
        receiver_proc = Process(target=receiver.receive, args=(num,))
        start = time.time()
        sender_proc.start()
        receiver_proc.start()
        receiver_proc.join()
        sender_proc.join()
        end = time.time()
        time_sr.append(end - start)

    make_plot_packages(num_packages_array, time_gbn, time_sr)

    time_gbn = []  # performance of Go back N
    time_sr = []  # performance of Selective repeat

    for w in wind_array:
        print(w)
        sender = Sender(transmitter, 0, w, False)
        receiver = Receiver(transmitter, 0, w, False)
        sender_proc = Process(target=sender.send, args=(num_packages, p))
        receiver_proc = Process(target=receiver.receive, args=(num_packages,))
        start = time.time()
        sender_proc.start()
        receiver_proc.start()
        receiver_proc.join()
        sender_proc.join()
        end = time.time()
        time_gbn.append(end - start)

        sender = Sender(transmitter, 1, w, False)
        receiver = Receiver(transmitter, 1, w, False)
        sender_proc = Process(target=sender.send, args=(num_packages, p))
        receiver_proc = Process(target=receiver.receive, args=(num_packages,))
        start = time.time()
        sender_proc.start()
        receiver_proc.start()
        receiver_proc.join()
        sender_proc.join()
        end = time.time()
        time_sr.append(end - start)

    make_plot_wind(wind_array, time_gbn, time_sr)