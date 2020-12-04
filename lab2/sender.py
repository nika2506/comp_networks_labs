from other_classes import Package, Message
from functions import rand_is_corrupted


class Sender:
    def __init__(self, transmitter, protocol, window_size, corruption_probability=0.1, show_comments=False):
        self.transmitter = transmitter
        self.protocol = protocol
        self.window_size = window_size
        self.corruption_probability = corruption_probability
        self.show_comments = show_comments
        self.total_sent = 0
        self.total_received_ack = 0
        if show_comments:
            print("Sender created")

    def send(self, data):
        num_packages = len(data)
        self.transmitter.send_receiver(Package(0, num_packages, False))
        while True:
            ack = self.transmitter.get_sender()
            if ack is not None:
                break
        window = []
        while True:
            if len(window) < self.window_size and self.total_sent < num_packages:
                frame = Package(self.total_sent, data[self.total_sent], rand_is_corrupted(self.corruption_probability))
                self.transmitter.send_receiver(frame)
                window.append(frame)
                self.total_sent += 1
                if self.show_comments:
                    print("Sender: Package № %d sent to receiver, corruption = %r" %
                          (frame.sequence, frame.is_corrupted))

            ack = self.transmitter.get_sender()
            if ack is not None:
                if ack.message == Message.ACK:
                    self.total_received_ack += 1
                    del window[0]
                    if self.show_comments:
                        print("Sender: Package № %d received" % ack.sequence)
                    if self.total_received_ack == num_packages:
                        if self.show_comments:
                            print("Sender: finished")
                        return
                elif ack.message == Message.NAK:
                    if self.protocol == 0:
                        self.go_back_n(window, self.corruption_probability)
                    elif self.protocol == 1:
                        self.selective_repeat(window, ack, self.corruption_probability)

    def go_back_n(self, window, corruption_probability):
        for entry in window:
            entry.is_corrupted = rand_is_corrupted(corruption_probability)
            self.transmitter.send_receiver(entry)
            if self.show_comments:
                print("Sender: Package № %d sent to receiver, corruption = %r" %
                      (entry.sequence, entry.is_corrupted))

    def selective_repeat(self, window, frame, corruption_probability):
        result = window[frame.sequence - self.total_received_ack]
        result.is_corrupted = rand_is_corrupted(corruption_probability)
        self.transmitter.send_receiver(result)
        if self.show_comments:
            print("Sender: Package № %d sent to receiver, corruption = %r" %
                  (result.sequence, result.is_corrupted))
