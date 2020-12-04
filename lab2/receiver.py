from other_classes import Package, Message


class Receiver:
    def __init__(self, transmitter, protocol, window_size=None, show_comments=False):
        self.transmitter = transmitter
        self.protocol = protocol
        self.window_size = window_size
        self.show_comments = show_comments
        self.total_received = 0
        self.total_sent_ack = 0
        if self.show_comments:
            print("Receiver created")

    def receive(self):
        while True:
            num_packages = self.transmitter.get_receiver()
            if num_packages is not None:
                break
        num_packages = num_packages.message
        self.transmitter.send_sender(Package(0, Message.ACK, False))
        if self.protocol == 0:
            return self.receive_go_back_n(num_packages)
        elif self.protocol == 1:
            return self.receive_selective_repeat(num_packages)

    def receive_go_back_n(self, num_packages):
        result = []
        while True:
            frame = self.transmitter.get_receiver()
            if frame is None or frame.sequence != self.total_received:
                continue
            if not frame.is_corrupted:
                result.append(frame.message)
                self.total_received += 1
                if self.show_comments:
                    print("Receiver: Received package № %d" % frame.sequence)

                self.transmitter.send_sender(Package(frame.sequence, Message.ACK, False))
                self.total_sent_ack += 1
                if self.show_comments:
                    print("Receiver: Sent ack for package № %d" % frame.sequence)
                if self.total_sent_ack == num_packages:
                    if self.show_comments:
                        print("Receiver: finished")
                    return result
            else:
                self.transmitter.send_sender(Package(frame.sequence, Message.NAK, False))
                if self.show_comments:
                    print("Receiver: Sent nak for package № %d" % frame.sequence)

    def receive_selective_repeat(self, num_packages):
        result = []
        buffer = []
        while True:
            frame = self.transmitter.get_receiver()
            if frame is None:
                continue
            if self.total_received == frame.sequence:
                if frame.is_corrupted:
                    self.transmitter.send_sender(Package(frame.sequence, Message.NAK, False))
                    if self.show_comments:
                        print("Receiver: Sent nak for package № %d" % frame.sequence)
                    continue

                result.append(frame.message)
                self.total_received += 1
                if self.show_comments:
                    print("Receiver: Received package № %d" % frame.sequence)

                self.transmitter.send_sender(Package(frame.sequence, Message.ACK, False))
                self.total_sent_ack += 1
                if self.show_comments:
                    print("Receiver: Sent ack for package № %d" % frame.sequence)

                while len(buffer):
                    if buffer[0].is_corrupted:
                        self.transmitter.send_sender(Package(buffer[0].sequence, Message.NAK, False))
                        if self.show_comments:
                            print("Receiver: Sent nak for package № %d" % buffer[0].sequence)
                        del buffer[0]
                        break
                    result.append(buffer[0].message)
                    self.total_received += 1
                    if self.show_comments:
                        print("Receiver: Received package № %d" % buffer[0].sequence)
                    self.transmitter.send_sender(Package(buffer[0].sequence, Message.ACK, False))
                    self.total_sent_ack += 1
                    if self.show_comments:
                        print("Receiver: Sent ack for package № %d" % buffer[0].sequence)
                    del buffer[0]

                if self.total_sent_ack == num_packages:
                    if self.show_comments:
                        print("Receiver: finished")
                    return result
            else:
                buffer.append(frame)
