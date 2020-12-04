from enum import Enum
from multiprocessing import Queue


class Package:
    def __init__(self, sequence, message, is_corrupted):
        self.sequence = sequence
        self.message = message
        self.is_corrupted = is_corrupted


class Message(Enum):
    ACK = 0
    NAK = 1


class Transmitter:
    def __init__(self):
        self.sender_queue = Queue()
        self.receiver_queue = Queue()

    def get_sender(self):
        if self.sender_queue.empty():
            return None
        return self.sender_queue.get()

    def get_receiver(self):
        if self.receiver_queue.empty():
            return None
        return self.receiver_queue.get()

    def send_receiver(self, msg):
        self.receiver_queue.put(msg)

    def send_sender(self, msg):
        self.sender_queue.put(msg)
