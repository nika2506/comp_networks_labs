from numpy import random
import matplotlib.pyplot as plt

def rand_is_corrupted(p):
    return random.rand() < p


def make_plot_packages(data, time_gbn, time_sr):
    plt.plot(data, time_gbn, 'r', label="Go back N")
    plt.plot(data, time_sr, 'b', label="Selective repeat")
    plt.xlabel('Кол-во пакетов')
    plt.ylabel('Время работы программы')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def make_plot_wind(window_size, time_gbn, time_sr):
    plt.plot(window_size, time_gbn, 'r', label="Go back N")
    plt.plot(window_size, time_sr, 'b', label="Selective repeat")
    plt.xlabel('Размер окна')
    plt.ylabel('Время работы программы')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()