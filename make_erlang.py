import random
import math
import matplotlib.pyplot as plt
import seaborn as sbn

erlang = []


def generate_erlang(a, b):
    k = 1
    for j in range(a):
        k *= random.uniform(0, 1)

    return -math.log(k) / b


N = [5, 10, 100, 200, 400, 600, 800, 1000]
m = 8
theta = 0.125
array = []

for n in N:
    for i in range(n):
        num = generate_erlang(m, theta)
        array.append(num)
    erlang.append(array)
    array = []

sbn.kdeplot(erlang[7])  # Т.к. данная выборка должна быть точнее остальных
plt.title("Функция плотности вероятности")
leg = [r"$m = 8, \theta = 0.125$"]
plt.legend(leg, loc="upper right")
plt.grid()
plt.show()
