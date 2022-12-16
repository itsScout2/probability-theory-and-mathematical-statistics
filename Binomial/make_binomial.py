import random
import matplotlib.pyplot as plt
import seaborn as sbn


binomial = []


def generate_binomial(n, p):
    c = p / (1 - p)
    r = (1 - p) ** n
    s = r
    k = 0
    x = random.uniform(0, 1)

    while x > s:
        k += 1
        r *= c * (n - k + 1) / k
        s += r

    return k


n = 35
theta = 0.5
N = [5, 10, 100, 200, 400, 600, 800, 1000]
array = []

for j in N:
    for i in range(j):
        num = generate_binomial(n, theta)
        array.append(num)
    binomial.append(array)
    array = []

sbn.kdeplot(binomial[7])  # Т.к. данная выборка должна быть точнее остальных
plt.title("Функция плотности вероятности")
leg = [r"$n = 35, \theta = 0.5$"]
plt.legend(leg, loc="upper right")
plt.grid()
plt.show()
