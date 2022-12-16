import matplotlib.pyplot as plt
import make_erlang
import numpy as np
import math
import pandas as pd
from numpy import average, var
from scipy.stats import chi2, erlang, norm

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.options.display.expand_frame_repr = False


# %% Построение эмпирической функции распределения

def custom_CDF(s, t):
    summa = 0
    for i in s:
        summa += int(i <= t)
    return summa / len(s)


x = np.linspace(0, 180, 1000)
y = [custom_CDF(make_erlang.erlang[7], t) for t in x]

real_y = [1 - sum([1 / math.factorial(i) * math.exp(-make_erlang.theta * t)
                   * (make_erlang.theta * t) ** i for i in range(make_erlang.m)]) for t in x]

plt.plot(x, y, label='Эмперическая ф-ия')
plt.plot(x, real_y, label='Ф-ия распределения')
plt.title('Построение эмпирической функции распределения')
plt.grid()
plt.legend()
plt.show()


# %% Dmn

def supremum(s1, s2):
    x = np.linspace(0, 180, 1000)
    f1 = [custom_CDF(s1, t) for t in x]
    f2 = [custom_CDF(s2, t) for t in x]
    s = 0
    for i in range(len(x)):
        s = max(s, abs(f1[i] - f2[i]))
    return s


def D(s1, s2):
    m = len(s1)
    n = len(s2)

    return math.sqrt((m * n) / (m + n)) * supremum(s1, s2)


Dmn = []
for i in range(len(make_erlang.N)):
    for j in range(len(make_erlang.N)):
        Dmn.append(D(make_erlang.erlang[i], make_erlang.erlang[j]))

data = np.reshape(Dmn, (8, 8))

table = pd.DataFrame(data, columns=make_erlang.N, index=make_erlang.N)
print(table)

# %% Построение гистограммы и полигона частот

plt.hist(make_erlang.erlang[7], density=True, label='Гистограмма')
plt.plot(x, [(make_erlang.theta ** make_erlang.m *
              t ** (make_erlang.m - 1) * math.exp(-make_erlang.theta *
                                                  t)) / math.factorial(make_erlang.m - 1)
             for t in x], label='Ф-ия плотности')
plt.title('Построение гистограммы и полигона частот')
plt.legend()
plt.show()

# %% Вычисление выборочных моментов

E = make_erlang.m / make_erlang.theta
D = make_erlang.m / np.power(make_erlang.theta, 2)

means = [average(s) for s in make_erlang.erlang]
variances = [var(s) for s in make_erlang.erlang]

d = {'Выборочное среднее': means, 'Выборочная дисперсия': variances}
table = pd.DataFrame(data=d, index=make_erlang.N)
print(table)

absolute_means = [average(s) - E for s in make_erlang.erlang]
absolute_variances = [var(s) - D for s in make_erlang.erlang]

d_absolute = {'Абс погрешность выборочного среднего': absolute_means,
              'Абс погрешность выборочной дисперсии': absolute_variances}
table_absolute = pd.DataFrame(data=d_absolute, index=make_erlang.N)
print(table_absolute)

relative_means = ['{:.1%}'.format((average(s) - E) / E) for s in make_erlang.erlang]
relative_variances = ['{:.1%}'.format((var(s) - D) / D) for s in make_erlang.erlang]

d_relative = {'Относ погрешность выборочного среднего': relative_means,
              'Относ погрешность выборочной дисперсии': relative_variances}
table_relative = pd.DataFrame(data=d_relative, index=make_erlang.N)
print(table_relative)

# %% Получение оценок (метод моментов)

hat_theta1 = [means[i] / variances[i] for i in range(len(make_erlang.erlang))]

table_htheta1 = pd.DataFrame(data=hat_theta1, columns=['Оценка theta'], index=make_erlang.N)
print(table_htheta1)

# %% Получение оценок (метод макс правдоподобия)

hat_theta2 = [make_erlang.m / means[i] for i in range(len(make_erlang.erlang))]

table_htheta2 = pd.DataFrame(data=hat_theta2, columns=['Оценка theta'], index=make_erlang.N)
print(table_htheta2)

# %% Оптимальная оценка для theta

optimal_theta = [8 * len(make_erlang.erlang[i]) - 1 /
                 (sum(make_erlang.erlang[i]))
                 for i in range(len(make_erlang.erlang))]

table_optimal_theta = pd.DataFrame(data=optimal_theta, columns=['Оптимальная оценка theta'], index=make_erlang.N)
print(table_optimal_theta)


# %% Критерий согласия Колмогорова (Смирнова)

def supremum_r(emp, true):
    x = np.linspace(0, 180, 1000)
    s = 0
    for i in range(len(x)):
        s = max(s, abs(emp[i] - true[i]))

    return s


x = np.linspace(0, 180, 1000)
cdf_true = [1 - sum([1 / math.factorial(i) * math.exp(-make_erlang.theta * t)
                     * (make_erlang.theta * t) ** i for i in range(make_erlang.m)]) for t in x]
criter_Kolm = [1.22, 1.36, 1.63]

test_Kolm = [math.sqrt(make_erlang.N[i]) * supremum_r([custom_CDF(make_erlang.erlang[i], j) for j in x], cdf_true) for i
             in range(len(make_erlang.erlang))]

hyp_Kolm = [[u'H\u2080 принимается' if t < a else u'H\u2080 отвергается' for t in test_Kolm] for a in criter_Kolm]
data_Kolm = {'Тестовая статистика': test_Kolm, 'alpha=0.1': hyp_Kolm[0], 'alpha=0.05': hyp_Kolm[1],
             'alpha=0.01': hyp_Kolm[2]}
table_Kolm = pd.DataFrame(data=data_Kolm, index=make_erlang.N)

print(table_Kolm)


# %% Критерий согласия хи-квадрат

def test_chi_calc(sample: list, sample_index: int):
    s = 0
    l = list(sorted(sample))
    freq = []
    N = 50
    delta = (math.floor(l[-1]) + 1 - round(l[0])) / N
    lb, rb = math.floor(l[0]), math.floor(l[0]) + delta
    c = 0
    i = 0
    x = []
    # print([lb, rb])

    while i <= len(l):
        if i == len(l):
            x.append(lb + delta / 2)
            freq.append(c)
            break
        elif lb <= l[i] < rb:
            c += 1
            i += 1
        else:
            x.append(lb + delta / 2)
            freq.append(c)
            c = 0
            lb, rb = rb, rb + delta

    for i in range(len(x) - 1):
        prob = erlang.cdf(x[i + 1] - delta / 2, a=make_erlang.m, scale=1 / make_erlang.theta) - erlang.cdf(
            x[i] - delta / 2, a=make_erlang.m, scale=1 / make_erlang.theta)
        s += (freq[i] - sample_index * prob) ** 2 / (sample_index * prob)

    prob = erlang.cdf(x[-1] + delta / 2, a=make_erlang.m, scale=1 / make_erlang.theta) - erlang.cdf(x[-1] - delta / 2,
                                                                                                    a=make_erlang.m,
                                                                                                    scale=1 / make_erlang.theta)

    s += (freq[-1] - sample_index * prob) ** 2 / (sample_index * prob)

    return s


def accept_chi(sample, s, a):
    l = list(set(sample))
    chi = chi2.ppf(1 - a, df=len(l) - 1)

    return u'H\u2080 принимается' if s <= chi else u'H\u2080 отвергается'


test_chi_m = [test_chi_calc(make_erlang.erlang[i], make_erlang.N[i]) for i in range(len(make_erlang.erlang))]

crits = [0.1, 0.05, 0.01]
hyp_chi = [[accept_chi(make_erlang.erlang[i], test_chi_m[i], a) for i in range(len(make_erlang.erlang))] for a in crits]

chi = {'Тестовая статистика': test_chi_m, 'alpha=0.1': hyp_chi[0], 'alpha=0.05': hyp_chi[1],
       'alpha=0.01': hyp_chi[2]}
chi_table = pd.DataFrame(data=chi, index=make_erlang.N)
print(chi_table)


# %% Критерий согласия Колмогорова (Смирнова) для сложной гипотезы

def supremum_optimal(sample, theta):
    x = np.linspace(0, 180, 1000)
    s = 0
    for i in range(len(x)):
        s = max(s, abs(custom_CDF(sample, i) - erlang.cdf(i, a=make_erlang.m, scale=1 / theta)))
    return s


x = np.linspace(0, 180, 1000)

test_kolmogorov = [math.sqrt(make_erlang.N[i]) * supremum_optimal(make_erlang.erlang[i], hat_theta2[i]) for i in
                   range(len(make_erlang.erlang))]
crit_kolmogorov = [1.22, 1.36, 1.63]
hyp_kolmogorov = [[u'H\u2080 принимается' if t < a else u'H\u2080 отвергается' for t in test_kolmogorov] for a in
                  crit_kolmogorov]
kolmogorov = {'Тестовая статистика': test_kolmogorov, 'alpha=0.1': hyp_kolmogorov[0], 'alpha=0.05': hyp_kolmogorov[1],
              'alpha=0.01': hyp_kolmogorov[2]}
kolmogorov_table = pd.DataFrame(data=kolmogorov, index=make_erlang.N)
print(kolmogorov_table)


# %% Критерий согласия хи-квадрат для сложной гипотезы

def test_chi_m_calc_o(sample: list, sample_index: int, theta):
    s = 0
    l = list(sorted(sample))
    freq = []
    N = 50
    delta = (math.floor(l[-1]) + 1 - round(l[0])) / N
    lb, rb = math.floor(l[0]), math.floor(l[0]) + delta
    c = 0
    i = 0
    x = []
    # print([lb, rb])

    while i <= len(l):
        if i == len(l):
            x.append(lb + delta / 2)
            freq.append(c)
            break
        elif lb <= l[i] < rb:
            c += 1
            i += 1
        else:
            x.append(lb + delta / 2)
            freq.append(c)
            c = 0
            lb, rb = rb, rb + delta

    for i in range(len(x) - 1):
        prob = erlang.cdf(x[i + 1] - delta / 2, a=make_erlang.m, scale=1 / theta) - erlang.cdf(x[i] - delta / 2,
                                                                                               a=make_erlang.m,
                                                                                               scale=1 / theta)
        s += (freq[i] - sample_index * prob) ** 2 / (sample_index * prob)

    prob = erlang.cdf(x[-1] + delta / 2, a=make_erlang.m, scale=1 / theta) - erlang.cdf(x[-1] - delta / 2,
                                                                                        a=make_erlang.m,
                                                                                        scale=1 / theta)

    s += (freq[-1] - sample_index * prob) ** 2 / (sample_index * prob)

    return s


def accept_chi(sample, s, a):
    l = list(set(sample))
    chi = chi2.ppf(1 - a, df=len(l) - 1)

    return u'H\u2080 принимается' if s <= chi else u'H\u2080 отвергается'


test_chi = [test_chi_m_calc_o(make_erlang.erlang[i], make_erlang.N[i], hat_theta2[i]) for i in
            range(len(make_erlang.erlang))]

crits = [0.1, 0.05, 0.01]
hyp_chi = [[accept_chi(make_erlang.erlang[i], test_chi_m[i], a) for i in range(len(make_erlang.erlang))] for a in crits]

chi = {'Тестовая статистика': test_chi, 'alpha=0.1': hyp_chi[0], 'alpha=0.05': hyp_chi[1], 'alpha=0.01': hyp_chi[2]}
chi_table = pd.DataFrame(data=chi, index=make_erlang.N)
print(chi_table)

# %% Проверка гипотезы об однородности выборок


Kolm = 1.22

hyp_smirnov = [[u'H\u2080 принимается' if t <= Kolm * math.sqrt(
    1 / make_erlang.N[i] + 1 / make_erlang.N[j]) else u'H\u2080 отвергается' for i, t in enumerate(d)] for j, d in
               enumerate(np.reshape(Dmn, (8, 8)))]

smirnov = {
    '5': hyp_smirnov[0],
    '10': hyp_smirnov[1],
    '100': hyp_smirnov[2],
    '200': hyp_smirnov[3],
    '400': hyp_smirnov[4],
    '600': hyp_smirnov[5],
    '800': hyp_smirnov[6],
    '1000': hyp_smirnov[7],
}

table_d = pd.DataFrame(data=smirnov, index=make_erlang.N)
print(table_d)

# %% Вычисление критической области

theta_0 = make_erlang.theta
theta_1 = theta_0 - 0.01

y = [erlang.ppf(0.99, a=i * make_erlang.m, scale=1/theta_0) for i in make_erlang.N]

sums = []

for i in make_erlang.erlang:
    sums.append(sum(i))

data = {'Sum(x_i)': sums, 'c_alpha': y,
        'Итог': ['Принимаем theta = ' + (f"{theta_0}" if sums[i] < y[i] else f"{theta_1}") for i in
                 range(len(make_erlang.N))]}

table_crit = pd.DataFrame(data, index=make_erlang.N)
print(table_crit)

beta = [erlang.cdf(y[i], a=value * make_erlang.m, scale=1/theta_1) for i, value in enumerate(make_erlang.N)]

data_beta = {'Sum(x_i)': sums, 'c_alpha': y, 'beta': beta}

table_crit_beta = pd.DataFrame(data=data_beta, index=make_erlang.N)
print(table_crit_beta)


# %% Вычисление минимального необходимого количества...

alpha = [0.1, 0.05, 0.01]
beta = [0.1, 0.05, 0.01]
n_ab = []


for a, b in [(0.1, 0.9), (0.05, 0.95), (0.01, 0.99)]:
    t_a = norm.ppf(a, loc=0, scale=1)
    t_b = norm.ppf(b, loc=0, scale=1)
    n_ab.append(math.ceil(((t_a*theta_0 - t_b*theta_1)/(math.sqrt(8)*(theta_1 - theta_0)))**2))


data_ab = {'alpha': alpha, 'beta': beta, 'n': n_ab}
table_ab = pd.DataFrame(data=data_ab, index=['', '', ''])

print(table_ab)
