import matplotlib.pyplot as plt
from scipy.stats import binom, chi2, norm
import make_binomial
import numpy as np
import math
import pandas as pd
from numpy import average, var

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


x = np.linspace(0, 45, 100)
y = [custom_CDF(make_binomial.binomial[7], t) for t in x]

real_y = [binom.cdf(t, make_binomial.n, make_binomial.theta) for t in x]

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
for i in range(len(make_binomial.N)):
    for j in range(len(make_binomial.N)):
        Dmn.append(D(make_binomial.binomial[i], make_binomial.binomial[j]))

data = np.reshape(Dmn, (8, 8))

table = pd.DataFrame(data, columns=make_binomial.N, index=make_binomial.N)
print(table)

# %% Построение гистограммы и полигона частот

x = np.arange(0, make_binomial.n, 1)

plt.hist(make_binomial.binomial[7], density=True, label='Гистограмма')
plt.plot(x, [binom.pmf(t, make_binomial.n, make_binomial.theta) for t in x], label='Ф-ия плотности')
plt.title('Построение гистограммы и полигона частот')
plt.legend()
plt.show()


# %% Вычисление выборочных моментов

E = make_binomial.n * make_binomial.theta
D = make_binomial.n * make_binomial.theta * (1 - make_binomial.theta)

means = [average(s) for s in make_binomial.binomial]
variances = [var(s) for s in make_binomial.binomial]

d = {'Выборочное среднее': means, 'Выборочная дисперсия': variances}
table = pd.DataFrame(data=d, index=make_binomial.N)
print(table)

absolute_means = [average(s) - E for s in make_binomial.binomial]
absolute_variances = [var(s) - D for s in make_binomial.binomial]

d_absolute = {'Абс погрешность выборочного среднего': absolute_means,
              'Абс погрешность выборочной дисперсии': absolute_variances}
table_absolute = pd.DataFrame(data=d_absolute, index=make_binomial.N)
print(table_absolute)

relative_means = ['{:.1%}'.format((average(s) - E)/E) for s in make_binomial.binomial]
relative_variances = ['{:.1%}'.format((var(s) - D)/D) for s in make_binomial.binomial]

d_relative = {'Относ погрешность выборочного среднего': relative_means,
              'Относ погрешность выборочной дисперсии': relative_variances}
table_relative = pd.DataFrame(data=d_relative, index=make_binomial.N)
print(table_relative)

# %% Получение оценок (метод моментов)

hat_theta1 = [1 - variances[i]/means[i] for i in range(len(make_binomial.binomial))]

table_htheta1 = pd.DataFrame(data=hat_theta1, columns=['Оценка theta'], index=make_binomial.N)
print(table_htheta1)


# %% Получение оценок (метод макс правдоподобия)

hat_theta2 = [means[i]/make_binomial.n for i in range(len(make_binomial.binomial))]

table_htheta2 = pd.DataFrame(data=hat_theta2, columns=['Оценка theta'], index=make_binomial.N)
print(table_htheta2)

# %% Оптимальная оценка для theta

optimal_theta = [means[i]/make_binomial.n for i in range(len(make_binomial.binomial))]

table_optimal_theta = pd.DataFrame(data=optimal_theta, columns=['Оптимальная оценка theta'], index=make_binomial.N)
print(table_optimal_theta)

# %% Критерий согласия Колмогорова (Смирнова)

def supremum_r(emp, true):
    x = np.arange(0, make_binomial.n, 1)
    s = 0
    for i in x:
        s = max(s, abs(emp[i] - true[i]))

    return s


x = np.arange(0, make_binomial.n, 1)
cdf_true = [binom.cdf(i, make_binomial.n, make_binomial.theta) for i in x]
criter_Kolm = [1.22, 1.36, 1.63]

test_Kolm = [math.sqrt(make_binomial.N[i])*supremum_r([custom_CDF(make_binomial.binomial[i], j) for j in x], cdf_true) for i in range(len(make_binomial.binomial))]

hyp_Kolm = [[u'H\u2080 принимается' if t < a else u'H\u2080 отвергается' for t in test_Kolm] for a in criter_Kolm]
data_Kolm = {'Тестовая статистика':test_Kolm, 'alpha=0.1':hyp_Kolm[0],'alpha=0.05':hyp_Kolm[1],'alpha=0.01':hyp_Kolm[2]}
table_Kolm = pd.DataFrame(data=data_Kolm, index=make_binomial.N)

print(table_Kolm)


# %% Критерий согласия хи-квадрат

def test_chi_calc(sample: list, sample_index: int):
    s = 0
    l = list(set(sorted(sample)))
    freq = [sample.count(i) for i in l]

    for i in range(len(l)):
        prob = binom.pmf(l[i], make_binomial.n, make_binomial.theta)
        s += (freq[i] - sample_index * prob) ** 2 / (sample_index * prob)

    return s


def accept_chi(sample, s, a):
    l = list(set(sorted(sample)))
    chi = chi2.ppf(1 - a, df=len(l) - 1)

    return u'H\u2080 принимается' if s <= chi else u'H\u2080 отвергается'


test_chi = [test_chi_calc(make_binomial.binomial[i], make_binomial.N[i]) for i in range(len(make_binomial.binomial))]

crits = [0.1, 0.05, 0.01]
hyp_chi = [[accept_chi(make_binomial.binomial[i], test_chi[i], a) for i in range(len(make_binomial.binomial))] for a in crits]

chi = {'Тестовая статистика': test_chi, 'alpha=0.1': hyp_chi[0], 'alpha=0.05': hyp_chi[1],
         'alpha=0.01': hyp_chi[2]}
chi_table = pd.DataFrame(data=chi, index=make_binomial.N)
print(chi_table)


# %% {Критерий согласия Колмогорова (Смирнова) для сложной гипотезы

def supremum_optimal(emp, theta):
  x = np.arange(0, make_binomial.n, 1)
  true = [binom.cdf(i, make_binomial.n, theta) for i in x]
  s = 0
  for i in x:
    s = max(s, abs(emp[i] - true[i]))

  return s

test_kolmogorov = [math.sqrt(make_binomial.N[i])*supremum_optimal([custom_CDF(make_binomial.binomial[i], j) for j in x], hat_theta2[i]) for i in range(len(make_binomial.binomial))]
crit_kolmogorov = [1.22, 1.36, 1.63]
hyp_kolmogorov = [[u'H\u2080 принимается' if t<a else u'H\u2080 отвергается' for t in test_kolmogorov] for a in crit_kolmogorov]
kolmogorov = {'Тестовая статистика':test_kolmogorov, 'alpha=0.1':hyp_kolmogorov[0],'alpha=0.05':hyp_kolmogorov[1],'alpha=0.01':hyp_kolmogorov[2]}
kolmogorov_table = pd.DataFrame(data=kolmogorov, index=make_binomial.N)
print(kolmogorov_table)

# %% Критерий согласия хи-квадрат для сложной гипотезы

def test_chi_calc_o(sample: list, sample_index: int, theta):
    s = 0
    l = list(set(sorted(sample)))
    freq = [sample.count(i) for i in l]

    for i in range(len(l)):
        prob = binom.pmf(l[i], make_binomial.n, theta)
        s += (freq[i] - sample_index * prob) ** 2 / (sample_index * prob)

    return s


def accept_chi(sample, s, a):
    l = list(set(sorted(sample)))
    chi = chi2.ppf(1 - a, df=len(l) - 1)

    return u'H\u2080 принимается' if s <= chi else u'H\u2080 отвергается'

test_chi = [test_chi_calc_o(make_binomial.binomial[i], make_binomial.N[i], hat_theta2[i]) for i in range(len(make_binomial.binomial))]

crits = [0.1, 0.05, 0.01]
hyp_chi = [[accept_chi(make_binomial.binomial[i], test_chi[i], a) for i in range(len(make_binomial.binomial))] for a in crits]

chi = {'Тестовая статистика': test_chi, 'alpha=0.1': hyp_chi[0], 'alpha=0.05': hyp_chi[1], 'alpha=0.01': hyp_chi[2]}
chi_table = pd.DataFrame(data=chi, index=make_binomial.N)
print(chi_table)


# %% Проверка гипотезы об однородности выборок

Student = 1.282

for i in range(len(make_binomial.N)):
    for j in range(len(make_binomial.N)):
        t = (means[i] - means[j])/math.sqrt((make_binomial.N[j]-1) * variances[i] + (make_binomial.N[i] - 1) *
                                            variances[j]) * math.sqrt(make_binomial.N[j]*make_binomial.N[i]*
                                                                      (make_binomial.N[j] + make_binomial.N[i] - 2)/
                                                                      (make_binomial.N[j]+make_binomial.N[i]))

hyp = [[u'H\u2080 принимается' if abs(t)<=Student else u'H\u2080 отвергается' for t in d] for d in np.reshape(Dmn, (8, 8))]
test = {
    '5':hyp[0],
    '10':hyp[1],
    '100':hyp[2],
    '200':hyp[3],
    '400':hyp[4],
    '600':hyp[5],
    '800':hyp[6],
    '1000':hyp[7],
}

table_d = pd.DataFrame(data=test, index=make_binomial.N)
print(table_d)


# %% Вычисление критической области

theta_0 = make_binomial.theta
theta_1 = theta_0 + 0.03

y = [binom.ppf(0.99, i * make_binomial.n, theta_0) for i in make_binomial.N]

sums = []

for i in make_binomial.binomial:
    sums.append(sum(i))

data = {'Sum(x_i)': sums, 'c_alpha': y,
        'Итог': ['Принимаем theta = ' + (f"{theta_0}" if sums[i] < y[i] else f"{theta_1}") for i in
                 range(len(make_binomial.N))]}

table_crit = pd.DataFrame(data, index=make_binomial.N)
print(table_crit)

beta = [binom.cdf(y[i], value * make_binomial.n, theta_1) for i, value in enumerate(make_binomial.N)]

data_beta = {'Sum(x_i)': sums, 'c_alpha': y, 'beta': beta}

table_crit_beta = pd.DataFrame(data=data_beta, index=make_binomial.N)
print(table_crit_beta)

# %% Вычисление минимального необходимого количества...

alpha = [0.1, 0.05, 0.01]
beta = [0.1, 0.05, 0.01]
n_ab = []


for a, b in [(0.1, 0.9), (0.05, 0.95), (0.01, 0.99)]:
    t_a = norm.ppf(a, loc=0, scale=1)
    t_b = norm.ppf(b, loc=0, scale=1)
    n_ab.append(math.ceil(((t_b * math.sqrt(theta_1 * (1-theta_1)) - t_a * math.sqrt(theta_0 * (1-theta_0)))/(math.sqrt(35) * (theta_0 - theta_1)))**2))


data_ab = {'alpha': alpha, 'beta': beta, 'n': n_ab}
table_ab = pd.DataFrame(data=data_ab, index=['', '', ''])

print(table_ab)
