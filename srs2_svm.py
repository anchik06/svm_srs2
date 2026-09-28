import numpy as np
from scipy import optimize
import math

#1. задание функции f(x) и её производной
def f(x):
    return x**2 + x - 1

def df(x):
    return 2*x + 1

def phi(x):
    return 1.0 / (x + 1)

def dphi(x):
    return -1.0 / (x + 1)**2

#табулирование функции на заданном интервале
print("1. ТАБУЛИРОВАНИЕ ФУНКЦИИ на [0.5; 1]")
print(f"{'x':>8} | {'f(x)':>12}")
x_tab = np.linspace(0.5, 1.0, 11)
for xi in x_tab:
    print(f"{xi:8.4f} | {f(xi):12.6f}")

#2. программное определение участка, содержащего корень
print("2. ПОИСК УЧАСТКА, СОДЕРЖАЩЕГО КОРЕНЬ (условие f(a)*f(b) < 0)")
a, b = 0.5, 1.0
fa, fb = f(a), f(b)
print(f"a = {a}, f(a) = {fa:.6f}")
print(f"b = {b}, f(b) = {fb:.6f}")
print(f"f(a)*f(b) = {fa * fb:.6f}")
if fa * fb < 0:
    print("корень гарантированно существует на отрезке [a, b]")
else:
    print("условие f(a)*f(b) < 0 не выполняется!")

#3. метод бисекции
def bisection(f, a, b, eps, max_iter=100):
    history = []
    for k in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        err = (b - a) / 2.0
        history.append({
            'k': k, 'a': a, 'b': b,
            'x_curr': c, 'x_next': None,
            'f_x': fc, 'err': err
        })
        if abs(fc) < eps or err < eps:
            break
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, history

#4. метод хорд
def chord_method(f, a, b, eps, max_iter=100):
    history = []
    x_prev = a
    x_curr = b - f(b) * (b - a) / (f(b) - f(a))
    history.append({
        'k': 0, 'a': a, 'b': b,
        'x_curr': x_prev, 'x_next': x_curr,
        'f_x': f(x_prev), 'err': abs(x_curr - x_prev)
    })
    for k in range(1, max_iter):
        denom = f(x_curr) - f(x_prev)
        #защита от деления на ноль
        if abs(denom) < 1e-15:
            break
        x_next = x_curr - f(x_curr) * (x_curr - x_prev) / denom
        err = abs(x_next - x_curr)
        history.append({
            'k': k, 'a': a, 'b': b,
            'x_curr': x_curr, 'x_next': x_next,
            'f_x': f(x_curr), 'err': err
        })
        if err < eps or abs(f(x_next)) < eps:
            x_curr = x_next
            break
        x_prev = x_curr
        x_curr = x_next
    return x_curr, history

#5. метод ньютона
def newton_method(f, df, x0, eps, max_iter=100):
    history = []
    x_curr = x0
    for k in range(max_iter):
        fx = f(x_curr)
        dfx = df(x_curr)
        x_next = x_curr - fx / dfx
        err = abs(x_next - x_curr)
        history.append({
            'k': k, 'x_curr': x_curr, 'x_next': x_next,
            'f_x': fx, 'err': err
        })
        if err < eps or abs(f(x_next)) < eps:
            x_curr = x_next
            break
        x_curr = x_next
    return x_curr, history

#6. метод простой итерации
def simple_iteration(phi, x0, eps, max_iter=100):
    history = []
    x_curr = x0
    for k in range(max_iter):
        x_next = phi(x_curr)
        err = abs(x_next - x_curr)
        history.append({
            'k': k, 'x_curr': x_curr, 'x_next': x_next,
            'f_x': f(x_next), 'err': err
        })
        if err < eps:
            x_curr = x_next
            break
        x_curr = x_next
    return x_curr, history

#7–11. расчёты для 0,001 и 0,00001
epsilons = [1e-3, 1e-5]
results_table = []

def print_history(method_name, history, eps):
    """вывод таблицы итераций"""
    print(f"\n {method_name}, е = {eps} ")
    print(f"{'k':>3} | {'x_k':>14} | {'x_{k+1}':>14} | {'|ошибка|':>12} | {'|f(x)|':>12}")
    for row in history:
        xn = row['x_next'] if row['x_next'] is not None else row['x_curr']
        print(f"{row['k']:3d} | {row['x_curr']:14.9f} | {xn:14.9f} | "
              f"{row['err']:12.2e} | {abs(row['f_x']):12.2e}")

for eps in epsilons:
    print(f"РАСЧЁТЫ ДЛЯ ТОЧНОСТИ e = {eps}")

    #бисекция
    root_bis, hist_bis = bisection(f, a, b, eps)
    print_history("метод бисекции", hist_bis, eps)
    results_table.append({
        'method': 'Бисекция', 'eps': eps, 'x0': '-',
        'root': root_bis, 'iters': len(hist_bis),
        'residual': abs(f(root_bis)),
        'scipy': None, 'excel': None})

    # метод хорд
    root_ch, hist_ch = chord_method(f, a, b, eps)
    print_history("метод хорд", hist_ch, eps)
    results_table.append({
        'method': 'Хорды', 'eps': eps, 'x0': '-',
        'root': root_ch, 'iters': len(hist_ch),
        'residual': abs(f(root_ch)),
        'scipy': None, 'excel': None})

    # метод Ньютона (x0 = 0.5)
    root_n1, hist_n1 = newton_method(f, df, 0.5, eps)
    print_history("метод Ньютона (x0=0.5)", hist_n1, eps)
    results_table.append({
        'method': 'Ньютон (x0=0.5)', 'eps': eps, 'x0': 0.5,
        'root': root_n1, 'iters': len(hist_n1),
        'residual': abs(f(root_n1)),
        'scipy': None, 'excel': None
    })

    # метод Ньютона (x0 = 1) - второе начальное приближение
    root_n2, hist_n2 = newton_method(f, df, 1, eps)
    print_history("метод Ньютона (x0=1)", hist_n2, eps)
    results_table.append({
        'method': 'Ньютон (x0=1)', 'eps': eps, 'x0': 1,
        'root': root_n2, 'iters': len(hist_n2),
        'residual': abs(f(root_n2)),
        'scipy': None, 'excel': None})

    # простая итерация
    root_si, hist_si = simple_iteration(phi, 0.5, eps)
    print_history("простая итерация (x0=0.5)", hist_si, eps)
    results_table.append({
        'method': 'Простая итерация', 'eps': eps, 'x0': 0.5,
        'root': root_si, 'iters': len(hist_si),
        'residual': abs(f(root_si)),
        'scipy': None, 'excel': None
    })

# 10. проверка условия сходимости |ф'(x)| < 1
print("10. ПРОВЕРКА УСЛОВИЯ СХОДИМОСТИ |ф'(x)| < 1 на [0.5; 1]")
x_check = np.linspace(0.5, 1.0, 11)
print(f"{'x':>8} | {'|ф\'(x)|':>12} ")
max_dphi = 0
for xi in x_check:
    val = abs(dphi(xi))
    max_dphi = max(max_dphi, val)
    print(f"{xi:8.4f} | {val:12.6f} ")
print(f"максимум |ф'(x)| на [0.5; 1] = {max_dphi:.6f} < 1")
print("условие сходимости простой итерации ВЫПОЛНЯЕТСЯ")

#12. Проверка средствами SciPy
print("12. ПРОВЕРКА БИБЛИОТЕЧНЫМИ СРЕДСТВАМИ SciPy")
for eps in epsilons:
    print(f"\n Точность е = {eps}")

    # бисекция
    root_bisect_lib = optimize.bisect(f, a, b, xtol=eps, full_output=True)
    print(f"scipy.optimize.bisect:       x* = {root_bisect_lib[0]:.10f}, "
          f"итераций: {root_bisect_lib[1].iterations}")

    # ньютон
    root_newton_lib = optimize.newton(f, 0.5, fprime=df, tol=eps, full_output=True)
    print(f"scipy.optimize.newton:       x* = {root_newton_lib[0]:.10f}, "
          f"итераций: {root_newton_lib[1].iterations}")

    # root_scalar
    root_rs_lib = optimize.root_scalar(f, bracket=[a, b], method='brentq', xtol=eps)
    print(f"scipy.optimize.root_scalar:  x* = {root_rs_lib.root:.10f}, "
          f"итераций: {root_rs_lib.iterations}")


# 13. ИТОГОВАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА
print("13. ИТОГОВАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
#данные из Excel (вариант 1)
excel_data = {
    (1e-3, 'Бисекция'):           (0.6181641, 8),
    (1e-5, 'Бисекция'):           (0.6180344, 15),
    (1e-3, 'Хорды'):              (0.6179775, 3),
    (1e-5, 'Хорды'):              (0.6180328, 5),
    (1e-3, 'Ньютон (x0=0.5)'):    (0.6180340, 2),
    (1e-5, 'Ньютон (x0=0.5)'):    (0.6180340, 3),
    (1e-3, 'Простая итерация'):   (0.6181818, 6),
    (1e-5, 'Простая итерация'):   (0.6180328, 11),
}

# получаем значения из SciPy
scipy_bisect = {}
scipy_newton = {}
for eps in epsilons:
    r1 = optimize.bisect(f, a, b, xtol=eps, full_output=True)
    scipy_bisect[eps] = (r1[0], r1[1].iterations)
    r2 = optimize.newton(f, 0.5, fprime=df, tol=eps, full_output=True)
    scipy_newton[eps] = (r2[0], r2[1].iterations)

header = f"{'Метод':<22} | {'е':>8} | {'x0':>5} | {'Корень x*':>14} | {'Итер.':>5} | {'|f(x*)|':>10} | {'SciPy':>14} | {'Excel':>14}"
print(header)

for row in results_table:
    method = row['method']
    eps = row['eps']
    x0_str = str(row['x0'])

    # значение из Excel
    key_excel = (eps, method)
    if key_excel in excel_data:
        excel_root = f"{excel_data[key_excel][0]:.7f}"
    else:
        excel_root = "—"

    #значение из SciPy
    if 'Ньютон' in method:
        scipy_root = f"{scipy_newton[eps][0]:.10f}"
    else:
        scipy_root = f"{scipy_bisect[eps][0]:.10f}"

    print(f"{method:<22} | {eps:>8.1e} | {x0_str:>5} | {row['root']:14.10f} | "
          f"{row['iters']:5d} | {row['residual']:10.2e} | {scipy_root:>14} | {excel_root:>14}")

