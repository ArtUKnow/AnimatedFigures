import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from scipy.integrate import odeint

# Функция для вычисления системы дифференциальных уравнений
def diff_eq(y, t, func_v, func_w):
    y1, y2, y3, y4 = y
    dydt = [y3, y4, func_v(y1, y2, y3, y4), func_w(y1, y2, y3, y4)]
    return dydt

# Символьное объявление переменных
time_sym = sp.Symbol("t")
x_func = sp.Function('x')(time_sym)
fi_func = sp.Function('fi')(time_sym)
v_func = sp.Function('v')(time_sym)
w_func = sp.Function('w')(time_sym)

# КОНФИГ
rect_width = 1  # ширина прямоугольника
rect_length = 2  # длина прямоугольника
circle_radius = 0.2  # радиус круга
spring_distance = 3  # расстояние до пружин
mass_rect = 2  # масса прямоугольника
mass_circle = 4  # масса груза
gravity = 9.8  # ускорение свободного падения
rod_length = 2  # длина стержня
spring_stiffness = 100 # жесткость пружин
initial_conditions = [0, 0, 0, 0]  # начальные условия: x0, φ0, v0, w0

# Вычисление уравнений Лагранжа
# Определение коэффициентов матрицы инерции
a11 = mass_rect + mass_circle
a12 = -mass_circle * rod_length * sp.sin(fi_func)
a21 = sp.sin(fi_func)
a22 = -rod_length
# Определение правой стороны уравнений движения
b1 = -2 * spring_stiffness * x_func * (1 - (spring_distance**2 / sp.sqrt(x_func**2 + spring_distance**2))) + (mass_rect + mass_circle) * gravity - w_func**2 * sp.cos(fi_func) * mass_circle * rod_length
b2 = gravity * sp.sin(fi_func)

# Вычисление детерминантов для решения системы уравнений
detA = a11 * a22 - a12 * a21
detA1 = b1 * a22 - b2 * a21
detA2 = a11 * b2 - b1 * a21

# Определение ускорений по x и φ 
dvdt = detA1 / detA  # Ускорение по x
dwdt = detA2 / detA  # Ускорение по φ

# Временная сетка для решения уравнений
time_grid = np.linspace(0, 50, 500)

# Преобразование символьных выражений в функции
func_v = sp.lambdify([x_func, fi_func, v_func, w_func], dvdt, "numpy")
func_w = sp.lambdify([x_func, fi_func, v_func, w_func], dwdt, "numpy")

# Решение системы дифференциальных уравнений
solution = odeint(diff_eq, initial_conditions, time_grid, args=(func_v, func_w))

# Вычисление реакции вертикальных направляющих
solution_ra = odeint(diff_eq, initial_conditions, time_grid, args=(func_v, func_w))
reaction_force = [mass_circle * rod_length * (solution_ra[i, 3] * np.cos(solution[i, 1]) - solution_ra[i, 1]**2 * np.sin(solution[i, 1])) for i in range(len(solution))]

# Координаты точки A (центр прямоугольника)
AX = np.zeros_like(solution[:, 0])
AY = -solution[:, 0]

# Координаты точки B (центр груза)
BX = rod_length * np.sin(solution[:, 1])
BY = -(rod_length * np.cos(solution[:, 1]) + solution[:, 0])

# Начало построения графиков и анимации
fig = plt.figure()
plt.suptitle('Графики и Симуляция', fontsize=16)

# Настройка оси для симуляции
ax_sim = fig.add_subplot(1, 2, 1)
ax_sim.axis("equal")
ax_sim.set_title('Симуляция', fontsize=14)

# Подготовка данных для окружения
wall_left_x = [-rect_width / 2, -rect_width / 2]
wall_right_x = [rect_width / 2, rect_width / 2]
wall_y = [min(AY) - rect_length, max(AY) + rect_length]

# Построение окружения
ax_sim.plot(0, 0, marker=".", color="red")  # красная точка
ax_sim.plot(wall_left_x, wall_y, linestyle='--', color="grey")  # левая стена
ax_sim.plot(wall_right_x, wall_y, linestyle='--',  color="grey")  # правая стена
spring_left, = ax_sim.plot([-spring_distance, -rect_length / 2], [0, AY[0] + rect_width / 2], color="grey")  # левая пружина
spring_right, = ax_sim.plot([spring_distance, rect_length / 2], [0, AY[0] + rect_width / 2], color="grey")  # правая пружина
ax_sim.plot(-spring_distance, 0, marker=".", color="black")  # левое соединение
ax_sim.plot(spring_distance, 0, marker=".", color="black")  # правое соединение
ax_sim.axhline(0, linestyle=':', color='k')  # горизонтальная пунктирная линия

# Создание прямоугольника и круга
rect = Rectangle((-rect_width / 2, AY[0]), rect_width, rect_length, color="black")  # прямоугольник
circ = Circle((BX[0], BY[0]), circle_radius, color="grey")  # круг

# Построение радиус-вектора точки B
radius_vector, = ax_sim.plot([0, BX[0]], [0, BY[0]], color="grey")

# Построение графиков для x(t), φ(t) и RA(t)
ax_x = fig.add_subplot(4, 2, 2)
ax_x.plot(time_grid, solution[:, 0], color="black")
ax_x.set_title('График x(t)', fontsize=14)

ax_fi = fig.add_subplot(4, 2, 4)
ax_fi.plot(time_grid, solution[:, 1], color="black")
ax_fi.set_title('График φ(t)', fontsize=14)

ax_ra = fig.add_subplot(4, 2, 6)
ax_ra.plot(time_grid, reaction_force, color="black")
ax_ra.set_title('График RA(t)', fontsize=14)

plt.subplots_adjust(wspace=0.4, hspace=0.8)

# Добавление подписей к точкам, грузу и ползунку
ax_sim.text(-spring_distance - 0.2, 0, 'D', ha='right', va='bottom')
ax_sim.text(spring_distance + 0.2, 0, 'E', ha='left', va='bottom')
text_A = ax_sim.text(0, AY[0] - 0.2, 'A', ha='right', va='bottom')
text_B = ax_sim.text(BX[0] + 0.2, BY[0], 'B', ha='right', va='bottom')

# Функция для инициализации позиций
def init():
    rect.set_y(-rect_length / 2)
    ax_sim.add_patch(rect)
    circ.center = (0, 0)
    ax_sim.add_patch(circ)
    return rect, circ

# Функция для создания пружины
def spring(start, end, num_segments=6, amplitude=0.1):
    x_vals = np.linspace(start[0], end[0], num_segments)
    y_vals = np.linspace(start[1], end[1], num_segments)
    dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    amp_factor = amplitude * (2 + 0.5 * dist)  # Изменение амплитуды пружины в зависимости от расстояния
    for i in range(1, num_segments, 2):
        y_vals[i] += amp_factor
    return x_vals, y_vals

# Функция для обновления позиций на каждом кадре анимации
def animate(i):
    rect.set_y(AY[i] - rect_length / 2)
    spring_left_x, spring_left_y = spring((-spring_distance, 0), (-rect_width / 2, AY[i]), num_segments=12, amplitude=0.1)
    spring_right_x, spring_right_y = spring((spring_distance, 0), (rect_width / 2, AY[i]), num_segments=12, amplitude=0.1)
    spring_left.set_data(spring_left_x, spring_left_y)
    spring_right.set_data(spring_right_x, spring_right_y)
    radius_vector.set_data([0, BX[i]], [AY[i], BY[i]])
    circ.center = (BX[i], BY[i])

    # Обновление позиций текстовых меток для ползунка и груза
    text_A.set_position((-1, AY[i] - 0.2))  # Ползунок A
    text_B.set_position((BX[i] + 0.2, BY[i] + 0.2))  # Груз B

    return spring_left, spring_right, rect, radius_vector, circ

# Запуск анимации
anim = FuncAnimation(fig, animate, init_func=init, frames=500, blit=False, repeat=True, repeat_delay=0)
plt.show()
