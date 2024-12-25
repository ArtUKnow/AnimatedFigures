import numpy as np
import sympy as sym
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Определение символической переменной времени
time = sym.Symbol('t')

# Исходные урвнения из условия
radius = 1 + sym.sin(8 * time)
angle_phi = time + 0.5 * sym.sin(8 * time)

# Выражения координат
coord_x = radius * sym.cos(angle_phi)
coord_y = radius * sym.sin(angle_phi)

# Вычисление производных для скорости
velocity_x = sym.diff(coord_x, time)
velocity_y = sym.diff(coord_y, time)
speed = sym.sqrt(velocity_x**2 + velocity_y**2)

# Вычисление производных для ускорения
accel_x = sym.diff(velocity_x, time)
accel_y = sym.diff(velocity_y, time)
accel_magnitude = sym.sqrt(accel_x**2 + accel_y**2)

# Тангенциальное и нормальное ускорение
tangential_accel = sym.diff(speed, time)
normal_accel = sym.sqrt(accel_magnitude**2 - tangential_accel**2)

# Радиус кривизны
radius_of_curvature = speed**2 / normal_accel

# Компоненты тангенциального ускорения
tangential_accel_x = (velocity_x / speed) * tangential_accel
tangential_accel_y = (velocity_y / speed) * tangential_accel

# Нормальный единичный вектор
normal_accel_x = accel_x - tangential_accel_x
normal_accel_y = accel_y - tangential_accel_y
normal_magnitude = sym.sqrt(normal_accel_x**2 + normal_accel_y**2)
unit_normal_x = normal_accel_x / normal_magnitude
unit_normal_y = normal_accel_y / normal_magnitude

# Радиус кривизны по осям
curvature_radius_x = unit_normal_x * radius_of_curvature
curvature_radius_y = unit_normal_y * radius_of_curvature

# Временной массив
time_values = np.linspace(0, 2, 300)  # 2 секунды

# Инициализация массивов для хранения данных
pos_x = np.zeros_like(time_values)
pos_y = np.zeros_like(time_values)
vel_x = np.zeros_like(time_values)
vel_y = np.zeros_like(time_values)
accel_x_vals = np.zeros_like(time_values)
accel_y_vals = np.zeros_like(time_values)
radius_vec_x = np.zeros_like(time_values)
radius_vec_y = np.zeros_like(time_values)
curv_radius_x = np.zeros_like(time_values)
curv_radius_y = np.zeros_like(time_values)

# Вычисление значений для каждого момента времени
for idx, t_val in enumerate(time_values):
    # Позиция
    pos_x[idx] = coord_x.subs(time, t_val).evalf()
    pos_y[idx] = coord_y.subs(time, t_val).evalf()
    
    # Скорость
    vel_x[idx] = velocity_x.subs(time, t_val).evalf()
    vel_y[idx] = velocity_y.subs(time, t_val).evalf()
    
    # Ускорение
    accel_x_vals[idx] = accel_x.subs(time, t_val).evalf()
    accel_y_vals[idx] = accel_y.subs(time, t_val).evalf()
    
    # Радиус-вектор
    radius_vec_x[idx] = coord_x.subs(time, t_val).evalf()
    radius_vec_y[idx] = coord_y.subs(time, t_val).evalf()
    
    # Радиус кривизны
    curv_radius_x[idx] = curvature_radius_x.subs(time, t_val).evalf()
    curv_radius_y[idx] = curvature_radius_y.subs(time, t_val).evalf()

# Настройка графика
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(np.min(pos_x) - 1, np.max(pos_x) + 1)
ax.set_ylim(np.min(pos_y) - 1, np.max(pos_y) + 1)

# Траектория движения
ax.plot(pos_x, pos_y, label='Траектория')

# Оси координат
ax.axhline(0, color='black')
ax.axvline(0, color='black')

# Инициализация элементов анимации
point, = ax.plot(pos_x[0], pos_y[0], 'o', label='Точка')
velocity_line, = ax.plot(
    [pos_x[0], pos_x[0] + vel_x[0]],
    [pos_y[0], pos_y[0] + vel_y[0]],
    'r-', label='Скорость'
)
accel_line, = ax.plot(
    [pos_x[0], pos_x[0] + accel_x_vals[0]],
    [pos_y[0], pos_y[0] + accel_y_vals[0]],
    'g-', label='Ускорение'
)
radius_vector, = ax.plot(
    [0, pos_x[0]],
    [0, pos_y[0]],
    'c-', label='Радиус-вектор'
)
curvature_vector, = ax.plot(
    [pos_x[0], pos_x[0] + curv_radius_x[0]],
    [pos_y[0], pos_y[0] + curv_radius_y[0]],
    'b-', label='Радиус кривизны'
)

# Настройка стрелок
arrow_scale = 0.3
arrow_shape_x = np.array([-0.2 * arrow_scale, 0, -0.2 * arrow_scale])
arrow_shape_y = np.array([0.1 * arrow_scale, 0, -0.1 * arrow_scale])

def add_rotated_arrow(x, y, dx, dy, color):
    angle = math.atan2(dy, dx)
    rotated_x = arrow_shape_x * np.cos(angle) - arrow_shape_y * np.sin(angle)
    rotated_y = arrow_shape_x * np.sin(angle) + arrow_shape_y * np.cos(angle)
    arrow, = ax.plot(
        rotated_x + x + dx,
        rotated_y + y + dy,
        color + '-'
    )
    return arrow

# Добавление стрелок
velocity_arrow = add_rotated_arrow(pos_x[0], pos_y[0], vel_x[0], vel_y[0], 'r')
accel_arrow = add_rotated_arrow(pos_x[0], pos_y[0], accel_x_vals[0], accel_y_vals[0], 'g')
radius_arrow = add_rotated_arrow(pos_x[0], pos_y[0], 0, 0, 'c')
curvature_arrow = add_rotated_arrow(
    pos_x[0], pos_y[0], curv_radius_x[0], curv_radius_y[0], 'b'
)

def update_animation(frame):
    # Обновление позиции точки
    point.set_data(pos_x[frame], pos_y[frame])
    
    # Обновление линии скорости
    velocity_line.set_data(
        [pos_x[frame], pos_x[frame] + vel_x[frame]],
        [pos_y[frame], pos_y[frame] + vel_y[frame]]
    )
    
    # Обновление линии ускорения
    accel_line.set_data(
        [pos_x[frame], pos_x[frame] + accel_x_vals[frame]],
        [pos_y[frame], pos_y[frame] + accel_y_vals[frame]]
    )
    
    # Обновление радиус-вектора
    radius_vector.set_data(
        [0, pos_x[frame]],
        [0, pos_y[frame]]
    )
    
    # Обновление радиуса кривизны
    curvature_vector.set_data(
        [pos_x[frame], pos_x[frame] + curv_radius_x[frame]],
        [pos_y[frame], pos_y[frame] + curv_radius_y[frame]]
    )
    
    # Обновление стрелки скорости
    velocity_arrow.set_data(
        arrow_shape_x * np.cos(math.atan2(vel_y[frame], vel_x[frame])) - arrow_shape_y * np.sin(math.atan2(vel_y[frame], vel_x[frame])) + pos_x[frame] + vel_x[frame],
        arrow_shape_x * np.sin(math.atan2(vel_y[frame], vel_x[frame])) + arrow_shape_y * np.cos(math.atan2(vel_y[frame], vel_x[frame])) + pos_y[frame] + vel_y[frame]
    )
    
    # Обновление стрелки ускорения
    accel_arrow.set_data(
        arrow_shape_x * np.cos(math.atan2(accel_y_vals[frame], accel_x_vals[frame])) - arrow_shape_y * np.sin(math.atan2(accel_y_vals[frame], accel_x_vals[frame])) + pos_x[frame] + accel_x_vals[frame],
        arrow_shape_x * np.sin(math.atan2(accel_y_vals[frame], accel_x_vals[frame])) + arrow_shape_y * np.cos(math.atan2(accel_y_vals[frame], accel_x_vals[frame])) + pos_y[frame] + accel_y_vals[frame]
    )
    
    # Обновление стрелки радиус-вектора
    radius_arrow.set_data(
        arrow_shape_x * np.cos(math.atan2(pos_y[frame], pos_x[frame])) - arrow_shape_y * np.sin(math.atan2(pos_y[frame], pos_x[frame])) + pos_x[frame],
        arrow_shape_x * np.sin(math.atan2(pos_y[frame], pos_x[frame])) + arrow_shape_y * np.cos(math.atan2(pos_y[frame], pos_x[frame])) + pos_y[frame]
    )
    
    # Обновление стрелки радиуса кривизны
    curvature_arrow.set_data(
        arrow_shape_x * np.cos(math.atan2(curv_radius_y[frame], curv_radius_x[frame])) - arrow_shape_y * np.sin(math.atan2(curv_radius_y[frame], curv_radius_x[frame])) + pos_x[frame] + curv_radius_x[frame],
        arrow_shape_x * np.sin(math.atan2(curv_radius_y[frame], curv_radius_x[frame])) + arrow_shape_y * np.cos(math.atan2(curv_radius_y[frame], curv_radius_x[frame])) + pos_y[frame] + curv_radius_y[frame]
    )

# Создание анимации
animation = FuncAnimation(
    fig, update_animation, frames=300, interval=50, repeat=False
)

# Добавление подписей
ax.legend()
plt.show()
