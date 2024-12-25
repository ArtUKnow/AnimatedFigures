import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

# КОНФИГ
rect_width = 1       # Ширина прямоугольника
rect_length = 2      # Длина прямоугольника
circle_radius = 0.2  # Радиус круга
spring_distance = 3  # Расстояние до пружин
rod_length = 2       # Длина стержня

# Начальные позиции
AX = 0
AY = -0.5
BX = rod_length * np.sin(np.pi / 6)
BY = -(rod_length * np.cos(np.pi / 6) + AY)

# Начало построения графиков и анимации
fig = plt.figure(figsize=(12, 6))
plt.suptitle('Анимация', fontsize=16)

# Настройка оси для симуляции
ax_sim = fig.add_subplot(1, 2, 1)
ax_sim.axis("equal")
ax_sim.set_xlim(-spring_distance - rect_width - 1, spring_distance + rect_width + 1)
ax_sim.set_ylim(-spring_distance - rect_length - 1, spring_distance + rect_length + 1)
ax_sim.set_title('Симуляция', fontsize=14)

# Подготовка данных для окружения
wall_left_x = [-rect_width / 2, -rect_width / 2]
wall_right_x = [rect_width / 2, rect_width / 2]
wall_y = [ -spring_distance - rect_length, spring_distance + rect_length]

# Построение окружения
ax_sim.plot(0, 0, marker=".", color="red")  # Красная точка O
ax_sim.plot(wall_left_x, wall_y, linestyle='--', color="grey")  # Левая стена
ax_sim.plot(wall_right_x, wall_y, linestyle='--', color="grey")   # Правая стена

# Инициализация пружин с пустыми данными
spring_left, = ax_sim.plot([], [], color="grey")    # Левая пружина
spring_right, = ax_sim.plot([], [], color="grey")   # Правая пружина

# Соединения
ax_sim.plot(-spring_distance, 0, marker=".", color="black")  # Левое соединение
ax_sim.plot(spring_distance, 0, marker=".", color="black")   # Правое соединение
ax_sim.axhline(0, linestyle=':', color='k')                   # Горизонтальная пунктирная линия

# Создание прямоугольника и круга
rect = Rectangle((-rect_width / 2, AY - rect_length / 2), rect_width, rect_length, color="black")
circ = Circle((BX, BY), circle_radius, color="grey")

# Построение радиус-вектора точки B
radius_vector, = ax_sim.plot([AX, BX], [AY, BY], color="grey")

# Добавление патчей для прямоугольника и круга
ax_sim.add_patch(rect)
ax_sim.add_patch(circ)

# Добавление подписей к точкам, грузу и ползунку
ax_sim.text(-spring_distance - 0.2, 0, 'D', ha='right', va='bottom')
ax_sim.text(spring_distance + 0.2, 0, 'E', ha='left', va='bottom')
text_A = ax_sim.text(AX, AY - rect_length / 2 - 0.2, 'A', ha='right', va='bottom')
text_B = ax_sim.text(BX + 0.2, BY, 'B', ha='left', va='bottom')

# Функция для создания пружины
def spring(start, end, num_segments=12, amplitude=0.1):
    x_vals = np.linspace(start[0], end[0], num_segments)
    y_vals = np.linspace(start[1], end[1], num_segments)
    dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    amp_factor = amplitude * (2 + 0.5 * dist)  # Изменение амплитуды пружины в зависимости от расстояния
    for i in range(1, num_segments, 2):
        y_vals[i] += amp_factor
    return x_vals, y_vals

# Функция для инициализации позиций
def init():
    rect.set_y(AY - rect_length / 2)
    circ.center = (BX, BY)
    radius_vector.set_data([AX, BX], [AY, BY])
    spring_left.set_data([], [])
    spring_right.set_data([], [])
    return rect, circ, radius_vector, spring_left, spring_right, text_A, text_B

# Функция для обновления позиций на каждом кадре анимации
def animate(i):
    t = np.linspace(0, 2 * np.pi, 500)[i]
    
    # Плавное движение прямоугольника (вверх-вниз по синусоиде)
    rect_y = 2 * np.sin(t)
    rect.set_y(rect_y - rect_length / 2)
    
    # Координата центра прямоугольника
    AX = 0
    AY = rect_y

    # Плавное вращение круга вокруг центра прямоугольника
    angle = t
    circ_x = AX + rod_length * np.cos(angle)
    circ_y = AY + rod_length * np.sin(angle)
    circ.center = (circ_x, circ_y)
    
    # Обновление радиус-вектора
    radius_vector.set_data([AX, circ_x], [AY, circ_y])
    
    # Обновление пружин с новыми координатами
    spring_left_x, spring_left_y = spring((-spring_distance, 0), (-rect_width / 2, rect_y), num_segments=12, amplitude=0.1)
    spring_right_x, spring_right_y = spring((spring_distance, 0), (rect_width / 2, rect_y), num_segments=12, amplitude=0.1)
    spring_left.set_data(spring_left_x, spring_left_y)
    spring_right.set_data(spring_right_x, spring_right_y)
    
    # Обновление подписей
    text_A.set_position((AX, AY - rect_length / 2 - 0.2))
    text_B.set_position((circ_x + 0.2, circ_y))
    
    return spring_left, spring_right, rect, radius_vector, circ, text_A, text_B

anim = FuncAnimation(
    fig, 
    animate, 
    init_func=init, 
    frames=len(np.linspace(0, 2 * np.pi, 500)), 
    blit=True, 
    interval=20,    # Интервал между кадрами в миллисекундах
    repeat=True
)

plt.show()
