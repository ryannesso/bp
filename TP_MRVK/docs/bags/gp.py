#!/usr/bin/env python3
import rosbag
import matplotlib.pyplot as plt
import numpy as np

# ================= НАСТРОЙКИ =================
BAG_FILE = '5_shp_ok_1_with_norandom.bag'
ROBOT_RADIUS = 0.3  # Укажите радиус вашего робота (в метрах)
SPHERE_RADIUS = 0.3  # Укажите радиус сфер (в метрах)
SAFETY_MARGIN = 0.1  # Запас безопасности
# Линия столкновения: если график упадет ниже этой цифры - была авария
COLLISION_THRESHOLD = ROBOT_RADIUS + SPHERE_RADIUS
SAFE_DISTANCE = COLLISION_THRESHOLD + SAFETY_MARGIN
# =============================================

def main():
    print(f"Открытие файла {BAG_FILE}...")
    bag = rosbag.Bag(BAG_FILE)

    # Хранилища для данных
    robot = {'t': [], 'x': [], 'y': [], 'v':[], 'w': []}
    # Инициализируем хранилище для 9 сфер (от 1 до 9 включительно)
    spheres = {i: {'t': [], 'x': [], 'y':[]} for i in range(1, 10)}
    
    slam_map_msg = None
    global_costmap_msg = None

    # 1. Чтение данных из bag-файла
    for topic, msg, t in bag.read_messages():
        if topic == '/odometry/filtered':
            robot['t'].append(t.to_sec())
            
            orig_x = msg.pose.pose.position.x
            orig_y = msg.pose.pose.position.y
            
            # Точный поворот и смещение из robot.launch:
            # Смещение: x = -0.205792, y = -4.379866
            # Поворот (yaw): 1.568965 рад (~90 градусов)
            init_x = -0.205792
            init_y = -4.379866
            init_yaw = 1.568965
            
            # Матрица поворота + смещение из локальной одометрии (0,0) в глобальные координаты Gazebo
            world_x = init_x + orig_x * np.cos(init_yaw) - orig_y * np.sin(init_yaw)
            world_y = init_y + orig_x * np.sin(init_yaw) + orig_y * np.cos(init_yaw)
            
            robot['x'].append(world_x)
            robot['y'].append(world_y)
            # Вычисляем полную линейную скорость (вектор)
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            robot['v'].append(np.hypot(vx, vy))
            robot['w'].append(msg.twist.twist.angular.z)
            
        elif topic.startswith('/moving_sphere_'):
            # Извлекаем номер сферы из топика (например, из '/moving_sphere_3/odom' достаем 3)
            try:
                sphere_id = int(topic.split('/')[1].split('_')[2])
                spheres[sphere_id]['t'].append(t.to_sec())
                spheres[sphere_id]['x'].append(msg.pose.pose.position.x)
                spheres[sphere_id]['y'].append(msg.pose.pose.position.y)
            except Exception as e:
                pass
        
        elif topic == '/map':
            slam_map_msg = msg
        elif topic == '/move_base/global_costmap/costmap':
            global_costmap_msg = msg
    bag.close()
    print("Данные успешно прочитаны!")

    # Конвертируем в numpy arrays для удобства
    r_t = np.array(robot['t'])
    # Нормализуем время (чтобы график начинался с 0 секунд)
    t0 = r_t[0]
    r_t_norm = r_t - t0
    
    r_x = np.array(robot['x'])
    r_y = np.array(robot['y'])
    r_v = np.array(robot['v'])
    r_w = np.array(robot['w'])

    # 2. Вычисление дистанции до препятствий
    min_distances = []
    distances_to_spheres = {sid: [] for sid in range(1, 10)} # Рассчитано до 9 сфер
    
    for i in range(len(r_t)):
        current_time = r_t[i]
        curr_rx = r_x[i]
        curr_ry = r_y[i]
        
        dists = []
        for sid in range(1, 10):
            if len(spheres.get(sid, {}).get('t', [])) == 0:
                distances_to_spheres[sid].append(float('nan'))
                continue
            
            # Так как сообщения приходят в разное время, интерполируем позицию сферы 
            # на точный момент времени позиции робота
            sx = np.interp(current_time, spheres[sid]['t'], spheres[sid]['x'])
            sy = np.interp(current_time, spheres[sid]['t'], spheres[sid]['y'])
            
            dist = np.hypot(curr_rx - sx, curr_ry - sy)
            dists.append(dist)
            distances_to_spheres[sid].append(dist)
            
        if dists:
            min_distances.append(min(dists))
        else:
            min_distances.append(float('nan'))

    import os
    
    # Создаем папку с именем bag файла (без .bag)
    bag_basename = os.path.splitext(os.path.basename(BAG_FILE))[0]
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), bag_basename)
    os.makedirs(output_dir, exist_ok=True)
    
    DPI = 800

    # Цветовая палитра для сфер
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'cyan', 'magenta', 'olive', 'gray']

    print(f"Сохранение графиков в папку: {output_dir}")

    # =========================================================
    # График 1: 2D Траектории
    # =========================================================
    fig_traj = plt.figure(figsize=(10, 8))
    ax_traj = fig_traj.add_subplot(111)
    ax_traj.plot(r_x, r_y, label='Robot Path', color='blue', linewidth=2.5)
    ax_traj.plot(r_x[0], r_y[0], 'go', markersize=8, label='Start') # Старт
    ax_traj.plot(r_x[-1], r_y[-1], 'ro', markersize=8, label='Goal') # Финиш
    
    for sid in range(1, 10):
        if len(spheres.get(sid, {}).get('x', [])) > 0:
            ax_traj.plot(spheres[sid]['x'], spheres[sid]['y'], '--', 
                         color=colors[(sid-1)%len(colors)], label=f'Sphere {sid}', alpha=0.7)
            
    ax_traj.set_title('Topological Path (2D Trajectories)', fontsize=14, fontweight='bold')
    ax_traj.set_xlabel('X [meters]', fontsize=12)
    ax_traj.set_ylabel('Y [meters]', fontsize=12)
    ax_traj.legend()
    ax_traj.grid(True, linestyle=':', alpha=0.7)
    ax_traj.axis('equal') # Сохраняем пропорции пространства

    fig_traj.tight_layout()
    fig_traj.savefig(os.path.join(output_dir, '1_trajectory.png'), dpi=DPI)
    fig_traj.savefig(os.path.join(output_dir, '1_trajectory.pdf'), dpi=DPI)
    plt.close(fig_traj)

    # =========================================================
    # График 2: Безопасность
    # =========================================================
    fig_dist = plt.figure(figsize=(10, 6))
    ax_dist = fig_dist.add_subplot(111)
    
    # Рисуем график расстояния для КАЖДОЙ сферы
    for sid in range(1, 10):
        if len(distances_to_spheres.get(sid, [])) > 0 and not np.isnan(distances_to_spheres[sid]).all():
            ax_dist.plot(r_t_norm, distances_to_spheres[sid], color=colors[(sid-1)%len(colors)], alpha=0.4, label=f'To Sphere {sid}')

    # Поверх всех рисуем минимальную дистанцию жирно
    ax_dist.plot(r_t_norm, min_distances, color='black', linewidth=2.5, label='Min Distance')
    
    # Рисуем критическую красную линию столкновения
    ax_dist.axhline(y=COLLISION_THRESHOLD, color='red', linestyle='--', linewidth=2, label='Collision Threshold')
    # Рисуем зеленую линию безопасного буфера
    ax_dist.axhline(y=SAFE_DISTANCE, color='green', linestyle=':', linewidth=2, label='Safe Distance')
    
    # Заливаем красным зону под критической линией для наглядности
    ax_dist.fill_between(r_t_norm, 0, COLLISION_THRESHOLD, color='red', alpha=0.1)

    ax_dist.set_title('Safety: Minimum Distance to Nearest Sphere', fontsize=12, fontweight='bold')
    ax_dist.set_xlabel('Time [seconds]', fontsize=12)
    ax_dist.set_ylabel('Distance [m]', fontsize=12)
    ax_dist.grid(True, linestyle=':', alpha=0.7)
    # Помещаем легенду за пределами основного окна, чтобы график не перекрывался
    ax_dist.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig_dist.tight_layout()
    fig_dist.savefig(os.path.join(output_dir, '2_safety.png'), dpi=DPI, bbox_inches='tight')
    fig_dist.savefig(os.path.join(output_dir, '2_safety.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close(fig_dist)

    # =========================================================
    # График 3: Скорости
    # =========================================================
    fig_vel = plt.figure(figsize=(10, 6))
    ax_vel = fig_vel.add_subplot(111)
    
    ax_vel.plot(r_t_norm, r_v, color='blue', label='Linear Velocity (m/s)', linewidth=2)
    
    # Создаем вторую ось Y для угловой скорости (чтобы графики не сплющило)
    ax_w = ax_vel.twinx()
    ax_w.plot(r_t_norm, r_w, color='orange', label='Angular Velocity (rad/s)', linewidth=1.5, alpha=0.8)
    
    ax_vel.set_title('Kinematics: Robot Velocities over Time', fontsize=12, fontweight='bold')
    ax_vel.set_xlabel('Time [seconds]', fontsize=12)
    ax_vel.set_ylabel('Linear Vel [m/s]', color='blue', fontsize=12)
    ax_w.set_ylabel('Angular Vel [rad/s]', color='orange', fontsize=12)
    
    ax_vel.grid(True, linestyle=':', alpha=0.7)
    
    # Собираем легенды с двух осей в одну
    lines1, labels1 = ax_vel.get_legend_handles_labels()
    lines2, labels2 = ax_w.get_legend_handles_labels()
    ax_vel.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig_vel.tight_layout()
    fig_vel.savefig(os.path.join(output_dir, '3_kinematics.png'), dpi=DPI)
    fig_vel.savefig(os.path.join(output_dir, '3_kinematics.pdf'), dpi=DPI)
    plt.close(fig_vel)

    # =========================================================
    # Вспомогательная функция для генерации карт
    # =========================================================
    def plot_occupancy_grid(msg, title, filename_base):
        if msg is None:
            return
            
        # Загружаем данные и решейпим в 2D
        data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        
        # Поворачиваем всю карту на 90 градусов влево (против часовой стрелки)
        # Это нужно, так как одометрия робота повернута относительно локальной системы координат карты
        rotated_data = np.rot90(data, k=1)
        
        # ROS OccupancyGrid: -1: Неизвестно, 0: Свободно, 100: Препятствие
        # Создаем маску: -1 заменяем на 25 (светло-серый, как в RViz)
        masked_data = np.where(rotated_data == -1, 25, rotated_data)
        
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        w = msg.info.width
        h = msg.info.height
        
        # После поворота на 90 градусов влево (x,y) -> (-y, x)
        # Рассчитываем новые границы extent для imshow
        # Оригинальные границы: x [ox, ox + w*res], y [oy, oy + h*res]
        # Новые границы в "повернутом" мире:
        new_xmin = - (oy + h * res)
        new_xmax = - oy
        new_ymin = ox
        new_ymax = ox + w * res
        extent = [new_xmin, new_xmax, new_ymin, new_ymax]
                  
        fig_map = plt.figure(figsize=(10, 8))
        ax_map = fig_map.add_subplot(111)
        
        # origin='lower' потому что в OccGrid нулевой индекс это нижний левый угол
        ax_map.imshow(masked_data, cmap='gray_r', origin='lower', extent=extent, vmin=0, vmax=100)
        
        ax_map.set_title(title, fontsize=14, fontweight='bold')
        ax_map.set_xlabel('X [meters]', fontsize=12)
        ax_map.set_ylabel('Y [meters]', fontsize=12)
        ax_map.grid(False)
        ax_map.axis('equal')
        
        # Накладываем траекторию робота для контекста
        # Используем оригинальные r_x, r_y, так как они уже в мировых координатах
        ax_map.plot(r_x, r_y, label='Robot Path', color='blue', linewidth=1.5, alpha=0.8)
        ax_map.legend(loc='best')

        fig_map.tight_layout()
        fig_map.savefig(os.path.join(output_dir, f'{filename_base}.png'), dpi=DPI)
        fig_map.savefig(os.path.join(output_dir, f'{filename_base}.pdf'), dpi=DPI)
        plt.close(fig_map)

    # =========================================================
    # График 4: SLAM Map
    # =========================================================
    if slam_map_msg:
        plot_occupancy_grid(slam_map_msg, 'SLAM Map (/map)', '4_slam_map')

    # =========================================================
    # График 5: Global Costmap
    # =========================================================
    if global_costmap_msg:
        plot_occupancy_grid(global_costmap_msg, 'Global Costmap', '5_global_costmap')

    print("✅ Все графики успешно сохранены в форматах PNG и PDF!")

if __name__ == '__main__':
    main()