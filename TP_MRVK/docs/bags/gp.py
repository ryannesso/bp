#!/usr/bin/env python3
import rosbag
import matplotlib.pyplot as plt
import numpy as np
import math

# ================= НАСТРОЙКИ =================
BAG_FILE = 'cohead_dwa.bag'
ROBOT_RADIUS = 0.3  # Укажите радиус вашего робота (в метрах)
SPHERE_RADIUS = 0.3  # Укажите радиус сфер (в метрах)
SAFETY_MARGIN = 0.1  # Запас безопасности
# Линия столкновения: если график упадет ниже этой цифры - была авария
COLLISION_THRESHOLD = ROBOT_RADIUS + SPHERE_RADIUS
SAFE_DISTANCE = COLLISION_THRESHOLD + SAFETY_MARGIN
USE_TF_FOR_MAP = True  # Если есть /tf с map<->odom, используем для выравнивания карты
# =============================================

def quat_to_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )

def invert_transform(tx, ty, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    inv_x = -c * tx - s * ty
    inv_y = s * tx - c * ty
    inv_yaw = -yaw
    return inv_x, inv_y, inv_yaw

def apply_transform(x, y, tx, ty, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    out_x = tx + c * x - s * y
    out_y = ty + s * x + c * y
    return out_x, out_y

def main():
    print(f"Открытие файла {BAG_FILE}...")
    bag = rosbag.Bag(BAG_FILE)

    # Хранилища для данных
    robot = {'t': [], 'x': [], 'y': [], 'v':[], 'w': [], 'x_odom': [], 'y_odom': []}
    # Инициализируем хранилище для 9 сфер (от 1 до 9 включительно)
    spheres = {i: {'t': [], 'x': [], 'y':[]} for i in range(1, 10)}
    
    slam_map_msg = None
    global_costmap_msg = None
    tf_odom_map = []

    # 1. Чтение данных из bag-файла
    for topic, msg, t in bag.read_messages():
        if topic == '/odometry/filtered':
            robot['t'].append(t.to_sec())
            
            orig_x = msg.pose.pose.position.x
            orig_y = msg.pose.pose.position.y
            robot['x_odom'].append(orig_x)
            robot['y_odom'].append(orig_y)
            
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

        elif topic == '/tf' or topic == '/tf_static':
            for tr in msg.transforms:
                parent = tr.header.frame_id.strip('/')
                child = tr.child_frame_id.strip('/')
                if (parent == 'odom' and child == 'map') or (parent == 'map' and child == 'odom'):
                    tx = tr.transform.translation.x
                    ty = tr.transform.translation.y
                    yaw = quat_to_yaw(tr.transform.rotation)
                    if parent == 'map' and child == 'odom':
                        tx, ty, yaw = invert_transform(tx, ty, yaw)
                    stamp = tr.header.stamp.to_sec()
                    if stamp == 0.0:
                        stamp = t.to_sec()
                    tf_odom_map.append((stamp, tx, ty, yaw))
        
        elif topic == '/map':
            slam_map_msg = msg
        elif topic == '/move_base/global_costmap/costmap':
            global_costmap_msg = msg
    bag.close()
    print("Данные успешно прочитаны!")

    if not robot['t']:
        print("No /odometry/filtered messages found. Nothing to plot.")
        return

    # Конвертируем в numpy arrays для удобства
    r_t = np.array(robot['t'])
    # Нормализуем время (чтобы график начинался с 0 секунд)
    t0 = r_t[0]
    r_t_norm = r_t - t0
    
    r_x_world = np.array(robot['x'])
    r_y_world = np.array(robot['y'])
    r_x_odom = np.array(robot['x_odom'])
    r_y_odom = np.array(robot['y_odom'])
    r_v = np.array(robot['v'])
    r_w = np.array(robot['w'])

    use_tf_map = USE_TF_FOR_MAP and len(tf_odom_map) > 0
    if use_tf_map:
        tf_odom_map.sort(key=lambda item: item[0])
        tf_times = np.array([item[0] for item in tf_odom_map])

        def get_tf_at(ts):
            idx = int(np.searchsorted(tf_times, ts, side='right') - 1)
            if idx < 0:
                idx = 0
            return tf_odom_map[idx]

        r_x_map = []
        r_y_map = []
        for i, ts in enumerate(r_t):
            _, tx, ty, yaw = get_tf_at(ts)
            mx, my = apply_transform(r_x_odom[i], r_y_odom[i], tx, ty, yaw)
            r_x_map.append(mx)
            r_y_map.append(my)
        r_x_map = np.array(r_x_map)
        r_y_map = np.array(r_y_map)
    else:
        if USE_TF_FOR_MAP:
            print("No /tf map<->odom transform found. Using odom coordinates for map overlay.")
            print("If the map is still shifted, record /tf and /tf_static in the bag.")
        r_x_map = r_x_odom
        r_y_map = r_y_odom

    # 2. Вычисление дистанции до препятствий
    min_distances = []
    distances_to_spheres = {sid: [] for sid in range(1, 10)} # Рассчитано до 9 сфер
    
    for i in range(len(r_t)):
        current_time = r_t[i]
        curr_rx = r_x_world[i]
        curr_ry = r_y_world[i]
        
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

    # ================= SUMMARY STATISTICS =================
    min_distances_np = np.array(min_distances, dtype=float)
    duration = float(r_t_norm[-1]) if len(r_t_norm) else 0.0

    def nanmin_or_nan(arr):
        return float(np.nanmin(arr)) if np.isfinite(arr).any() else float('nan')

    min_dist_value = nanmin_or_nan(min_distances_np)

    if len(r_t_norm) >= 2:
        dt = np.diff(r_t_norm)
        coll_mask = np.isfinite(min_distances_np) & (min_distances_np < COLLISION_THRESHOLD)
        safe_mask = np.isfinite(min_distances_np) & (min_distances_np < SAFE_DISTANCE)
        collision_time = float(np.sum(dt[coll_mask[:-1]]))
        unsafe_time = float(np.sum(dt[safe_mask[:-1]]))
    else:
        collision_time = 0.0
        unsafe_time = 0.0

    collision = np.isfinite(min_dist_value) and (min_dist_value < COLLISION_THRESHOLD)
    success = not collision

    avg_v = float(np.nanmean(r_v)) if len(r_v) else float('nan')
    max_v = float(np.nanmax(r_v)) if len(r_v) else float('nan')
    avg_abs_w = float(np.nanmean(np.abs(r_w))) if len(r_w) else float('nan')

    sphere_min = {}
    for sid in range(1, 10):
        arr = np.array(distances_to_spheres[sid], dtype=float)
        sphere_min[sid] = nanmin_or_nan(arr)

    closest_sphere_id = None
    closest_sphere_dist = float('nan')
    if any(np.isfinite(list(sphere_min.values()))):
        closest_sphere_id = min(
            sphere_min,
            key=lambda k: sphere_min[k] if np.isfinite(sphere_min[k]) else float('inf')
        )
        closest_sphere_dist = sphere_min[closest_sphere_id]

    collision_pct = (collision_time / duration * 100.0) if duration > 0 else 0.0
    unsafe_pct = (unsafe_time / duration * 100.0) if duration > 0 else 0.0

    print("\n===== SUMMARY =====")
    print(f"Bag: {BAG_FILE}")
    print(f"Duration [s]: {duration:.2f}")
    print(f"Min distance [m]: {min_dist_value:.3f}")
    print(f"Collision: {collision}")
    print(f"Time < collision threshold [s]: {collision_time:.2f} ({collision_pct:.2f}%)")
    print(f"Time < safe distance [s]: {unsafe_time:.2f} ({unsafe_pct:.2f}%)")
    print(f"Avg linear v [m/s]: {avg_v:.3f}")
    print(f"Max linear v [m/s]: {max_v:.3f}")
    print(f"Avg angular |w| [rad/s]: {avg_abs_w:.3f}")
    if closest_sphere_id is not None:
        print(f"Closest sphere: {closest_sphere_id} (min {closest_sphere_dist:.3f} m)")
    print(f"Experiment success: {success}")

    summary_txt = os.path.join(output_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Bag: {}\n".format(BAG_FILE))
        f.write("Duration_s: {:.2f}\n".format(duration))
        f.write("Min_distance_m: {:.3f}\n".format(min_dist_value))
        f.write("Collision: {}\n".format(collision))
        f.write("Collision_time_s: {:.2f}\n".format(collision_time))
        f.write("Collision_pct: {:.2f}\n".format(collision_pct))
        f.write("Unsafe_time_s: {:.2f}\n".format(unsafe_time))
        f.write("Unsafe_pct: {:.2f}\n".format(unsafe_pct))
        f.write("Avg_linear_v: {:.3f}\n".format(avg_v))
        f.write("Max_linear_v: {:.3f}\n".format(max_v))
        f.write("Avg_abs_angular_w: {:.3f}\n".format(avg_abs_w))
        if closest_sphere_id is not None:
            f.write("Closest_sphere_id: {}\n".format(closest_sphere_id))
            f.write("Closest_sphere_min_m: {:.3f}\n".format(closest_sphere_dist))
        f.write("Experiment_success: {}\n".format(success))

    summary_csv = os.path.join(output_dir, "summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(
            "bag,duration_s,min_distance_m,collision,collision_time_s,collision_pct,"
            "unsafe_time_s,unsafe_pct,avg_linear_v,max_linear_v,avg_abs_angular_w,"
            "closest_sphere_id,closest_sphere_min_m,experiment_success\n"
        )
        f.write(
            "{bag},{duration:.2f},{min_dist:.3f},{collision},{coll_time:.2f},{coll_pct:.2f},"
            "{unsafe_time:.2f},{unsafe_pct:.2f},{avg_v:.3f},{max_v:.3f},{avg_w:.3f},"
            "{closest_id},{closest_dist:.3f},{success}\n".format(
                bag=BAG_FILE,
                duration=duration,
                min_dist=min_dist_value,
                collision=collision,
                coll_time=collision_time,
                coll_pct=collision_pct,
                unsafe_time=unsafe_time,
                unsafe_pct=unsafe_pct,
                avg_v=avg_v,
                max_v=max_v,
                avg_w=avg_abs_w,
                closest_id=(closest_sphere_id if closest_sphere_id is not None else ""),
                closest_dist=closest_sphere_dist,
                success=success
            )
        )

    # Цветовая палитра для сфер
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'cyan', 'magenta', 'olive', 'gray']

    print(f"Сохранение графиков в папку: {output_dir}")

    # =========================================================
    # График 1: 2D Траектории
    # =========================================================
    fig_traj = plt.figure(figsize=(10, 8))
    ax_traj = fig_traj.add_subplot(111)
    ax_traj.plot(r_x_world, r_y_world, label='Robot Path', color='blue', linewidth=2.5)
    ax_traj.plot(r_x_world[0], r_y_world[0], 'go', markersize=8, label='Start') # Старт
    ax_traj.plot(r_x_world[-1], r_y_world[-1], 'ro', markersize=8, label='Goal') # Финиш
    
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
        
        # ROS OccupancyGrid: -1: Неизвестно, 0: Свободно, 100: Препятствие
        # Создаем маску: -1 заменяем на 25 (светло-серый, как в RViz)
        masked_data = np.where(data == -1, 25, data)
        
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        w = msg.info.width
        h = msg.info.height
        
        # Границы карты в системе координат map
        extent = [ox, ox + w * res, oy, oy + h * res]
                  
        fig_map = plt.figure(figsize=(10, 8))
        ax_map = fig_map.add_subplot(111)
        
        # origin='lower' потому что в OccGrid нулевой индекс это нижний левый угол
        ax_map.imshow(masked_data, cmap='gray_r', origin='lower', extent=extent, vmin=0, vmax=100)
        
        ax_map.set_title(title, fontsize=14, fontweight='bold')
        ax_map.set_xlabel('X [meters]', fontsize=12)
        ax_map.set_ylabel('Y [meters]', fontsize=12)
        ax_map.grid(False)
        ax_map.axis('equal')
        
        # Накладываем траекторию робота в карте
        ax_map.plot(r_x_map, r_y_map, label='Robot Path', color='blue', linewidth=1.5, alpha=0.8)
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