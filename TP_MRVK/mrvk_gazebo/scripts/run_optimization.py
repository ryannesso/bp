#!/usr/bin/env python3
import rospy
import actionlib
import subprocess
import tf
import signal
import sys
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
from skopt import gp_minimize, dump, load
from skopt.space import Real, Integer
from gazebo_msgs.msg import ContactsState

# --- КОНСТАНТЫ И НАСТРОЙКИ ---
CHECKPOINT_PATH = "/tmp/dwa_optimizer_checkpoint.pkl"
g_optimizer_result = None

# --- Настройки для вашего проекта mrvk ---
# Начальная поза робота (берется из launch-файла при каждом запуске)
# Нужна только для цели "вернуться на старт"
INIT_POSE_X = -3.977 
INIT_POSE_Y = 10.693

# Штрафы
WEIGHT_TIME = 1.0       # Штраф за каждую секунду движения
WEIGHT_FAILURE = 1000.0 # Огромный штраф за любой провал (столкновение, таймаут)

# Цели для миссии (x, y, yaw в радианах)
MISSION_GOALS = [
    (6.019, 2.099, -0.187),
    (0.031, -3.926, 0.000),
    (INIT_POSE_X, INIT_POSE_Y, 0.0) 
]

# Имена параметров (должны точно соответствовать порядку в `space`)
PARAM_NAMES = [
    'alpha', 'beta', 'gamma', 'predict_time', 'vx_samples', 'vth_samples',
    'max_vel_x', 'min_vel_x', 'max_vel_th', 'acc_lim_x', 'acc_lim_th'
]


# ======================= КЛАСС ДЛЯ ДЕТЕКЦИИ СТОЛКНОВЕНИЙ =======================
class CollisionDetector:
    """Прослушивает топики бамперов и выставляет флаг при столкновении."""
    def __init__(self):
        self.collision_detected = False
        self.subscribers = []
        
    def start(self):
        """Активирует подписчиков в начале итерации."""
        self.reset()
        bumper_topics = ["/bumper_front_state", "/bumper_rear_state"]
        self.subscribers = []
        for topic in bumper_topics:
            sub = rospy.Subscriber(topic, ContactsState, self.collision_callback, queue_size=1)
            self.subscribers.append(sub)
        rospy.loginfo(f"CollisionDetector активирован, слушает {len(self.subscribers)} топиков.")

    def stop(self):
        """Отключает подписчиков в конце итерации."""
        for sub in self.subscribers:
            sub.unregister()
        self.subscribers = []
        rospy.loginfo("CollisionDetector деактивирован.")
    
    def collision_callback(self, msg):
        if msg.states:
            if not self.collision_detected:
                rospy.logwarn(">>> СТОЛКНОВЕНИЕ ОБНАРУЖЕНО! Миссия провалена. <<<")
            self.collision_detected = True
            
    def reset(self):
        self.collision_detected = False

    def has_collided(self):
        return self.collision_detected

g_collision_detector = CollisionDetector()
# ==============================================================================


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Сохранение, Обработка Ctrl+C) ---
def save_checkpoint(res):
    global g_optimizer_result
    g_optimizer_result = res
    try:
        with open(CHECKPOINT_PATH, "wb") as f:
            dump(res, f)
    except Exception as e:
        rospy.logwarn(f"Не удалось сохранить чекпоинт: {e}")

def signal_handler(sig, frame):
    """Обрабатывает Ctrl+C для вывода промежуточных результатов."""
    print("\n" + "="*50)
    rospy.logwarn(">>> Оптимизация прервана! Отображение лучших найденных результатов. <<<")
    if g_optimizer_result:
        print(f"Минимальная стоимость (J) из {len(g_optimizer_result.x_iters)} итераций: {g_optimizer_result.fun:.4f}")
        optimal_params = dict(zip(PARAM_NAMES, g_optimizer_result.x))
        print("\nОптимальные параметры, найденные на данный момент:")
        for name, val in optimal_params.items():
            print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")
    else:
        print("Итерации не завершены.")
    print("="*50)
    sys.exit(0)


# ==============================================================================
# --- ЦЕЛЕВАЯ ФУНКЦИЯ ДЛЯ ОПТИМИЗАЦИИ (с "холодным" перезапуском) ---
# ==============================================================================
def objective_function(params):
    # 1. Распаковка и установка параметров
    params_to_set = dict(zip(PARAM_NAMES, params))
    
    log_params_str = ", ".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in params_to_set.items()])
    rospy.loginfo(f"--- Новая итерация. Тестирование параметров: {log_params_str} ---")

    try:
        for name, val in params_to_set.items():
            param_path = f"/move_base/ImprovedDWALocalPlanner/{name}"
            # Убеждаемся, что тип правильный
            target_type = int if 'samples' in name else float
            rospy.set_param(param_path, target_type(val))
    except Exception as e:
        rospy.logerr(f"Критическая ошибка: не удалось установить ROS параметр. Ошибка: {e}")
        return 99999.0

    # 2. ПОЛНЫЙ ПЕРЕЗАПУСК ВСЕГО LAUNCH-ФАЙЛА
    # !!! УКАЖИТЕ ЗДЕСЬ ИМЯ ВАШЕГО ГЛАВНОГО LAUNCH-ФАЙЛА !!!
    launch_file_name = "test.launch" 
    launch_package_name = "mrvk_gazebo"
    
    launch_command = ["roslaunch", launch_package_name, launch_file_name]
    rospy.loginfo(f"Запуск полного окружения: {launch_command}")
    
    launch_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Выполнение миссии
    move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    
    if not move_base_client.wait_for_server(rospy.Duration(120.0)):
        rospy.logerr("Система не запустилась за 60 секунд! Проверьте launch-файл.")
        try:
            launch_process.terminate(); launch_process.wait(timeout=5.0)
        except:
            pass
        return 99999.0

    g_collision_detector.start()
        
    start_time = rospy.Time.now()
    mission_successful = True
    
    for i, goal_coords in enumerate(MISSION_GOALS):
        if rospy.is_shutdown():
            mission_successful = False
            break
            
        rospy.loginfo(f"Отправка цели {i+1}/{len(MISSION_GOALS)}: {goal_coords}")
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = goal_coords[0]
        goal.target_pose.pose.position.y = goal_coords[1]
        q = tf.transformations.quaternion_from_euler(0, 0, goal_coords[2])
        goal.target_pose.pose.orientation = Quaternion(*q)

        move_base_client.send_goal(goal)
        
        goal_timeout = rospy.Duration(90.0)
        wait_start_time = rospy.Time.now()
        
        while (rospy.Time.now() - wait_start_time) < goal_timeout:
            if g_collision_detector.has_collided():
                mission_successful = False
                break
            
            current_state = move_base_client.get_state()
            if current_state in [actionlib.GoalStatus.SUCCEEDED, actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
                break
                
            rospy.sleep(0.2)
        
        if not mission_successful:
            move_base_client.cancel_all_goals()
            break

        final_state = move_base_client.get_state()
        if final_state != actionlib.GoalStatus.SUCCEEDED:
            rospy.logwarn(f"Цель {i+1} провалена. Статус: {final_state}")
            mission_successful = False
            move_base_client.cancel_all_goals()
            break
        rospy.loginfo(f"Цель {i+1} успешно достигнута.")
            
    end_time = rospy.Time.now()
    total_mission_time = (end_time - start_time).to_sec()

    # 4. Остановка LAUNCH-ФАЙЛА
    rospy.loginfo("Завершение полного окружения...")
    launch_process.send_signal(signal.SIGINT)
    try:
        launch_process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        rospy.logwarn("Процесс навигации не завершился штатно, принудительное завершение.")
        launch_process.kill()
        launch_process.wait()
    
    g_collision_detector.stop()

    # 5. Расчет стоимости
    failure_penalty = WEIGHT_FAILURE if not mission_successful else 0
    time_penalty = WEIGHT_TIME * total_mission_time
    cost = time_penalty + failure_penalty
    
    rospy.loginfo(f"--- Итерация завершена. Время: {total_mission_time:.2f}s, Провал: {not mission_successful}. ===> Стоимость: {cost:.2f} ---")
    rospy.sleep(5.0)
    return cost

# ==============================================================================
# --- ТОЧКА ВХОДА В СКРИПТ (MAIN) ---
# ==============================================================================
if __name__ == '__main__':
    try:
        rospy.init_node('dwa_optimizer', anonymous=True, log_level=rospy.INFO)
        signal.signal(signal.SIGINT, signal_handler)
        
        space = [
            Real(0.1, 5.0, name='alpha'),
            Real(1.0, 50.0, name='beta'),
            Real(0.05, 2.0, name='gamma'),
            Real(1.0, 4.0, name='predict_time'),
            Integer(10, 30, name='vx_samples'),
            Integer(20, 50, name='vth_samples'),
            Real(0.3, 1.5, name='max_vel_x'),
            Real(-0.3, -0.05, name='min_vel_x'),
            Real(0.5, 3.14, name='max_vel_th'),
            Real(1.0, 3.5, name='acc_lim_x'),
            Real(1.5, 5.0, name='acc_lim_th')
        ]
        
        N_CALLS = 100 
        x0, y0 = None, None
        
        try:
            res_loaded = load(CHECKPOINT_PATH)
            x0 = res_loaded.x_iters
            y0 = res_loaded.func_vals
            n_completed = len(x0)
            N_CALLS = max(0, N_CALLS - n_completed)
            rospy.loginfo(f"Возобновление с чекпоинта. Выполнено {n_completed} итераций. Осталось: {N_CALLS}")
            g_optimizer_result = res_loaded
        except (FileNotFoundError, EOFError, ValueError):
            rospy.loginfo("Чекпоинт не найден. Начинаем новую оптимизацию.")
        
        rospy.loginfo("Оптимизатор готов к запуску. Убедитесь, что roscore запущен.")
        rospy.loginfo("Нажмите Enter для начала...")
        input()

        if N_CALLS > 0:
            rospy.loginfo(f"Запуск Байесовской оптимизации на {N_CALLS} новых итераций...")
            res = gp_minimize(
                objective_function, space, x0=x0, y0=y0,
                n_calls=N_CALLS, n_random_starts=min(5, N_CALLS),
                callback=[save_checkpoint], verbose=True, random_state=123
            )
            g_optimizer_result = res
            
        if g_optimizer_result:
            print("\n" + "="*50)
            print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
            print(f"Минимальная стоимость (J): {g_optimizer_result.fun:.4f}")
            optimal_params = dict(zip(PARAM_NAMES, g_optimizer_result.x))
            print("\nОптимальные параметры найдены:")
            for name, val in optimal_params.items():
                print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")
            print("="*50)
        else:
            print("Оптимизация не была выполнена или прервана до первой итерации.")

    except rospy.ROSInterruptException:
        rospy.loginfo("Оптимизация прервана (ROS shutdown).")
    except Exception as e:
        rospy.logfatal(f"Непредвиденная ошибка в главном цикле: {e}")