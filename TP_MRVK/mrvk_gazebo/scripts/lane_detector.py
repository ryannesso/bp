#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

class LaneToObstacles:
    def __init__(self):
        rospy.init_node('vision_obstacles', anonymous=True)
        
        # Топик камеры (убедись что совпадает)
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        # Топик, куда будем слать "виртуальные препятствия"
        self.pc_pub = rospy.Publisher("/vision/obstacles", PointCloud2, queue_size=1)
        
        self.bridge = CvBridge()
        
        # === 1. КАЛИБРОВКА (Твои значения) ===
        self.src_points = np.float32([[5, 308], [797, 330], [443, 166], [346, 166]])
        
        # Параметры IPM (превращаем в вид сверху 800x600)
        self.IMG_W, self.IMG_H = 800, 600
        self.dst_points = np.float32([
            [200, self.IMG_H],   
            [600, self.IMG_H],   
            [600, 0],            
            [200, 0]             
        ])
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        # === 2. МАСШТАБИРОВАНИЕ (Внимательно тут!) ===
        # Нам нужно перевести ПИКСЕЛИ в МЕТРЫ.
        # PX_PER_METER_X: Если на картинке ширина дороги 400px (от 200 до 600),
        # а в реальности ширина тротуара около 2.5 метра, то 400/2.5 = 160 px/m.
        # Если в Rviz точки будут слишком широко — увеличивай это число.
        self.PX_PER_METER_X = 160.0  
        self.PX_PER_METER_Y = 160.0  
        
        # Сдвиг центра картинки относительно робота
        self.ROBOT_OFFSET_X_PX = 400.0 # Центр картинки по ширине

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.process(cv_image)
        except CvBridgeError as e:
            print(e)

    def process(self, img):
        # 1. IPM - Вид сверху
        warped = cv2.warpPerspective(img, self.M, (self.IMG_W, self.IMG_H))
        
        # 2. ФИЛЬТРАЦИЯ (ТВОИ ПАРАМЕТРЫ)
        hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]
        
        # Условия из твоих ползунков:
        # S < 29  И  51 < L < 103
        mask = np.zeros_like(s_channel)
        condition = (s_channel < 29) & (l_channel > 51) & (l_channel < 103)
        mask[condition] = 255
        
        # Убираем шум (дырки в асфальте и мелкие точки травы)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Убрать шум
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Закрыть дыры

        # 3. ПОИСК ГРАНИЦ (Это будут наши стены)
        edges = cv2.Canny(mask, 100, 200)
        
        # Получаем координаты белых точек
        y_idxs, x_idxs = np.where(edges > 0)
        
        if len(x_idxs) > 0:
            # 4. КОНВЕРТАЦИЯ В МЕТРЫ
            # X робота = Вперед по картинке (инвертируем Y пиксели, т.к. 0 сверху)
            x_ros = (self.IMG_H - y_idxs) / self.PX_PER_METER_Y
            
            # Y робота = Влево по картинке (Центр картинки - X пиксель)
            y_ros = (self.ROBOT_OFFSET_X_PX - x_idxs) / self.PX_PER_METER_X
            
            # Z = 0
            z_ros = np.zeros_like(x_ros) + 0.4
            
            # Упаковка в PointCloud2
            points = np.column_stack((x_ros, y_ros, z_ros))
            
            header = Header()
            header.stamp = rospy.Time.now()
            # ВАЖНО: Точки даем относительно базы робота
            header.frame_id = "base_link" 
            
            pc_msg = pc2.create_cloud_xyz32(header, points)
            self.pc_pub.publish(pc_msg)

        # Отладка: покажи края (нажми q в окне чтобы закрыть, если мешает)
        cv2.imshow("Lane Edges", edges) 
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        LaneToObstacles()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass