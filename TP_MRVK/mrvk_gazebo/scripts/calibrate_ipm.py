#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class CalibrationNode:
    def __init__(self):
        rospy.init_node('ipm_calibrator', anonymous=True)
        
        # ВАЖНО: Проверь название топика! Может быть /camera/rgb/image_raw
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.bridge = CvBridge()
        self.image = None
        self.points = []

    def callback(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                print(f"Точка добавлена: [{x}, {y}]")
                self.points.append([x, y])
            else:
                print("Уже выбрано 4 точки. Нажмите 'c' для сброса или 'q' для выхода.")

    def run(self):
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)

        print("ИНСТРУКЦИЯ:")
        print("1. Кликните 4 точки на изображении, которые образуют ПРЯМОУГОЛЬНИК на земле (трапеция на экране).")
        print("   Порядок: Нижний-Левый -> Нижний-Правый -> Верхний-Правый -> Верхний-Левый")
        print("2. Нажмите 'q' чтобы получить код для вставки.")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.image is not None:
                display_img = self.image.copy()
                
                # Рисуем точки и линии
                for p in self.points:
                    cv2.circle(display_img, tuple(p), 5, (0, 0, 255), -1)
                
                if len(self.points) == 4:
                    pts = np.array(self.points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)

                cv2.imshow("Calibration", display_img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') and len(self.points) == 4:
                break
            if key & 0xFF == ord('c'):
                self.points = []
                print("Точки сброшены.")

            rate.sleep()

        cv2.destroyAllWindows()
        
        if len(self.points) == 4:
            print("\n" + "="*40)
            print("СКОПИРУЙТЕ ЭТИ КООРДИНАТЫ:")
            print(f"src_points = np.float32({self.points})")
            print("="*40 + "\n")

if __name__ == '__main__':
    try:
        node = CalibrationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass