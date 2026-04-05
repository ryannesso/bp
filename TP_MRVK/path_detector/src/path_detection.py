#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import joblib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from path_detector.msg import DetectedPath

class MLPathDetectorNode:
    def __init__(self):
        rospy.init_node("ml_path_detector", anonymous=True)
        self.bridge = CvBridge()
        
        model_path = "/home/yehor/work_disk/catkin_ws/src/TP_MRVK/path_detector/src/road_detector_rf.pkl"
        try:
            self.rf_model = joblib.load(model_path)
        except Exception as e:
            rospy.logerr(f"Model load error: {e}")
            exit(1)
        
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1)
        self.path_pub = rospy.Publisher("/shoddy/detected_path", DetectedPath, queue_size=1)
        self.vis_pub = rospy.Publisher("/path_detector/debug_image", Image, queue_size=1)
        self.mask_pub = rospy.Publisher("/path_detector/mask", Image, queue_size=1)
        
        rospy.loginfo("ML Path Detector initialized.")

    def extract_features(self, img):
        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        texture = cv2.magnitude(sobel_x, sobel_y)
        texture = cv2.GaussianBlur(texture, (5, 5), 0)
        y_coord = np.tile(np.linspace(0, 1, h), (w, 1)).T
        feature_map = np.dstack([lab, hsv, texture, y_coord])
        return feature_map.reshape(-1, feature_map.shape[2])

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return
            
        orig_h, orig_w = img.shape[:2]
        small_img = cv2.resize(img, (160, 120))
        
        # 1. Инференс модели
        features = self.extract_features(small_img)
        predictions = self.rf_model.predict(features)
        
        # 2. Маска (1 - дорога, 0 - всё остальное)
        ml_mask = predictions.reshape((120, 160)).astype(np.uint8) * 255
        
        # 3. Морфология
        kernel = np.ones((5,5), np.uint8)
        ml_mask = cv2.morphologyEx(ml_mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. Ресайз до оригинала
        final_mask = cv2.resize(ml_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # 5. ДЛЯ C++ ПЛАГИНА: 
        # C++ плагин ждет 255 там, где надо поставить препятствие.
        # Значит, мы отправляем инвертированную маску (дорога = 0, всё остальное = 255)
        mask_for_cpp = cv2.bitwise_not(final_mask)
        
        dp = DetectedPath()
        dp.height = orig_h
        dp.width = orig_w
        dp.frame = mask_for_cpp.flatten().tolist()
        self.path_pub.publish(dp)

        # 6. ВИЗУАЛИЗАЦИЯ (Опционально)
        # --- ВИЗУАЛИЗАЦИЯ (Без использования cv2.addWeighted) ---
        vis_img = img.copy()
        
        # Инвертируем маску, чтобы найти НЕ-дорогу (препятствия)
        # final_mask: 255 = дорога, 0 = препятствие
        not_road = (final_mask == 0)
        
        # Просто красим все "не-дорожные" пиксели в красный цвет
        # [0, 0, 255] в BGR это красный
        vis_img[not_road] = [0, 0, 255]
        
        # Публикуем ЧБ маску (дорога = 255, остальное = 0)
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(final_mask, "mono8"))
        
        # Публикуем для визуализации
        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
        
        # Окно отладки (если нужно)
        # cv2.imshow("ML Detection Debug", vis_img)
        # cv2.waitKey(1)

if __name__ == '__main__':
    try:
        MLPathDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass