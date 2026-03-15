#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import OccupancyGrid
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import os

class RoadSegmentationNode:
    def __init__(self):
        rospy.init_node('road_segmentation_node')
        
        # Параметры
        self.model_path = rospy.get_param('~model_path', '')
        self.orig_w, self.orig_h = 640, 480 # Стандартное разрешение камеры в Gazebo
        self.target_w, self.target_h = 320, 240
        self.grid_res = 0.05
        self.world_w, self.world_h = 4.0, 5.0 # 4м ширина, 5м вперед
        
        self.bridge = CvBridge()
        
        # Инициализация ONNX
        if os.path.exists(self.model_path):
            rospy.loginfo(f"Loading model from {self.model_path}")
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape 
            
            output_info = self.session.get_outputs()
            self.output_names = [o.name for o in output_info]
            
            if self.input_shape[1] == 3: # NCHW
                self.target_h, self.target_w = self.input_shape[2], self.input_shape[3]
                self.is_nchw = True
            else: # NHWC
                self.target_h, self.target_w = self.input_shape[1], self.input_shape[2]
                self.is_nchw = False
            rospy.loginfo(f"Model: {self.target_w}x{self.target_h}, Format: {'NCHW' if self.is_nchw else 'NHWC'}, Outputs: {self.output_names}")
        else:
            rospy.logerr(f"Model file not found! Dummy mode enabled.")
            self.session = None
            self.is_nchw = True

        # Паблишеры
        self.grid_pub = rospy.Publisher('/path_detector/road_grid', OccupancyGrid, queue_size=1)
        self.pc_pub = rospy.Publisher('/path_detector/grass_cloud', PointCloud2, queue_size=1)
        self.mask_pub = rospy.Publisher('/road_segmentation/debug_mask', Image, queue_size=1)
        
        # Подписчики
        self.img_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback, queue_size=1)
        
        # Точки для IPM (Трапеция -> Прямоугольник)
        # ВАЖНО: Масштабируем точки под разрешение нейросети (target_w/h)
        scale_x = self.target_w / 320.0
        scale_y = self.target_h / 240.0
        
        self.src_pts = np.float32([
            [100 * scale_x, 150 * scale_y], [220 * scale_x, 150 * scale_y],
            [320 * scale_x, 240 * scale_y], [0 * scale_x, 240 * scale_y]
        ])
        
        self.bev_w = int(self.world_w / self.grid_res)
        self.bev_h = int(self.world_h / self.grid_res)
        
        self.dst_pts = np.float32([
            [0, 0], [self.bev_w, 0],
            [self.bev_w, self.bev_h], [0, self.bev_h]
        ])
        
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        
        rospy.loginfo("Road Segmentation Node Ready.")

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        # 1. Предобработка
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(img_rgb, (self.target_w, self.target_h))
        
        # Эвристика: если это HybridNets, то он требует нормализацию ImageNet
        is_hybridnets = "segmentation" in self.output_names
        if is_hybridnets:
            input_img = input_img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            input_img = (input_img - mean) / std
        else:
            input_img = input_img.astype(np.float32)
        
        if self.is_nchw:
            input_tensor = np.transpose(input_img, (2, 0, 1)) # HWC -> CHW
            input_tensor = np.expand_dims(input_tensor, axis=0) # CHW -> NCHW
        else:
            input_tensor = np.expand_dims(input_img, axis=0) # HWC -> NHWC

        input_tensor = input_tensor.astype(np.float32)

        # 2. Инференс
        if self.session:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            is_hybridnets = "segmentation" in self.output_names
            if is_hybridnets:
                seg_idx = self.output_names.index("segmentation")
                seg_out = outputs[seg_idx] # (1, C, H, W)
                raw_out = seg_out[0]
                mask_indices = np.squeeze(np.argmax(seg_out, axis=1))
                mask = np.where((mask_indices == 1) | (mask_indices == 2), 1, 0).astype(np.uint8)
            else:
                raw_out = outputs[0][0] # Убираем Batch
                
                # Определяем формат выхода [C, H, W] или [H, W, C]
                if len(raw_out.shape) == 3:
                    if raw_out.shape[0] < 10:
                        mask_indices = np.argmax(raw_out, axis=0)
                    elif raw_out.shape[2] < 10:
                        mask_indices = np.argmax(raw_out, axis=-1)
                    else:
                        mask_indices = (raw_out[0] > 0.0).astype(np.uint8) 
                else:
                    mask_indices = (raw_out > 0.0).astype(np.uint8)

                # В репозитории: Road = 2, Grass = 0.
                mask = np.where((mask_indices == 2) | (mask_indices == 1), 1, 0).astype(np.uint8)
            
            # Отладочный лог раз в 30 кадров
            if getattr(self, 'frame_count', 0) % 30 == 0:
                rospy.loginfo(f"Output shape: {raw_out.shape}, indices unique: {np.unique(mask_indices)}")
            self.frame_count = getattr(self, 'frame_count', 0) + 1
        else:
            # Dummy mode: просто белый прямоугольник посередине (имитация дороги)
            mask = np.zeros((self.target_h, self.target_w), dtype=np.uint8)
            cv2.rectangle(mask, (100, 100), (220, 240), 1, -1)

        # 3. IPM (Warp to Bird's Eye View)
        # mask сейчас: 1 = дорога, 0 = трава
        mask_8u = (mask * 255).astype(np.uint8)
        bev_mask = cv2.warpPerspective(mask_8u, self.M, (self.bev_w, self.bev_h))
        
        # 4. Публикация Grid
        self.publish_grid(bev_mask, msg.header.stamp)
        
        # 5. Публикация PointCloud2 (Трава)
        self.publish_pc(bev_mask, msg.header.stamp)
        
        # 6. Визуализация (увеличиваем окна для ПЛОТНОЙ видимости)
        viz_w, viz_h = 800, 600
        overlay = cv2.resize(cv_img, (viz_w, viz_h))
        mask_viz = cv2.resize(mask_8u, (viz_w, viz_h))
        
        road_overlay = overlay.copy()
        road_overlay[mask_viz > 127] = [0, 255, 0] # Green
        grass_overlay = overlay.copy()
        grass_overlay[mask_viz <= 127] = [0, 0, 255] # Red
        
        cv2.addWeighted(road_overlay, 0.4, overlay, 0.6, 0, overlay)
        cv2.addWeighted(grass_overlay, 0.2, overlay, 0.8, 0, overlay)
        cv2.imshow("1. Prediction Overlay (Green=Road)", overlay)
        
        # 2. Raw Mask (Upscaled)
        cv2.imshow("2. Raw Mask (White=Road)", mask_viz)
        
        # 3. Bird's Eye View
        # Увеличим BEV для лучшей видимости
        bev_big = cv2.resize(bev_mask, (400, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("3. Bird's Eye View (Navigation)", bev_big)
        
        cv2.waitKey(1)
        
        # Раз в 30 кадров логируем статистику, чтобы понять почему всё зеленое
        if getattr(self, 'frame_count', 0) % 30 == 0:
            rospy.loginfo(f"Output stats: min={np.min(raw_out):.2f}, max={np.max(raw_out):.2f}")
        
        if self.mask_pub.get_num_connections() > 0:
            self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_8u, "mono8"))

    def publish_grid(self, bev_mask, stamp):
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = "base_link"
        grid.info.resolution = self.grid_res
        
        # В ROS OccupancyGrid:
        # width - количество ячеек по оси X (вперед)
        # height - количество ячеек по оси Y (вбок)
        # В BEV картинке: i (строки) - это X, j (колонки) - это Y
        
        grid.info.width = self.bev_h
        grid.info.height = self.bev_w
        
        # Центрируем по Y: робот стоит в (X=0, Y=0). 
        # Сетка начинается от X=0 и идет вперед.
        # По Y она начинается от -world_w/2.
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = -self.world_w / 2.0
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        
        # Готовим данные
        # bev_mask[i, j]: i=0 (далеко), i=max (близко). j=0 (лево), j=max (право)
        # Нам нужно: X=0 (близко) в начало, Y=min (право) в начало.
        
        # 1. Поворачиваем маску так, чтобы строки были по X, а колонки по Y
        # Сейчас i - это X, но i=0 это "далеко". 
        # Инвертируем i, чтобы i=0 было "близко" (у ног робота)
        flipped_mask = np.flipud(bev_mask).copy()
        
        # Игнорируем слепую зону (1 метр = 20 ячеек)
        # Принудительно ставим "белый" (свободно) перед роботом
        cells_to_clear = int(1.0 / self.grid_res)
        flipped_mask[:cells_to_clear, :] = 255
        
        # 2. Инвертируем j, чтобы j=0 было "право" (Y = -world_w/2)
        flipped_mask = np.fliplr(flipped_mask)
        
        # 3. Транспонируем, чтобы получить [X, Y] (в ROS OccupancyGrid [y*width + x])
        # На самом деле нам нужно [Y][X], где X идет первым в памяти.
        # В ROS: grid.data[y * width + x]
        final_grid = np.where(flipped_mask.T > 127, 0, 100).astype(np.int8)
        
        grid.data = final_grid.flatten().tolist()
        self.grid_pub.publish(grid)

    def publish_pc(self, bev_mask, stamp):
        points = []
        skip = 2 
        # Координаты X (вперед) и Y (влево)
        for i in range(self.bev_h):
            if i % skip != 0: continue
            for j in range(self.bev_w):
                if j % skip != 0: continue
                
                if bev_mask[i, j] < 127: # Grass
                    # i=0 (далеко) -> x = world_h
                    # i=max (близко) -> x = 0
                    x = (self.bev_h - 1 - i) * self.grid_res
                    
                    # Игнорируем слепую зону: не ставим препятствия ближе чем 1.0 метра
                    # (потому что нейросеть часто не дорисовывает низ картинки)
                    if x < 1.0:
                        continue
                        
                    # j=0 (лево) -> y = +world_w/2
                    # j=max (право) -> y = -world_w/2
                    y = self.world_w / 2.0 - j * self.grid_res
                    points.append([x, y, 0.0])
        
        header = Header(stamp=stamp, frame_id="base_link")
        cloud = point_cloud2.create_cloud_xyz32(header, points)
        self.pc_pub.publish(cloud)

if __name__ == '__main__':
    try:
        node = RoadSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
