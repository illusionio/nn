# detection_engine.py
# 功能：封装 YOLOv8 模型的加载与推理逻辑，提供干净的检测接口
# 修改：增加距离估计和危险等级显示，增加目标跟踪（分配稳定 ID）

from ultralytics import YOLO
import io
import sys
import os
import cv2
import numpy as np   # 新增导入，用于数据处理


class ModelLoadError(Exception):
    """模型加载失败专用异常"""
    pass


class DetectionEngine:
    """
    目标检测引擎类。
    负责加载 YOLO 模型并对输入图像帧执行推理，
    同时屏蔽模型内部的冗余打印输出（如进度条、日志等），
    使主程序输出更整洁。
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25):
        """
        初始化检测引擎。

        参数:
            model_path (str): YOLO 模型文件路径或名称（如 'yolov8n.pt'）
            conf_threshold (float): 置信度阈值，低于此值的检测结果将被过滤
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # 加载 YOLO 模型（在初始化时完成，避免每次检测重复加载）
        self.model = self._load_model()
        self.tracker = None          # 懒加载，第一次调用 detect 时创建
        self.enable_tracking = True  # 可通过配置控制是否启用跟踪

    def _load_model(self):
        """
        私有方法：加载 YOLO 模型，并抑制其标准输出和错误输出。
        
        原因：YOLO 在加载模型或首次推理时会自动打印信息（如设备、尺寸等），
        这些信息在 GUI 或自动化脚本中属于干扰。通过临时重定向 stdout/stderr 来静默加载。
        
        返回:
            YOLO: 已加载的模型实例
        """
        # 保存原始的标准输出和错误流
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # 临时重定向到 StringIO 缓冲区（丢弃所有输出）
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            # 判断是本地文件还是官方模型名
            if os.path.isfile(self.model_path):
                model = YOLO(self.model_path)
            else:
                # 尝试作为 Ultralytics 官方模型加载（会自动下载）
                model = YOLO(self.model_path)
            return model
        except FileNotFoundError as e:
            raise ModelLoadError(f"Model file not found: {self.model_path}") from e
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg:
                raise ModelLoadError(
                    "GPU memory insufficient. Try using CPU or a smaller model (e.g., yolov8n.pt)."
                ) from e
            elif "AssertionError" in msg and ("model" in msg or "state_dict" in msg):
                raise ModelLoadError(f"Corrupted or incompatible model weights: {self.model_path}") from e
            else:
                raise ModelLoadError(f"Runtime error during model loading: {msg}") from e
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading YOLO model: {e}") from e
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _estimate_distance(self, box_height, known_height=1.6, focal_length=700):
        """根据检测框高度估算距离（米）"""
        if box_height < 1:
            return 999.9
        return (known_height * focal_length) / box_height

    def _get_danger_level(self, distance):
        """根据距离判定危险等级"""
        if distance < 10:
            return "DANGER"
        elif distance < 20:
            return "WARNING"
        else:
            return "SAFE"

    def detect(self, frame):
        """
        对单帧图像执行目标检测。

        参数:
            frame (np.ndarray): 输入图像，格式为 HWC（高度, 宽度, 通道），BGR 或 RGB 均可（YOLO 内部会处理）

        返回:
            tuple:
                - annotated_frame (np.ndarray): 带有检测框、标签和置信度的可视化图像（HWC, BGR 格式）
                - results (List[ultralytics.engine.results.Results]): 原始检测结果对象列表（通常长度为1）
        """
        # 保存原始输出流
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # 临时静音 YOLO 的 verbose 输出（如 "image 1/1 ..."）
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            # 执行推理：传入图像、置信度阈值，并关闭详细日志（verbose=False）
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            annotated_frame = results[0].plot()   # 获取 YOLO 默认标注图

            # ---------- 新增：距离估计、危险等级 + 目标跟踪 ----------
            if results[0].boxes is not None:
                raw_boxes = results[0].boxes
                
                # 提取原始检测数据（用于跟踪器）
                boxes_xyxy = raw_boxes.xyxy.cpu().numpy()   # (N, 4)
                confs = raw_boxes.conf.cpu().numpy()        # (N,)
                clses = raw_boxes.cls.cpu().numpy()         # (N,)
                
                # 构建检测列表 [(x1,y1,x2,y2,conf,cls), ...]
                detections = []
                for i in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    detections.append((x1, y1, x2, y2, confs[i], int(clses[i])))
                
                # 跟踪器更新，获得带 ID 的检测列表（每个元素多一个 id）
                if self.enable_tracking:
                    if self.tracker is None:
                        self.tracker = SimpleTracker()
                    tracked_dets = self.tracker.update(detections)   # (x1,y1,x2,y2,conf,cls,id)
                else:
                    tracked_dets = [d + (None,) for d in detections]  # 无 ID
                
                # --- 原有距离估计和危险等级绘制（保持不变）---
                for box in raw_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_height = y2 - y1
                    distance = self._estimate_distance(box_height)
                    danger = self._get_danger_level(distance)
                    info_text = f"{danger} {distance:.1f}m"
                    cv2.putText(annotated_frame, info_text, (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # --- 新增：在图上绘制跟踪 ID（叠加在 YOLO 默认标注上）---
                for det in tracked_dets:
                    x1, y1, x2, y2, conf, cls, obj_id = det
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    if obj_id is not None:
                        id_text = f"ID:{obj_id}"
                        # 放在距离文字上方（y1 - 25 避免重叠）
                        cv2.putText(annotated_frame, id_text, (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            # -------------------------------------------------

            return annotated_frame, results
        except Exception as e:
            # 不抛出异常，而是返回原图以维持流程
            print(f"⚠️ Warning: Detection failed on current frame: {e}")
            return frame.copy(), []
        finally:
            # 恢复标准输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class SimpleTracker:
    """基于 IoU 匹配的简易目标跟踪器，为连续帧中检测到的物体分配稳定 ID。"""
    
    def __init__(self, iou_threshold=0.3, max_lost=5):
        """
        参数：
            iou_threshold: 判断是否为同一目标的最小 IoU
            max_lost: 目标消失多少帧后删除轨迹
        """
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1                # 下一个可用的 ID
        self.tracks = []                # 每个轨迹为 {'id': int, 'box': (x1,y1,x2,y2), 'lost_count': int}

    def update(self, detections):
        """
        输入：detections = [(x1, y1, x2, y2, conf, cls), ...]  (list of tuples)
        输出：带 ID 的检测列表 = [(x1, y1, x2, y2, conf, cls, id), ...]
        """
        if not detections:
            # 没有检测，所有轨迹 lost_count +1
            for t in self.tracks:
                t['lost_count'] += 1
            self._remove_lost_tracks()
            return []

        # 计算 IoU 矩阵
        iou_matrix = []
        for trk in self.tracks:
            trk_box = trk['box']
            row = []
            for det in detections:
                det_box = det[:4]
                row.append(self._box_iou(trk_box, det_box))
            iou_matrix.append(row)

        # 贪心匹配：按 IoU 降序分配
        matched_track_idx = []
        matched_det_idx = []
        used_track = set()
        used_det = set()

        # 生成所有 (track_idx, det_idx) 对并按 IoU 排序
        pairs = [(i, j) for i in range(len(self.tracks)) for j in range(len(detections))]
        pairs.sort(key=lambda x: iou_matrix[x[0]][x[1]], reverse=True)

        for track_idx, det_idx in pairs:
            if track_idx in used_track or det_idx in used_det:
                continue
            if iou_matrix[track_idx][det_idx] >= self.iou_threshold:
                matched_track_idx.append(track_idx)
                matched_det_idx.append(det_idx)
                used_track.add(track_idx)
                used_det.add(det_idx)

        # 更新匹配到的轨迹
        for trk_idx, det_idx in zip(matched_track_idx, matched_det_idx):
            self.tracks[trk_idx]['box'] = detections[det_idx][:4]   # 更新位置
            self.tracks[trk_idx]['lost_count'] = 0

        # 未匹配的检测 → 新轨迹
        for j, det in enumerate(detections):
            if j not in used_det:
                self.tracks.append({
                    'id': self.next_id,
                    'box': det[:4],
                    'lost_count': 0
                })
                self.next_id += 1

        # 未匹配的轨迹 lost_count +1
        for i, trk in enumerate(self.tracks):
            if i not in used_track:
                trk['lost_count'] += 1

        # 移除超期轨迹
        self._remove_lost_tracks()

        # 组装输出：每个检测（包含新匹配的和新创建的）都带上对应的 ID
        output = []
        # 先处理匹配上的（保证 ID 正确）
        for trk_idx, det_idx in zip(matched_track_idx, matched_det_idx):
            det = detections[det_idx]
            trk_id = self.tracks[trk_idx]['id']
            output.append(det + (trk_id,))
        # 再处理新轨迹
        for j, det in enumerate(detections):
            if j not in used_det:
                # 新轨迹的 ID 就是最近分配的 next_id-1
                new_id = self.next_id - 1
                output.append(det + (new_id,))
        return output

    def _remove_lost_tracks(self):
        """删除丢失超过 max_lost 帧的轨迹"""
        self.tracks = [t for t in self.tracks if t['lost_count'] <= self.max_lost]

    @staticmethod
    def _box_iou(box1, box2):
        """计算两个边界框的 IoU，box 格式 (x1, y1, x2, y2)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0