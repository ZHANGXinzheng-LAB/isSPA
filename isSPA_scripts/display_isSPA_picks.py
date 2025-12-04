#!/usr/bin/env python3

import sys
import os
import re
import mrcfile
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QLineEdit, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage

class StarFileViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("isSPA display")
        self.setGeometry(50, 50, 630, 630)
        
        # 数据存储
        self.micrographs = defaultdict(list)  # 微图形名称到颗粒列表的映射
        self.micrograph_names = []  # 微图形名称的有序列表
        self.current_index = 0  # 当前显示的微图形索引
        self.circle_diameter = 200  # 默认圆圈直径
        self.pixmap = QPixmap()
        self.scale = 0.0
        
        # 初始化UI
        self.init_ui()
        
    def init_ui(self):
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 打开文件按钮
        self.open_button = QPushButton("Import STAR file")
        self.open_button.clicked.connect(self.open_file)
        control_layout.addWidget(self.open_button)
        
        # 圆圈直径控制
        control_layout.addWidget(QLabel("Particle diameter:"))
        self.diameter_edit = QLineEdit(str(self.circle_diameter))
        self.diameter_edit.setMaximumWidth(50)
        self.diameter_edit.returnPressed.connect(self.update_diameter)
        control_layout.addWidget(self.diameter_edit)
        
        '''
        self.increase_button = QPushButton("增大")
        self.increase_button.clicked.connect(self.increase_diameter)
        control_layout.addWidget(self.increase_button)
        
        self.decrease_button = QPushButton("减小")
        self.decrease_button.clicked.connect(self.decrease_diameter)
        control_layout.addWidget(self.decrease_button)
        '''
        
        # 导航按钮
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_image)
        control_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_image)
        control_layout.addWidget(self.next_button)
        
        main_layout.addLayout(control_layout)
        
        # 阈值滑块
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Score:"))
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 1000)  # 假设得分范围是0-10，放大100倍
        self.threshold_slider.setValue(620)  # 默认阈值6.6
        self.threshold_slider.valueChanged.connect(self.update_particles)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("6.20")
        threshold_layout.addWidget(self.threshold_label)
        
        main_layout.addLayout(threshold_layout)
        
        # 信息显示
        self.info_label = QLabel("Import STAR file")
        main_layout.addWidget(self.info_label)
        
        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(650, 650)
        self.image_label.setStyleSheet("border: 1px solid black;")
        main_layout.addWidget(self.image_label)
        
        # 初始状态
        self.update_controls_state(False)
    
    def update_controls_state(self, enabled):
        """更新控件状态"""
        self.diameter_edit.setEnabled(enabled)
        #self.increase_button.setEnabled(enabled)
        #self.decrease_button.setEnabled(enabled)
        self.prev_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.threshold_slider.setEnabled(enabled)
    
    def open_file(self):
        """打开STAR文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choose particles file", "", "STAR Files (*.star);;All Files (*)")
        
        if not file_path:
            return

        # 设置STAR文件所在目录
        self.star_file_dir = os.getcwd()
        #print(self.star_file_dir)
        
        try:
            self.parse_star_file(file_path)
            self.current_index = 0
            self.update_display()
            self.update_controls_state(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法解析STAR文件: {str(e)}")
    
    def parse_star_file(self, file_path):
        """解析STAR文件"""
        self.micrographs.clear()
        self.micrograph_names.clear()
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 检查文件版本
        #if not content.startswith("# version 30001"):
         #   raise ValueError("不支持的STAR文件版本")
        
        # 查找data_particles部分
        particles_section = re.search(r'data_particles\s+(.*?)(?=data_|\Z)', content, re.DOTALL)
        if not particles_section:
            raise ValueError("找不到data_particles数据")
        
        particles_content = particles_section.group(1)
        
        # 提取列索引
        column_indices = {}
        lines = particles_content.split('\n')
        
        for line in lines[1:]:
            if line.startswith('_rln'):
                parts = line.split()
                column_name = parts[0]
                column_index = int(parts[1].strip('#')) - 1
                column_indices[column_name] = column_index
            elif line.strip() and not line.startswith('#'):
                break
        
        # 检查必要的列
        required_columns = ['_rlnMicrographName', '_rlnCoordinateX', '_rlnCoordinateY', '_rlnAutopickFigureOfMerit']
        for col in required_columns:
            if col not in column_indices:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 提取颗粒数据
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= max(column_indices.values()) + 1:
                    micrograph_name = parts[column_indices['_rlnMicrographName']]
                    x = float(parts[column_indices['_rlnCoordinateX']])
                    y = float(parts[column_indices['_rlnCoordinateY']])
                    score = float(parts[column_indices['_rlnAutopickFigureOfMerit']])
                    
                    self.micrographs[micrograph_name].append({
                        'x': x,
                        'y': y,
                        'score': score
                    })
        
        # 获取微图形名称列表
        self.micrograph_names = list(self.micrographs.keys())
        
        if not self.micrograph_names:
            raise ValueError("STAR文件中没有找到照片数据")
    
    def update_display(self):
        """更新显示"""
        if not self.micrograph_names:
            return
        
        # 获取当前微图形名称和颗粒
        micrograph_name = self.micrograph_names[self.current_index]
        particles = self.micrographs[micrograph_name]
        
        # 获取阈值
        threshold = self.threshold_slider.value() / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        
        # 过滤颗粒
        filtered_particles = [p for p in particles if p['score'] >= threshold]
        
        # 更新信息标签
        self.info_label.setText(
            f"Micrograph: {micrograph_name.split('/')[-1]}\n"
            f"Total particles: {len(particles)} | "
            f"Displayed particles: {len(filtered_particles)} | "
            f"Score: {threshold:.2f}"
        )
        
        # 加载图像（这里需要根据实际情况实现图像加载）
        # 由于STAR文件只包含图像路径，实际图像可能不在同一位置
        # 这里我们创建一个模拟图像用于演示
        pixmap = self.create_image_with_particles(micrograph_name, filtered_particles)
        self.image_label.setPixmap(pixmap)

    def update_particles(self):
        """更新显示"""
        if not self.micrograph_names:
            return
        
        # 获取当前微图形名称和颗粒
        micrograph_name = self.micrograph_names[self.current_index]
        particles = self.micrographs[micrograph_name]
        
        # 获取阈值
        threshold = self.threshold_slider.value() / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        
        # 过滤颗粒
        filtered_particles = [p for p in particles if p['score'] >= threshold]
        
        # 更新信息标签
        self.info_label.setText(
            f"Micrograph: {micrograph_name.split('/')[-1]}\n"
            f"Total particles: {len(particles)} | "
            f"Displayed particles: {len(filtered_particles)} | "
            f"Score: {threshold:.2f}"
        )
        
        # 加载图像（这里需要根据实际情况实现图像加载）
        # 由于STAR文件只包含图像路径，实际图像可能不在同一位置
        # 这里我们创建一个模拟图像用于演示
        pixmap = self.create_image_with_particles(micrograph_name, filtered_particles, 1)
        self.image_label.setPixmap(pixmap)

    def mrc_to_qpixmap(self, filepath):
        try:
            with mrcfile.open(filepath) as mrc:
                ori_data = mrc.data.astype(np.float32)
                pixel_size = mrc.voxel_size

                # 处理3D数据（取第一层）
                if ori_data.ndim == 3:
                    ori_data = ori_data[0]
            
            fft_signal = np.fft.rfft2(ori_data)
    
            # 优化2: 预计算频率网格（避免重复计算）
            rows, cols = ori_data.shape
            cutoff_freq = 1/15

            # 优化3: 向量化计算频率网格
            freq_x = np.fft.rfftfreq(cols, pixel_size['x'])
            freq_y = np.fft.fftfreq(rows, pixel_size['y'])

            # 优化4: 使用广播而不是meshgrid
            freq_2d_sq = freq_y[:, np.newaxis]**2 + freq_x**2  # 平方避免开方
            freq_2d = np.sqrt(freq_2d_sq)  # 只在需要时计算

            # 优化5: 简化滤波器设计
            filter_mask = np.zeros_like(fft_signal, dtype=np.float32)

            # 硬截止部分
            hard_mask = freq_2d <= cutoff_freq
            filter_mask[hard_mask] = 1.0

            # 软边缘部分 - 优化计算
            soft_band_width = 8 / (pixel_size['x'] * rows)
            soft_low = cutoff_freq
            soft_high = cutoff_freq + soft_band_width

            soft_region = (freq_2d > soft_low) & (freq_2d <= soft_high)
            if np.any(soft_region):
                # 预计算cosine衰减
                normalized_dist = (freq_2d[soft_region] - soft_low) / soft_band_width
                filter_mask[soft_region] = 0.5 * np.cos(np.pi * normalized_dist) + 0.5

            # 应用滤波器
            filtered_fft_signal = fft_signal * filter_mask

            # 逆变换
            filtered_signal = np.fft.irfft2(filtered_fft_signal, s=ori_data.shape)

            # 优化6: 向量化后处理
            data = np.real(filtered_signal)
            mean = np.mean(data)
            std = np.std(data)

            n_sigma = 3
            min_val = mean - n_sigma * std
            max_val = mean + n_sigma * std

            # 使用clip代替多个mask操作
            data = np.clip(data, min_val, max_val)
            data = (data - min_val) / (2 * n_sigma * std) * 255

            data = data.astype(np.uint8)

            
            qimage = QImage(data.data, cols, rows, cols, QImage.Format.Format_Grayscale8)

            # 缩小8倍
            pixmap = QPixmap.fromImage(qimage).scaled(
                int(cols/6), int(rows/6),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
            return pixmap, cols, rows
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取文件：{str(e)}")
            return QPixmap(), 0, 0
    
    def create_image_with_particles(self, micrograph_name, particles, mode=0):
        if mode == 1:
            # 创建QPainter在pixmap上绘制
            pixmap = self.pixmap.copy()
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.GlobalColor.green, 1))

            for particle in particles:
                x = particle['x'] * self.scale
                y = particle['y'] * self.scale
                score = particle['score']

                # 绘制圆圈
                diameter = int(self.circle_diameter * self.scale)
                radius = self.circle_diameter * self.scale // 2
                painter.drawEllipse(int(x - radius), int(y - radius), diameter, diameter)

            # 绘制得分
            #painter.drawText(int(x + radius + 2), int(y), f"{score:.1f}")

            painter.end()
            return pixmap
        else:  
            # 构建mrc文件路径
            # 假设STAR文件中的路径是相对于STAR文件所在目录的相对路径
            mrc_path = os.path.join(self.star_file_dir, micrograph_name)
            
            pixmap, original_width, original_height = self.mrc_to_qpixmap(mrc_path)
            self.pixmap = pixmap.copy()
            if pixmap.isNull():
                return pixmap

            # 计算缩放比例
            self.scale = pixmap.width() / original_width

            # 创建QPainter在pixmap上绘制
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.GlobalColor.green, 1))

            for particle in particles:
                x = particle['x'] * self.scale
                y = particle['y'] * self.scale
                score = particle['score']

                # 绘制圆圈
                diameter = int(self.circle_diameter * self.scale)
                radius = self.circle_diameter * self.scale // 2
                painter.drawEllipse(int(x - radius), int(y - radius), diameter, diameter)

            # 绘制得分
            #painter.drawText(int(x + radius + 2), int(y), f"{score:.1f}")

            painter.end()
            return pixmap
        
    def update_diameter(self):
        """更新圆圈直径"""
        try:
            new_diameter = int(self.diameter_edit.text())
            if new_diameter > 0:
                self.circle_diameter = new_diameter
                self.update_display()
            else:
                QMessageBox.warning(self, "警告", "直径必须为正整数")
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的整数")
    
    def prev_image(self):
        """显示上一张图像"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def next_image(self):
        """显示下一张图像"""
        if self.current_index < len(self.micrograph_names) - 1:
            self.current_index += 1
            self.update_display()

def main():
    app = QApplication(sys.argv)
    viewer = StarFileViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()