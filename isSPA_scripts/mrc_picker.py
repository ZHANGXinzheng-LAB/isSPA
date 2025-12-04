#!/usr/bin/env python3

import os
import shutil
import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
#from low_pass_filter import apply_lowpass_filter_2d
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QEvent

class MRCViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRC图片选择工具")
        self.setGeometry(100, 100, 700, 800)
        
        self.file_list = []
        self.selected = []
        self.current_index = 0
        
        self.init_ui()
        
    def init_ui(self):
        # 创建控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.folder_btn = QPushButton("选择文件夹")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMouseTracking(True)  # 启用鼠标追踪
        self.prev_btn = QPushButton("上一张")
        self.next_btn = QPushButton("下一张")
        self.select_btn = QPushButton("选中")
        self.deselect_btn = QPushButton("取消选中")  # 新增按钮
        self.status_label = QLabel("状态: ")
        self.save_btn = QPushButton("保存选中图片")

        # 添加跳转控件
        self.jump_label = QLabel("跳转到:")
        self.jump_input = QLineEdit()
        self.jump_input.setFixedWidth(200)
        self.jump_input.setPlaceholderText("输入序号")
        self.jump_btn = QPushButton("跳转")
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.folder_btn)
        layout.addWidget(self.image_label)
        
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)

        # 修改按钮布局
        action_layout = QHBoxLayout()  # 新增操作按钮布局
        action_layout.addWidget(self.select_btn)
        action_layout.addWidget(self.deselect_btn)
        layout.addLayout(action_layout)

        # 跳转控件布局
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(self.jump_label)
        jump_layout.addWidget(self.jump_input)
        jump_layout.addWidget(self.jump_btn)
        #jump_layout.addStretch()
        layout.addLayout(jump_layout)

        layout.addWidget(self.status_label)
        layout.addWidget(self.save_btn)
        
        central_widget.setLayout(layout)
        
        # 连接信号
        self.folder_btn.clicked.connect(self.load_folder)
        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn.clicked.connect(self.show_next)
        self.select_btn.clicked.connect(self.toggle_selection)
        self.deselect_btn.clicked.connect(self.deselect_current)
        self.save_btn.clicked.connect(self.save_selected)
        self.jump_btn.clicked.connect(self.jump_to_image)
        self.jump_input.returnPressed.connect(self.jump_to_image)  # 支持回车键跳转

        self.image_label.installEventFilter(self)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def eventFilter(self, obj, event):
        """处理图片点击事件"""
        if obj == self.image_label and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.RightButton:
                # 仅当图片已被选中时才取消选中
                if self.selected[self.current_index]:
                    self.deselect_current()
            if event.button() == Qt.MouseButton.LeftButton:
                # 仅当图片已被选中时才取消选中
                if not self.selected[self.current_index]:
                    self.toggle_selection()
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """处理键盘事件"""
        key = event.key()
        if key == Qt.Key.Key_Left:
            self.show_previous()
        elif key == Qt.Key.Key_Right:
            self.show_next()
        else:
            super().keyPressEvent(event)
        
    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择包含电镜照片的文件夹")
        if folder:
            self.file_list = sorted(
                [os.path.join(folder, f) for f in os.listdir(folder) 
                 if f.lower().endswith('.mrc')],
                key=lambda x: os.path.basename(x)
            )
            if not self.file_list:
                QMessageBox.warning(self, "错误", "文件夹中没有找到MRC文件！")
                return
            
            self.selected = [False] * len(self.file_list)
            self.current_index = 0
            self.update_display()

    def deselect_current(self):
        """取消当前图片的选中状态"""
        if 0 <= self.current_index < len(self.selected):
            self.selected[self.current_index] = False
            self.update_display()  # 立即更新状态显示

    def jump_to_image(self):
        """跳转到指定序号的图片"""
        if not self.file_list:
            QMessageBox.warning(self, "错误", "请先加载文件夹！")
            return
            
        text = self.jump_input.text()
        if not text.isdigit():
            QMessageBox.warning(self, "输入错误", "请输入有效的数字！")
            return
            
        index = int(text) - 1 # 用户输入的是序号（从1开始），转换为索引（从0开始）
        
        if index < 0 or index >= len(self.file_list):
            QMessageBox.warning(self, "范围错误", f"序号应在1到{len(self.file_list)}之间！")
            return
            
        self.current_index = index
        self.update_display()
        self.jump_input.clear()  # 清空输入框
            
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
            freq_2d = np.sqrt(freq_2d_sq) 

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
            
            # 缩小6倍
            return QPixmap.fromImage(qimage).scaled(
                int(cols/6), int(rows/6),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取文件：{str(e)}")
            return QPixmap()
            
    def update_display(self):
        if 0 <= self.current_index < len(self.file_list):
            pixmap = self.mrc_to_qpixmap(self.file_list[self.current_index])
            self.image_label.setPixmap(pixmap)
            
            status = f"当前图片：{self.current_index+1}/{len(self.file_list)} | "
            status += "已选中" if self.selected[self.current_index] else "未选中"
            self.status_label.setText(status)
        else:
            self.image_label.clear()
            self.status_label.setText("没有更多图片")

        if self.selected[self.current_index]:
            self.image_label.setStyleSheet("border: 3px solid green;")
        else:
            self.image_label.setStyleSheet("border: 3px solid gray;")
            
    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
            
    def show_next(self):
        if self.current_index < len(self.file_list) - 1:
            self.current_index += 1
            self.update_display()
            
    def toggle_selection(self):
        if 0 <= self.current_index < len(self.selected):
            self.selected[self.current_index] = True
            if self.selected[self.current_index]:
                self.show_next()
            else:
                self.update_display()
                
    def save_selected(self):
        if not any(self.selected):
            QMessageBox.warning(self, "警告", "没有选中的图片！")
            return
            
        dest_folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if dest_folder:
            try:
                count = 0
                for i, selected in enumerate(self.selected):
                    if selected:
                        filename = os.path.basename(self.file_list[i])
                        dest_path = os.path.join(dest_folder, filename)
                        shutil.copy2(self.file_list[i], dest_path)
                        count += 1
                        
                QMessageBox.information(self, "完成", 
                    f"成功保存 {count} 张图片到：\n{dest_folder}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

if __name__ == "__main__":
    app = QApplication([])
    viewer = MRCViewer()
    viewer.show()
    app.exec()