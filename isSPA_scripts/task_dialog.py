#!/usr/bin/env python3
import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QLabel, QLineEdit, QPushButton,
    QFormLayout, QVBoxLayout, QHBoxLayout, QComboBox, QStackedWidget, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt

class TaskConfigDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("任务配置")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 任务类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("任务类型:"))
        self.task_type = QComboBox()
        self.task_type.addItems([
            "preprocess", 
            "projection", 
            "isSPA", 
            "postprocess"
            #"自定义脚本"
        ])
        self.task_type.currentIndexChanged.connect(self.update_parameters_form)
        type_layout.addWidget(self.task_type)
        layout.addLayout(type_layout)
        
        # 参数堆叠窗口
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        
        # 创建不同任务类型的参数表单
        self.task_forms = {}
        self.task_forms["preprocess"] = self.create_preprocess_form()
        self.task_forms["projection"] = self.create_projection_form()
        #self.task_forms["isSPA"] = self.create_isSPA_form()
        #self.task_forms["postprocess"] = self.create_postprocess_form()
        #self.task_forms["自定义脚本"] = self.create_custom_script_form()
        
        # 添加所有表单到堆叠窗口
        for form in self.task_forms.values():
            self.stacked_widget.addWidget(form)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        btn_ok = QPushButton("确定")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(btn_ok)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
        
        # 初始化显示第一个任务的表单
        self.update_parameters_form(0)
    
    def create_form_group(self, title):
        """创建带标题的表单组"""
        group = QGroupBox(title)
        form_layout = QFormLayout()
        group.setLayout(form_layout)
        return group, form_layout
    
    def create_preprocess_form(self):
        """创建数据预处理任务的参数表单"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 输入输出组
        io_group, io_form = self.create_form_group("输入/输出设置")
        self.preprocess_input = QLineEdit()
        # 从主界面获取默认值
        if self.main_window.manual_inputs["Micrographs Directory"].text():
            micrographs_dir = self.main_window.manual_inputs["Micrographs Directory"].text()
            self.preprocess_input.setText(micrographs_dir)
        io_form.addRow(QLabel("Input"), self.preprocess_input)
        
        self.preprocess_bin = QLineEdit()
        if self.main_window.advanced_params["Bin"].text():
            bin1 = self.main_window.advanced_params["Bin"].text()
            self.preprocess_bin.setText(bin1)
        io_form.addRow(QLabel("Bin"), self.preprocess_bin)

        self.preprocess_pz = QLineEdit()
        if self.main_window.manual_inputs["Pixel Size (Å)"].text():
            pz = self.main_window.manual_inputs["Pixel Size (Å)"].text()
            self.preprocess_pz.setText(pz)
        io_form.addRow(QLabel("Pixel Size (Å)"), self.preprocess_pz)

        self.preprocess_ctf = QLineEdit()
        if self.main_window.manual_inputs["CTF Input"].text():
            ctf = self.main_window.manual_inputs["CTF Input"].text()
            self.preprocess_ctf.setText(ctf)
        io_form.addRow(QLabel("CTF Input"), self.preprocess_ctf)

        self.preprocess_output = QLineEdit()
        self.preprocess_output.setText("./micrograph_ctf.lst")
        io_form.addRow(QLabel("Output"), self.preprocess_output)
        
        layout.addWidget(io_group)
        
        return widget

    def create_projection_form(self):
        """创建数据预处理任务的参数表单"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 输入输出组
        io_group, io_form = self.create_form_group("输入/输出设置")
        self.projection_template = QLineEdit()
        # 从主界面获取默认值
        if self.main_window.manual_inputs["Template"].text():
            template = self.main_window.manual_inputs["Template"].text()
            self.projection_template.setText(template)
        io_form.addRow(QLabel("Template"), self.projection_template)
        
        self.projection_input = QLineEdit()
        if self.main_window.advanced_params["Angle Step File"].text():
            angle_file = self.main_window.advanced_params["Angle Step File"].text()
            self.projection_input.setText(angle_file)
        io_form.addRow(QLabel("Angle Step File"), self.projection_input)

        self.projection_parts = QLineEdit()
        if self.main_window.selected_gpus:
            p = len(self.main_window.selected_gpus)
            self.projection_parts.setText(p)
        io_form.addRow(QLabel("Parts"), self.projection_parts)

        self.projection_head = QLineEdit()
        self.projection_head.setText("./micrograph_ctf.lst")
        io_form.addRow(QLabel("Output"), self.preprocess_output)

        self.preprocess_output = QLineEdit()
        self.preprocess_output.setText(f'{self.main_window.file_path}/relion_projection_head.star')
        io_form.addRow(QLabel("Output"), self.preprocess_output)
        
        layout.addWidget(io_group)
        
        return widget
    
    def create_particle_picking_form(self):
        """创建粒子挑选任务的参数表单"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 输入组
        input_group, input_form = self.create_form_group("输入设置")
        self.picking_input = QLineEdit()
        self.picking_input.setPlaceholderText("输入微镜图像目录")
        input_form.addRow(QLabel("输入目录:"), self.picking_input)
        
        btn_browse_input = QPushButton("浏览...")
        btn_browse_input.clicked.connect(lambda: self.browse_directory(self.picking_input))
        input_form.addRow(QLabel(""), btn_browse_input)
        
        layout.addWidget(input_group)
        
        # 参数组
        param_group, param_form = self.create_form_group("挑选参数")
        
        # 从主界面获取像素尺寸作为默认值
        pixel_size = 1.0
        if self.main_window.param_display["Pixel Size (A)"].text():
            try:
                pixel_size = float(self.main_window.param_display["Pixel Size (A)"].text())
            except ValueError:
                pass
        
        self.picking_diameter = QDoubleSpinBox()
        self.picking_diameter.setRange(10, 500)
        self.picking_diameter.setValue(150)  # 典型值
        self.picking_diameter.setSuffix(" Å")
        param_form.addRow(QLabel("粒子直径:"), self.picking_diameter)
        
        self.picking_threshold = QDoubleSpinBox()
        self.picking_threshold.setRange(0.1, 0.9)
        self.picking_threshold.setValue(0.3)
        param_form.addRow(QLabel("挑选阈值:"), self.picking_threshold)
        
        layout.addWidget(param_group)
        
        # 输出组
        output_group, output_form = self.create_form_group("输出设置")
        self.picking_output = QLineEdit()
        self.picking_output.setPlaceholderText("输出STAR文件路径")
        output_form.addRow(QLabel("输出文件:"), self.picking_output)
        
        btn_browse_output = QPushButton("浏览...")
        btn_browse_output.clicked.connect(lambda: self.browse_file(self.picking_output, "*.star"))
        output_form.addRow(QLabel(""), btn_browse_output)
        
        layout.addWidget(output_group)
        
        # 设置默认输出路径
        if self.main_window.project_dir.text():
            default_dir = self.main_window.project_dir.text()
            self.picking_output.setText(os.path.join(default_dir, "Picking", "particles.star"))
        
        return widget
    
    def create_2d_classification_form(self):
        """创建2D分类任务的参数表单"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 输入组
        input_group, input_form = self.create_form_group("输入设置")
        self.class2d_input = QLineEdit()
        self.class2d_input.setPlaceholderText("输入粒子STAR文件")
        input_form.addRow(QLabel("输入文件:"), self.class2d_input)
        
        btn_browse_input = QPushButton("浏览...")
        btn_browse_input.clicked.connect(lambda: self.browse_file(self.class2d_input, "*.star"))
        input_form.addRow(QLabel(""), btn_browse_input)
        
        layout.addWidget(input_group)
        
        # 分类参数组
        param_group, param_form = self.create_form_group("分类参数")
        
        self.class2d_classes = QSpinBox()
        self.class2d_classes.setRange(2, 100)
        self.class2d_classes.setValue(50)
        param_form.addRow(QLabel("分类数量:"), self.class2d_classes)
        
        # 从主界面获取角度步长作为默认值
        angle_step = 5.0
        if self.main_window.manual_inputs["Angle Step (degree)"].text():
            try:
                angle_step = float(self.main_window.manual_inputs["Angle Step (degree)"].text())
            except ValueError:
                pass
        
        self.class2d_angle_step = QDoubleSpinBox()
        self.class2d_angle_step.setRange(0.1, 30.0)
        self.class2d_angle_step.setValue(angle_step)
        self.class2d_angle_step.setSuffix(" °")
        param_form.addRow(QLabel("角度步长:"), self.class2d_angle_step)
        
        self.class2d_iterations = QSpinBox()
        self.class2d_iterations.setRange(1, 50)
        self.class2d_iterations.setValue(25)
        param_form.addRow(QLabel("迭代次数:"), self.class2d_iterations)
        
        layout.addWidget(param_group)
        
        # 输出组
        output_group, output_form = self.create_form_group("输出设置")
        self.class2d_output = QLineEdit()
        self.class2d_output.setPlaceholderText("输出目录")
        output_form.addRow(QLabel("输出目录:"), self.class2d_output)
        
        btn_browse_output = QPushButton("浏览...")
        btn_browse_output.clicked.connect(lambda: self.browse_directory(self.class2d_output))
        output_form.addRow(QLabel(""), btn_browse_output)
        
        layout.addWidget(output_group)
        
        # 设置默认输出路径
        if self.main_window.project_dir.text():
            default_dir = self.main_window.project_dir.text()
            self.class2d_output.setText(os.path.join(default_dir, "Class2D"))
        
        return widget
    
    def create_3d_reconstruction_form(self):
        """创建3D重建任务的参数表单"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 输入组
        input_group, input_form = self.create_form_group("输入设置")
        self.recon3d_input = QLineEdit()
        self.recon3d_input.setPlaceholderText("输入粒子STAR文件")
        input_form.addRow(QLabel("输入文件:"), self.recon3d_input)
        
        btn_browse_input = QPushButton("浏览...")
        btn_browse_input.clicked.connect(lambda: self.browse_file(self.recon3d_input, "*.star"))
        input_form.addRow(QLabel(""), btn_browse_input)
        
        self.recon3d_symmetry = QLineEdit("C1")
        input_form.addRow(QLabel("对称性:"), self.recon3d_symmetry)
        
        layout.addWidget(input_group)
        
        # 重建参数组
        param_group, param_form = self.create_form_group("重建参数")
        
        # 从主界面获取电压作为默认值
        voltage = 300.0
        if self.main_window.param_display["Voltage (kV)"].text():
            try:
                voltage = float(self.main_window.param_display["Voltage (kV)"].text())
            except ValueError:
                pass
        
        self.recon3d_voltage = QDoubleSpinBox()
        self.recon3d_voltage.setRange(80, 300)
        self.recon3d_voltage.setValue(voltage)
        self.recon3d_voltage.setSuffix(" kV")
        param_form.addRow(QLabel("加速电压:"), self.recon3d_voltage)
        
        # 从主界面获取Cs作为默认值
        cs = 2.7
        if self.main_window.param_display["Cs (mm)"].text():
            try:
                cs = float(self.main_window.param_display["Cs (mm)"].text())
            except ValueError:
                pass
        
        self.recon3d_cs = QDoubleSpinBox()
        self.recon3d_cs.setRange(0.1, 10.0)
        self.recon3d_cs.setValue(cs)
        self.recon3d_cs.setSuffix(" mm")
        param_form.addRow(QLabel("球差系数:"), self.recon3d_cs)
        
        self.recon3d_resolution = QDoubleSpinBox()
        self.recon3d_resolution.setRange(1, 50)
        self.recon3d_resolution.setValue(8.0)
        self.recon3d_resolution.setSuffix(" Å")
        param_form.addRow(QLabel("目标分辨率:"), self.recon3d_resolution)
        
        layout.addWidget(param_group)
        
        # 输出组
        output_group, output_form = self.create_form_group("输出设置")
        self.recon3d_output = QLineEdit()
        self.recon3d_output.setPlaceholderText("输出目录")
        output_form.addRow(QLabel("输出目录:"), self.recon3d_output)
        
        btn_browse_output = QPushButton("浏览...")
        btn_browse_output.clicked.connect(lambda: self.browse_directory(self.recon3d_output))
        output_form.addRow(QLabel(""), btn_browse_output)
        
        layout.addWidget(output_group)
        
        # 设置默认输出路径
        if self.main_window.project_dir.text():
            default_dir = self.main_window.project_dir.text()
            self.recon3d_output.setText(os.path.join(default_dir, "Reconstruction3D"))
        
        return widget
    
    def create_custom_script_form(self):
        """创建自定义脚本任务的参数表单"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 脚本选择组
        script_group, script_form = self.create_form_group("脚本设置")
        self.custom_script = QLineEdit()
        self.custom_script.setPlaceholderText("脚本文件路径")
        script_form.addRow(QLabel("脚本路径:"), self.custom_script)
        
        btn_browse_script = QPushButton("浏览...")
        btn_browse_script.clicked.connect(lambda: self.browse_file(self.custom_script, "*.py"))
        script_form.addRow(QLabel(""), btn_browse_script)
        
        layout.addWidget(script_group)
        
        # 参数组
        param_group, param_form = self.create_form_group("脚本参数")
        self.custom_params = QLineEdit()
        self.custom_params.setPlaceholderText("--input=file.star --output=dir/")
        param_form.addRow(QLabel("命令行参数:"), self.custom_params)
        
        layout.addWidget(param_group)
        
        return widget
    
    def update_parameters_form(self, index):
        """更新显示当前任务类型的参数表单"""
        task_name = self.task_type.currentText()
        self.stacked_widget.setCurrentWidget(self.task_forms[task_name])
    
    def browse_file(self, line_edit, filter="All Files (*)"):
        """浏览文件"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择文件", line_edit.text(), filter
        )
        if filename:
            line_edit.setText(filename)
    
    def browse_directory(self, line_edit):
        """浏览目录"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择目录", line_edit.text()
        )
        if directory:
            line_edit.setText(directory)
    
    def get_task_config(self):
        """获取任务配置"""
        task_type = self.task_type.currentText()
        config = {"type": task_type}
        
        if task_type == "preprocess":
            config.update({
                "command": "preprocess.py",
                "params": [
                    self.preprocess_input.text(), 
                    self.preprocess_bin.text(), 
                    self.preprocess_pz.text(), 
                    self.preprocess_ctf.text(), 
                    '-o', self.preprocess_output.text()
                ]
            })
        elif task_type == "projection":
            config.update({
                "command": "project_once.py",
                "params": {
                    "input_dir": self.picking_input.text(),
                    "output": self.picking_output.text(),
                    "diameter": self.picking_diameter.value(),
                    "threshold": self.picking_threshold.value()
                }
            })
        elif task_type == "2D分类":
            config.update({
                "command": "2d_classification.py",
                "params": {
                    "input": self.class2d_input.text(),
                    "output_dir": self.class2d_output.text(),
                    "classes": self.class2d_classes.value(),
                    "angle_step": self.class2d_angle_step.value(),
                    "iterations": self.class2d_iterations.value()
                }
            })
        elif task_type == "3D重建":
            config.update({
                "command": "3d_reconstruction.py",
                "params": {
                    "input": self.recon3d_input.text(),
                    "output_dir": self.recon3d_output.text(),
                    "symmetry": self.recon3d_symmetry.text(),
                    "voltage": self.recon3d_voltage.value(),
                    "cs": self.recon3d_cs.value(),
                    "resolution": self.recon3d_resolution.value()
                }
            })
        elif task_type == "自定义脚本":
            config.update({
                "command": self.custom_script.text(),
                "params": self.custom_params.text().split()  # 将参数字符串分割为列表
            })
        
        return config