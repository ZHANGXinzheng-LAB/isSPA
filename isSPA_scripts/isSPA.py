#!/usr/bin/env python3

import sys
import os
import subprocess
import shlex
import math
from threading import Thread
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QMessageBox, QHBoxLayout, QFileDialog, QTextEdit, QLabel, QLineEdit, QPushButton, QFormLayout, QVBoxLayout, QComboBox, QListWidget, QSplitter, QFrame, QDialog
)
from PyQt6.QtCore import QTimer, QProcess, Qt
from PyQt6.QtGui import QDoubleValidator, QIntValidator, QTextCursor

import json
from pathlib import Path
from collections import deque
from GPU_select import GpuSelectionDialog
from task_dialog import TaskConfigDialog
import numpy as np
from cal_isSPA_memory import cal_isSPA_memory
from processer_manager import ProcessManager

class TaskQueue:
    def __init__(self):
        self.queue = deque()
        self.current_task = None
        
    def add_task(self, name, command, params=None):
        """添加任务到队列"""
        task = {
            'name': name,
            'command': command,
            'params': params or {}
        }
        self.queue.append(task)

    def add_task_from_config(self, config):
        """添加任务到队列"""
        task = {
            'name': config['type'],
            'command': config['command'],
            'params': config['params']
        }
        self.queue.append(task)
    
    def next_task(self):
        """获取下一个任务"""
        if self.queue:
            self.current_task = self.queue.popleft()
            return self.current_task
        return None

    def clear(self):
        """清空任务队列"""
        self.queue.clear()
        self.current_task = None

    def length(self):
        return len(list(self.queue))

class InSituSPA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("isSPA_1.2")
        self.setGeometry(100, 100, 900, 500)

        self.script_path = Path(os.path.dirname(__file__))
        self.file_path = self.script_path.parent / 'files'

        #self.process_manager = ProcessManager()
        #self.process_manager.task_started.connect(self.on_task_started)
        #self.process_manager.task_finished.connect(self.on_task_finished)
        #self.process_manager.task_output.connect(self.on_task_output)
        #self.process_manager.all_tasks_finished.connect(self.on_all_tasks_finished)
        
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # 主水平布局
        main_layout = QHBoxLayout(main_widget)
        #main_layout.setContentsMargins(10, 10, 10, 10)
        #main_layout.setSpacing(15)
        
        # 左侧面板 - 参数设置
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        #left_layout.setContentsMargins(10, 10, 10, 10)
        
        # 右侧面板 - 任务和日志
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        #right_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建分割器，允许调整左右面板大小
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 550])  # 初始比例
        
        main_layout.addWidget(splitter)

        # 左侧面板内容 - 参数设置区域
        self.setup_left_panel(left_layout)        
        # 右侧面板内容 - 任务和日志区域
        self.setup_right_panel(right_layout)

        self.load_initial_config()  # 在初始化时加载配置

        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.handle_finished)

        # 初始化任务队列
        self.task_queue = TaskQueue()
        self.selected_gpus = [0]  # 默认使用GPU 

    def setup_left_panel(self, layout):
        # RELION控制部分
        self.btn_launch = QPushButton("Start RELION")
        self.btn_launch.clicked.connect(self.launch_relion)
        layout.addWidget(self.btn_launch)
        
        # 项目目录
        self.project_dir = os.getcwd()

        # 配置文件
        self.config_filename = "isSPA_config.json"

        # 导入参数按钮
        self.btn_import = QPushButton("Import parameters from RELION")
        self.btn_import.setToolTip("Choose the '<b>micrographs_ctf.star</b>' file from CtfFind job")
        self.btn_import.clicked.connect(self.import_parameters)
        layout.addWidget(self.btn_import)
        
        # 参数显示区域
        self.manual_inputs = {}
        form = QFormLayout()
        read_parameters = ["Pixel Size (Å)", "Voltage (kV)", "Cs (mm)", "CTF Input", "Micrographs Directory", "Number of micrographs"]
        for param in read_parameters:
            self.manual_inputs[param] = QLineEdit()
            #self.manual_inputs[param].setReadOnly(True)
            form.addRow(QLabel(param), self.manual_inputs[param])
        #layout.addLayout(form)

        directory_edit = QLineEdit()
        self.manual_inputs["Template"] = directory_edit
        select_button = QPushButton("select", self)
        select_button.clicked.connect(lambda:self.choose_template())
        directory_h_box = QHBoxLayout()
        directory_h_box.addWidget(directory_edit)
        directory_h_box.addWidget(select_button)
        form.addRow(QLabel("3D Template"), directory_h_box)

        directory_edit1 = QLineEdit()
        self.manual_inputs["FSC"] = directory_edit1
        select_button1 = QPushButton("select", self)
        select_button1.clicked.connect(lambda:self.choose_fsc())
        directory_h_box1 = QHBoxLayout()
        directory_h_box1.addWidget(directory_edit1)
        directory_h_box1.addWidget(select_button1)
        form.addRow(QLabel("Template FSC"), directory_h_box1)

        self.manual_inputs["Protein Mass (kDa)"] = QComboBox()
        self.manual_inputs["Protein Mass (kDa)"].addItems(["> 800", "> 1000"])
        #self.manual_inputs["Protein Mass (kDa)"].addItems(["> 400", "> 800", "> 1000"])
        self.manual_inputs["Protein Mass (kDa)"].activated.connect(self.set_angle_bin)
        directory_h_box4 = QHBoxLayout()
        directory_h_box4.addWidget(self.manual_inputs["Protein Mass (kDa)"])
        form.addRow(QLabel("Protein Mass (kDa)"), directory_h_box4)

        write_parameters = [
            #("Protein Mass (kDa)", ""),
            #("Angle Step (degree)", "3"), 
            #("n", "1"), 
            #("Highest Resolution (Å)", "6"), 
            #("Lowest Resolution (Å)", "400"), 
            ("Diameter (Å)", "250"),
            #("Recall Level", "0.333"),
            #("Score Threshold", "6.4"), 
            ("First Image", "1"), 
            ("Last Image", "10"), 
            ("Output", "Output.lst"), 
            #("Window Size (px)", "512"),
            #("Number of GPUs", "2")
            ]
        for param, default in write_parameters:
            self.manual_inputs[param] = QLineEdit(default)
            #self.manual_inputs[param].setReadOnly(True)
            form.addRow(param, self.manual_inputs[param])
        layout.addLayout(form)

        self.btn_save_config = QPushButton("Save Config")
        self.btn_save_config.clicked.connect(self.save_config)
        self.btn_load_config = QPushButton("Load Config")
        self.btn_load_config.clicked.connect(self.load_config)
        directory_h_box2 = QHBoxLayout()
        directory_h_box2.addWidget(self.btn_save_config)
        directory_h_box2.addWidget(self.btn_load_config)
        layout.addLayout(directory_h_box2)

        # 添加开始按钮
        self.btn_start = QPushButton("Start Processing")
        self.btn_start.clicked.connect(self.start_processing)
        layout.addWidget(self.btn_start)

        self.btn_display = QPushButton("Display Particles")
        self.btn_display.clicked.connect(self.display_particles)
        layout.addWidget(self.btn_display)

        # 高级参数区域
        self.advanced_params = {}
        self.advanced_widget = QWidget()
        advanced_layout = QFormLayout()

        directory_edit2 = QLineEdit()
        self.advanced_params["Angle Step File"] = directory_edit2
        self.advanced_params['Angle Step File'].setText(f"{self.file_path}/C1_delta2_mirror.txt")
        select_button2 = QPushButton("select", self)
        select_button2.clicked.connect(lambda:self.choose_angle_file())
        directory_h_box3 = QHBoxLayout()
        directory_h_box3.addWidget(directory_edit2)
        directory_h_box3.addWidget(select_button2)
        advanced_layout.addRow(QLabel("Angle Step File"), directory_h_box3)
        
        # 高级参数定义
        advanced_parameters = [
            #("Symmetry", "C1", "str"), 
            #("Angle Step File", "3", "str"),
            ("Psi Step (degree)", "3", "float"), 
            ("Highest Resolution (Å)", "8", "int"), 
            ("Lowest Resolution (Å)", "400", "int"),
            ("Score Threshold", "6.4", "float"), 
            #("Window Size (px)", "512", "int"),
            ("Bin", "2", "float"),
            ("n", "3", "float"),
        ]
        
        for text, default, dtype in advanced_parameters:
            input_box = QLineEdit(default)
            validator = QDoubleValidator() if dtype == "float" else QIntValidator()
            input_box.setValidator(validator)
            self.advanced_params[text] = input_box
            advanced_layout.addRow(text, input_box)
        
        
        self.advanced_widget.setLayout(advanced_layout)
        self.advanced_widget.hide()  # 默认隐藏
        
        # 高级设置按钮
        self.btn_advanced = QPushButton("Advanced Settings ▼")
        self.btn_advanced.clicked.connect(self.toggle_advanced)
        layout.insertWidget(3, self.btn_advanced)  # 插入到合适位置
        layout.insertWidget(4, self.advanced_widget)

    def setup_right_panel(self, layout):
        # 添加任务队列显示
        self.task_list = QListWidget()
        self.task_list.setMaximumHeight(100)
        layout.addWidget(QLabel("Task queue:"))
        layout.addWidget(self.task_list)

        #self.btn_add_task = QPushButton("Add Task")
        #self.btn_add_task.clicked.connect(self.add_task_dialog)
        #self.btn_delete_task = QPushButton("Delete Task")
        self.btn_kill = QPushButton("Kill All Tasks")
        self.btn_kill.clicked.connect(self.kill_all_tasks)
        self.btn_kill.setEnabled(False)
        #self.btn_delete_task.clicked.connect(self.delete_task)
        directory_h_box3 = QHBoxLayout()
        #directory_h_box3.addWidget(self.btn_add_task)
        #directory_h_box3.addWidget(self.btn_delete_task)
        directory_h_box3.addWidget(self.btn_kill)
        layout.addLayout(directory_h_box3)

        # 添加日志显示区域
        logs_label = QLabel("Log")
        layout.addWidget(logs_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(250)
        layout.addWidget(self.log_text)

        # 日志控制按钮
        log_btn_layout = QHBoxLayout()
        self.btn_clear_log = QPushButton("Clear")
        self.btn_clear_log.clicked.connect(lambda: self.log_text.clear())
        log_btn_layout.addWidget(self.btn_clear_log)
        
        self.btn_save_log = QPushButton("Save")
        self.btn_save_log.clicked.connect(self.save_log_to_file)
        log_btn_layout.addWidget(self.btn_save_log)

        layout.addLayout(log_btn_layout)
    
    def add_task_dialog(self):
        """添加任务对话框"""
        dialog = TaskConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            task_config = dialog.get_task_config()
            self.add_task_to_queue(task_config)

    def add_task_to_queue(self, task_config):
        """添加任务到队列"""
        task_name = task_config["type"]
        command = task_config["command"]
        params = task_config["params"]
        
        # 添加到队列
        self.task_queue.add_task(task_name, command, params)
        
        # 显示在任务列表
        #params_str = " ".join([f"{k}={v}" for k, v in params.items()]) 
        self.task_list.addItem(f"{task_name}")

    def choose_template(self):
        """选择文件夹"""
        file_name, ok = QFileDialog.getOpenFileName(self, "Choose the file of the template")
        if file_name:
            self.manual_inputs["Template"].setText(f"{file_name}")

    def choose_angle_file(self):
        """选择文件夹"""
        file_name, ok = QFileDialog.getOpenFileName(self, "Choose the file of the angle step", f'{self.file_path}')
        if file_name:
            self.advanced_params["Angle Step File"].setText(f"{file_name}")

    def choose_fsc(self):
        """选择文件夹"""
        file_name, ok = QFileDialog.getOpenFileName(self, "Choose the FSC file of the template")
        if file_name:
            self.manual_inputs["FSC"].setText(f"{file_name}")
            self.advanced_params["n"].setText("3")

    def toggle_advanced(self):
        """切换高级设置可见性"""
        if self.advanced_widget.isVisible():
            self.advanced_widget.hide()
            self.btn_advanced.setText("Advanced Settings ▼")
        else:
            self.advanced_widget.show()
            self.btn_advanced.setText("Advanced Settings ▲")

    def kill_all_tasks(self):
        """终止所有任务"""
        self.log_message("Killing all tasks...")
        self.process.kill()
        self.task_queue.clear()
        self.task_list.clear()

        #self.process_manager.kill_all_tasks()
        
        self.btn_start.setEnabled(True)
        self.btn_kill.setEnabled(False)

    def add_tasks(self, params, gpu_n):
        """添加任务到队列"""
        params1 = [f"{params['micrographs_dir']}", f"{params['template']}", f"{params['bin']}", f"{params['pixel_size']}", f"{params['ctf_input']}"]

        if params['bin'] > 1.0:
            pixel_b = np.round(params['pixel_size'] * params['bin'], 6)
            config_text = [f"Input = micrograph_bin{params['bin']}_ctf.lst\n"]
            template = params['template'].split('.mrc')[0]+f"_pz{pixel_b}.mrc"
        else:
            config_text = [f"Input = micrograph_ctf.lst\n"]
            template = params['template']
            pixel_b = params['pixel_size']
        
        # 添加到队列
        self.task_queue.add_task('preprocess', "preprocess.py", params1)
        self.task_list.addItem('preprocess')

        '''
        if params['symmetry'] == "":
            params2 = f"{self.file_path}/C1_delta{params['angle_step']}_mirror.txt -p {len(self.selected_gpus)}"
        else:
            params2 = f"{self.file_path}/{params['symmetry']}_delta{params['angle_step']}_mirror.txt -p {len(self.selected_gpus)}"
        
        self.task_queue.add_task('euler_to_list', "euler_angles_txt_to_star.py", params2)
        
        params3 = f"{params['template']} 'C1_delta{params['angle_step']}_mirror_part?.star'"
        self.task_queue.add_task('projection', "relion_project_once.py", params3)
        self.task_list.addItem('projection')
        '''

        with open(f"{self.project_dir}/relion_projection_head.star", 'w') as f:
            f.write(f"# version 50001\n\ndata_optics\n\nloop_\n_rlnOpticsGroup #1 \n_rlnOpticsGroupName #2 \n_rlnAmplitudeContrast #3 \n_rlnSphericalAberration #4 \n_rlnVoltage #5 \n_rlnImagePixelSize #6 \n_rlnImageSize #7 \n_rlnImageDimensionality #8 \n_rlnCtfDataAreCtfPremultiplied #9 \n1 opticsGroup1 0.100000 {params['cs']} {params['voltage']} {pixel_b} 400 2 0\n\n\n# version 50001\n\ndata_particles\n\nloop_\n_rlnAngleRot #1 \n_rlnAngleTilt #2 \n_rlnAnglePsi #3\n")

        params2 = [f"{template}", f"{params['angle_step_file']}", "-p", f"{gpu_n[0]}"]
        self.task_queue.add_task('projection', "project_once.py", params2)
        self.task_list.addItem('projection')
        
        config_text.append(f"Picking_templates = projections_{params['angle_step_file'].split('/')[-1].split('.txt')[0]}.mrcs\nEuler_angles_file = projections_{params['angle_step_file'].split('/')[-1].split('.txt')[0]}.star\nOutput = {params['output']}\n")
        config_text.append(f"Pixel_size = {pixel_b}\nPhi_step = {params['psi_step']}\nn = {params['n']}\nVoltage = {params['voltage']}\nCs = {params['cs']}\nAmplitude_contrast = 0.1\nHighest_resolution = {params['high_res']}\nLowest_resolution = {params['low_res']}\nDiameter = {math.ceil(params['d']*1.15/pixel_b)}\n\nNorm_type = 1\nInvert = 1\nScore_threshold = {params['score']}\nFirst_image = {params['first_image']-1}\nLast_image = {params['last_image']}\nGPU_ID = {self.selected_gpus[0]}\nWindow_size = 512\nPhase_flip = 1\nOverlap = {math.ceil(params['d']*1.15/pixel_b)}")
        output_name = params['output'].split('.lst')[0]
        if gpu_n[0] == 1:
            with open(f"{self.project_dir}/config", 'w') as f:
                f.writelines(config_text)

            self.task_queue.add_task("isSPA_GPU", "isSPA", ["config"])
            self.task_list.addItem("isSPA")

            params4 = [f"{params['output']}", f"{params['last_image']-params['first_image']+1}", "8", "14", f"{params['bin']}", f"{params['pixel_size']}", f"{params['micrographs_dir']}", f"{params['score']}"]
            self.task_queue.add_task('postprocess', "postprocess.py", params4)
            self.task_list.addItem('postprocess')

            params5 = [f"{output_name}_{params['score']}_merge.star"]
            self.task_queue.add_task('formatting', "relion30_to_31.py", params5)
            self.task_list.addItem('STAR formatting')
        else:
            if gpu_n[1] > 0:
                for i in range(gpu_n[1]):
                    iter_start = i*len(self.selected_gpus)+1
                    output_part = f"{output_name}_part{iter_start}.lst"
                    config_text[1]=(f"Picking_templates = projections_{params['angle_step_file'].split('/')[-1].split('.txt')[0]}_part{iter_start}.mrcs\nEuler_angles_file = projections_{params['angle_step_file'].split('/')[-1].split('.txt')[0]}_part{iter_start}.star\nOutput = {output_part}\n")
                    with open(f"{self.project_dir}/config{iter_start}", 'w') as f:
                        f.writelines(config_text)
                    params3 = [f"config{iter_start}", "--gpus"]
                    if len(self.selected_gpus) > 1:
                        for j in self.selected_gpus[1:]:
                            params3.append(f"{j}")
                        self.task_queue.add_task('configuration', "generate_configs.py", params3)
                        self.task_list.addItem('configuration')

                    self.task_queue.add_task(f"parallel isSPA", "parallel_isSPA_runner.py", [f"config{i*len(self.selected_gpus)+1}", f"{len(self.selected_gpus)}"])
                    self.task_list.addItem(f"isSPA part{i*len(self.selected_gpus)+1}-{(i+1)*len(self.selected_gpus)}")

                    for k in range(len(self.selected_gpus)):
                        params4 = [f"{output_name}_part{iter_start+k}.lst", f"{params['last_image']-params['first_image']+1}", "8", "14", f"{params['bin']}", f"{params['pixel_size']}", f"{params['micrographs_dir']}", f"{params['score']}"]
                        self.task_queue.add_task('postprocess', "postprocess.py", params4)
                        self.task_list.addItem('postprocess')
            if gpu_n[2] > 0:
                res_start = gpu_n[1]*len(self.selected_gpus)+1
                output_part = f"{output_name}_part{res_start}.lst"
                config_text[1]=(f"Picking_templates = projections_{params['angle_step_file'].split('/')[-1].split('.txt')[0]}_part{res_start}.mrcs\nEuler_angles_file = projections_{params['angle_step_file'].split('/')[-1].split('.txt')[0]}_part{res_start}.star\nOutput = {output_part}\n")
                with open(f"{self.project_dir}/config{res_start}", 'w') as f:
                    f.writelines(config_text)
                if gpu_n[2] > 1:
                    params3 = [f"config{res_start}", "--gpus"]
                    for j in self.selected_gpus[1:gpu_n[2]]:
                        params3.append(f"{j}")
                    self.task_queue.add_task('configuration', "generate_configs.py", params3)
                    self.task_list.addItem('configuration')

                self.task_queue.add_task(f"parallel isSPA", "parallel_isSPA_runner.py", [f"config{res_start}", f"{gpu_n[2]}"])
                self.task_list.addItem(f"isSPA part{res_start}-{gpu_n[0]}")

                for k in range(gpu_n[2]):
                    params4 = [f"{output_name}_part{res_start+k}.lst", f"{params['last_image']-params['first_image']+1}", "8", "14", f"{params['bin']}", f"{params['pixel_size']}", f"{params['micrographs_dir']}", f"{params['score']}"]
                    self.task_queue.add_task('postprocess', "postprocess.py", params4)
                    self.task_list.addItem('postprocess')

            
            params5 = [f"{output_name}_part1_{params['score']}_merge.lst"]
            self.task_queue.add_task('merge', "merge_outputs.py", params5)
            self.task_list.addItem('merge')

            params6 = [f"{output_name}_all.lst", f"{params['last_image']-params['first_image']+1}", "8", "14", f"{params['bin']}", f"{params['pixel_size']}", f"{params['micrographs_dir']}", f"{params['score']}"]
            self.task_queue.add_task('postprocess', "postprocess.py", params6)
            self.task_list.addItem('postprocess')

            params7 = [f"{output_name}_all_{params['score']}_merge.star"]
            self.task_queue.add_task('formatting', "relion30_to_31.py", params7)
            self.task_list.addItem('STAR formatting')

    def start_processing(self):
        """启动处理流程"""
        # 禁用按钮防止重复点击
        self.btn_start.setEnabled(False)
        self.btn_kill.setEnabled(True)

        if self.task_queue.length() > 0:
            self.start_next_task()
        else:
            # 先弹出GPU选择对话框
            gpu_dialog = GpuSelectionDialog(self)
            if gpu_dialog.exec() == QDialog.DialogCode.Accepted:
                self.selected_gpus, gpu_mem = gpu_dialog.get_selected_gpus()
                self.log_message(f"Selected GPU: {self.selected_gpus}\nAvailable device memory: {gpu_mem} MB")
                #self.log_message(f"")    
                # 收集并验证参数
                params, error = self.collect_parameters()
                if error:
                    self.log_error(error)
                    self.btn_start.setEnabled(True)
                    return
                
                validation_errors = self.validate_parameters(params)
                if validation_errors:
                    self.log_error("\n".join(validation_errors))
                    self.btn_start.setEnabled(True)
                    return

                validate_GPU_errors, gpu_n = self.validate_GPU_selections(params, gpu_mem)
                if validate_GPU_errors:
                    for i in validate_GPU_errors:
                        self.log_message(i)
                else:
                    self.log_message("Device memory requirement fulfilled!")
                
                self.add_tasks(params, gpu_n)
                self.start_next_task()
            else:
                self.log_message("Cancel GPU selection")
                self.btn_start.setEnabled(True)

        '''
        # 生成执行命令
        script_path = os.path.join(os.path.dirname(__file__), "preprocess_isSPA.py")
        cmd = [
            "python3",
            script_path,
            f"--CTF_file={params['input']}",
            f"--pixel_size={params['pixel_size']}",
            f"--voltage={params['voltage']}",
            f"--angle_step={params['angle_step']}",
            f"--window_size={params['window_size']}",
            f"--first={params['first_image']}",
            f"--last={params['last_image']}"
        ]
        
        # 启动外部进程
        try:
            self.log_message("Preprocessing...")
            self.process.start(' '.join(cmd))
        except Exception as e:
            self.log_error(f"Fails: {str(e)}")
            self.btn_start.setEnabled(True)
        '''

    def start_next_task(self):
        """启动队列中的下一个任务"""
        task = self.task_queue.next_task()
        self.task_list.takeItem(0)
        if not task:
            self.log_message("All tasks are finished!")
            self.btn_start.setEnabled(True)
            return
        
        #self.log_message("")
        '''
        self.log_message(f"Task begins: {task['name']}\n")

        if task['name'].startswith == 'isSPA_GPUs':
            self.process_manager.start_parallel_tasks(task, len(self.selected_gpus))
            self.process_manager.all_tasks_finished.connect()
        else:
            '''
        self.log_message(task['command'] + " " + " ".join(task['params']))
        
        # 启动外部进程
        try:
            self.process.start(task['command'], task['params'])
        except Exception as e:
            self.log_error(f"任务启动失败: {str(e)}")
            self.btn_start.setEnabled(True)

    def build_command(self, task):
        """根据任务构建命令"""
        cmd = task['params'].split()
        return cmd
    
    def handle_stdout(self):
        """处理标准输出"""
        data = self.process.readAllStandardOutput()
        message = bytes(data).decode("utf8")
        self.log_message(message)

    def handle_stderr(self):
        """处理错误输出"""
        data = self.process.readAllStandardError()
        message = bytes(data).decode("utf8")
        self.log_error(message)

    def handle_finished(self):
        """处理完成信号"""
        if self.process.exitStatus() == QProcess.ExitStatus.NormalExit:
            exit_code = self.process.exitCode()
            if exit_code == 0:
                self.log_message("Task finished!")
                self.log_message("-"*80+"\n")
                # 启动下一个任务
                QTimer.singleShot(1000, self.start_next_task)  # 延迟1秒
            else:
                self.log_error(f"Task fails (code: {exit_code})")
                self.btn_start.setEnabled(True)
        else:
            self.log_error("Task terminates")
            self.btn_start.setEnabled(True)

    def escape_html(self, text):
        """转义HTML特殊字符"""
        return (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;'))

    def scroll_to_bottom(self):
        """自动滚动到日志底部"""
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def log_message(self, message):
        """记录普通信息"""
        #formatted_message = message.replace('\n', '<br>')
        # 转义HTML特殊字符
        #formatted_message = self.escape_html(formatted_message)
        #self.log_text.append(f"<font color='black'>{message}</font>")

        current_html = self.log_text.toHtml()
    
        # 将新消息转换为HTML
        formatted_message = message.replace('\n', '<br>')
        new_message = f"<font color='black'>{formatted_message}</font>"
        
        # 将新消息添加到现有内容中
        updated_html = current_html + new_message
        
        # 设置更新后的HTML内容
        self.log_text.setHtml(updated_html)
        self.scroll_to_bottom()

    def log_error(self, message):
        """记录错误信息"""
        self.log_text.append(f"<font color='red'>{message}</font>")

    def collect_parameters(self):
        """收集所有参数"""
        params = {
            # 自动参数
            'pixel_size': self.manual_inputs["Pixel Size (Å)"].text().strip(),
            'voltage': self.manual_inputs["Voltage (kV)"].text().strip(),
            'cs': self.manual_inputs["Cs (mm)"].text().strip(),
            'ctf_input': self.manual_inputs["CTF Input"].text().strip(),
            'micrographs_dir': self.manual_inputs["Micrographs Directory"].text().strip(),
            'template': self.manual_inputs["Template"].text().strip(),
            #'p_m': self.manual_inputs["Protein Mass (kD)"].text().strip(),
            #'recall': self.manual_inputs["Recall Level"].text().strip(),
            'd': self.manual_inputs["Diameter (Å)"].text().strip(),
            'first_image': self.manual_inputs["First Image"].text().strip(),
            'last_image': self.manual_inputs["Last Image"].text().strip(),
            'output': self.manual_inputs["Output"].text().strip(),
            #'gpu': self.manual_inputs["Number of GPUs"].text().strip(),
            
            # 手动参数
            'angle_step_file': self.advanced_params["Angle Step File"].text().strip(),
            'psi_step': self.advanced_params["Psi Step (degree)"].text().strip(),
            'high_res': self.advanced_params["Highest Resolution (Å)"].text().strip(),
            'low_res': self.advanced_params["Lowest Resolution (Å)"].text().strip(),
            'score': self.advanced_params["Score Threshold"].text().strip(),
            #'window': self.advanced_params["Window Size (px)"].text().strip(),
            'bin': self.advanced_params["Bin"].text().strip(),
            'n': self.advanced_params["n"].text().strip(),
            'FSC': self.manual_inputs["FSC"].text().strip(),
        }
        
        # 转换数值类型
        try:
            params['pixel_size'] = float(params['pixel_size'])
            params['voltage'] = int(float(params['voltage']))
            params['cs'] = float(params['cs'])
            params['ctf_input'] = str(params['ctf_input'])
            params['micrographs_dir'] = str(params['micrographs_dir'])
            params['template'] = str(params['template'])
            #params['p_m'] = int(params['p_m'])
            params['d'] = int(params['d'])
            #params['recall'] = float(params['recall'])
            params['first_image'] = int(params['first_image'])
            params['last_image'] = int(params['last_image'])
            params['output'] = str(params['output'])
            #params['gpu'] = int(params['gpu'])
            #params['symmetry'] = str(params['symmetry'])
            params['psi_step'] = float(params['psi_step'])
            params['high_res'] = float(params['high_res'])
            params['low_res'] = float(params['low_res'])
            params['score'] = float(params['score'])
            #params['window'] = int(params['window'])
            params['bin'] = float(params['bin'])
            params['n'] = float(params['n'])
            params['FSC'] = str(params['FSC'])
            params['angle_step_file'] = str(params['angle_step_file'])
            
        except ValueError:
            return None, "Parameters with wrong type, please check parameter type"

        #for i,j in params.items():
        #self.log_message("")
        self.log_message(f"{params}\n")
        
        return params, ""

    def validate_parameters(self, params):
        """扩展验证逻辑"""
        errors = []
        
        # 必要参数检查
        required = ['pixel_size', 'voltage', 'cs', 'ctf_input', 'template', 'angle_step_file', 'd', 'psi_step', 'high_res', 'low_res', 'score', 'bin', 'output']
        for key in required:
            if not params.get(key):
                errors.append(f"missing required parameters: {key}")
        
        # 数值范围验证
        #if params['angle_step'] <= 0 or params['angle_step'] > 360:
         #   errors.append("角度步长应在0-360度之间")
        if params['bin'] < 1.0:
            errors.append("Bin factor is smaller than one. Please check!")
        if params['first_image'] >= params['last_image']:
            errors.append("The first image number should be smaller than the last image number.")

        if not params['output'].endswith('.lst'):
            errors.append("The output must end with .lst")
        
        if params['FSC'] == "":
            self.log_message("*"*60)
            self.log_message(f"No FSC file found. Please make sure that template resolution is better than the highest resolution {params['high_res']} Å!")
            self.log_message("*"*60)
        
        # 电压合理性检查
        if not (80 <= params['voltage'] <= 300):
            errors.append("电压值应在80-300 kV之间")
    
        return errors

    def validate_GPU_selections(self, params, gpu_mem):
        message = []
        n_g = 1
        m_r = cal_isSPA_memory(params, n_g)
        # 默认相同GPU
        while min(gpu_mem) < m_r:
            n_g += 1
            m_r = cal_isSPA_memory(params, n_g)
        n_i = n_g // len(gpu_mem)
        n_r = n_g % len(gpu_mem)


        message.append(f"No. of required GPUs: {n_g}.")
        message.append(f"No. of available GPUs: {len(gpu_mem)}.")
        message.append(f"So, {n_i} iterations and {n_r} more GPUs are needed for calculation")

            #message.append("Select more GPUs")
            #message.append("Or you can choose another angle step file with larger steps (in advanced settings), but this may degrade isSPA performance, especially for small targets")

        return message, [n_g, n_i, n_r]

    def launch_relion(self):
        """启动RELION并禁用按钮"""
        try:
            self.relion_process = subprocess.Popen(["relion"])
            #self.btn_launch.setEnabled(False)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Unable to find RELION...")

    def display_particles(self):
        try:
            self.display_process = subprocess.Popen(["display_isSPA_picks.py"])
            #self.btn_launch.setEnabled(False)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Unable to find the display script...")

    def get_config_path(self):
        """获取配置文件路径"""
        project_dir = self.project_dir
        if project_dir:
            return Path(project_dir) / self.config_filename
        return Path.cwd() / self.config_filename
    
    def load_config(self):
        """加载配置文件"""
        file_name, ok = QFileDialog.getOpenFileName(self, "Load Configuration")
        #config_path = self.get_config_path()
        try:
            with open(file_name, 'r') as f:
                config = json.load(f)
                if config:
                    manual_params = config.get('params', {})
                    for name, value in manual_params.items():
                        if name in self.manual_inputs:
                            if name == "Protein Mass (kDa)":
                                self.manual_inputs[name].setCurrentText(value)
                            else:
                                self.manual_inputs[name].setText(value)
                        elif name in self.advanced_params:
                            self.advanced_params[name].setText(value)
        except Exception as e:
            print(f"Loading configuration fails: {str(e)}")
            return None

    def save_config(self):
        """保存当前配置"""
        config = {
            'params': {
                "Pixel Size (Å)": self.manual_inputs["Pixel Size (Å)"].text(),
                "Voltage (kV)": self.manual_inputs["Voltage (kV)"].text(),
                "Cs (mm)": self.manual_inputs["Cs (mm)"].text(),
                "CTF Input": self.manual_inputs["CTF Input"].text(),
                "Micrographs Directory": self.manual_inputs["Micrographs Directory"].text(),
                "Number of micrographs": self.manual_inputs["Number of micrographs"].text(),
                "Template": self.manual_inputs["Template"].text(),
                "Protein Mass (kDa)": self.manual_inputs["Protein Mass (kDa)"].currentText(),
                "Diameter (Å)": self.manual_inputs["Diameter (Å)"].text(),
                #"Recall Level": self.manual_inputs["Recall Level"].text(),

                "First Image": self.manual_inputs["First Image"].text(),
                "Last Image": self.manual_inputs["Last Image"].text(),
                "Output": self.manual_inputs["Output"].text(),
                #"Number of GPUs": self.manual_inputs["Number of GPUs"].text(), 
                "Angle Step File": self.advanced_params["Angle Step File"].text(),
                "Psi Step (degree)": self.advanced_params["Psi Step (degree)"].text(),
                "Highest Resolution (Å)": self.advanced_params["Highest Resolution (Å)"].text(),
                "Lowest Resolution (Å)": self.advanced_params["Lowest Resolution (Å)"].text(),
                "Score Threshold": self.advanced_params["Score Threshold"].text(),
                "FSC": self.manual_inputs["FSC"].text(),
                "n": self.advanced_params["n"].text(),
                "Bin": self.advanced_params["Bin"].text()
            }
        }
        
        try:
            file_name, ok = QFileDialog.getSaveFileName(self, "Save Configuration", f"{self.project_dir}", "All Files (*)")
            #config_path = self.get_config_path()
            with open(file_name, 'w') as f:
                json.dump(config, f, indent=4)
            QMessageBox.information(self, "Success", "Configuration is saved")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fails: {str(e)}")

    def load_initial_config(self):
        """初始化时加载配置"""
        config_path = self.get_config_path()
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                if config:
                    manual_params = config.get('params', {})
                    for name, value in manual_params.items():
                        if name in self.manual_inputs:
                            self.manual_inputs[name].setText(value)
                        elif name in self.advanced_params:
                            self.advanced_params[name].setText(value)

    def import_parameters(self):
        """从RELION项目导入参数"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择micrographs_ctf.star文件",
            self.project_dir,  # 初始目录
            "STAR Files (*.star);;All Files (*)"
        )
        
        if not filename:  # 用户取消选择
            return
        
        # 验证文件格式
        if not filename.endswith(".star"):
            QMessageBox.critical(self, "错误", "必须选择.star文件")
            return
        
        self.manual_inputs["CTF Input"].setText(filename)

        # 解析文件
        try:
            params = self.parse_star_file(filename)
            self.update_parameters(params)
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "文件不存在")
        except Exception as e:
            QMessageBox.critical(self, "解析错误", str(e))

    def parse_star_file(self, filename):
        """解析STAR文件，返回参数字典和统计信息"""
        results = {
            'optics_params': {},
            'micrographs_dir': "", 
            'micrographs_count': 0
        }
        
        current_block = None
        column_indices = {}  # 存储各参数对应的列索引
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # 跳过注释和空行
                    if line.startswith('#') or not line:
                        continue
                    
                    # 检测数据块类型
                    if line.startswith('data_'):
                        current_block = line.split('data_')[1].strip()
                        column_indices = {}  # 重置列索引
                        continue
                    
                    # 处理表头
                    if line.startswith('loop_'):
                        in_loop_header = True
                        column_index = 0
                        continue
                    
                    # 解析参数列索引 (例如 "_rlnVoltage #5")
                    if line.startswith('_') and '#' in line:
                        parts = line.split('#')
                        if len(parts) == 2:
                            param_name = parts[0].strip().lstrip('_')  # 去掉前导下划线
                            col_index = int(parts[1]) - 1  # 转换为0-based索引
                            column_indices[param_name] = col_index
                        continue
                    
                    # 处理数据行
                    if current_block == 'optics' and column_indices:
                        values = line.split()
                        for param, idx in column_indices.items():
                            if idx < len(values):
                                results['optics_params'][param] = values[idx]
                    
                    # 统计micrographs数据行
                    if current_block == 'micrographs':
                        if not line.startswith('_') and len(line) > 0:
                            if results['micrographs_dir'] == "":
                                results['micrographs_dir'] = '/'.join(line.split()[column_indices['rlnMicrographName']].split('/')[:-1])
                            results['micrographs_count'] += 1
        except Exception as e:
            raise RuntimeError(f"文件解析失败: {str(e)}")

        return results

    def set_angle_bin(self, index):
        pixel_size = float(self.manual_inputs["Pixel Size (Å)"].text().strip())
        if index == 0:
            self.advanced_params['Angle Step File'].setText(f'{self.file_path}/C1_delta3_mirror.txt')
            self.advanced_params['Psi Step (degree)'].setText('4')
            self.advanced_params['Highest Resolution (Å)'].setText('6')
            bin_f = np.round(5.5/pixel_size/2, 1)
            self.advanced_params['Bin'].setText(f"{bin_f}")
        elif index == 1:
            self.advanced_params['Angle Step File'].setText(f'{self.file_path}/C1_delta4_mirror.txt')
            self.advanced_params['Psi Step (degree)'].setText('5')
            self.advanced_params['Highest Resolution (Å)'].setText('8')
            bin_f = np.round(7.5/pixel_size/2, 1)
            self.advanced_params['Bin'].setText(f"{bin_f}")
        '''
        if index == 0:
            self.advanced_params['Angle Step File'].setText(f'{self.file_path}/C1_delta2_mirror.txt')
            self.advanced_params['Psi Step (degree)'].setText('3')
            self.advanced_params['Highest Resolution (Å)'].setText('4')
            bin_f = np.round(3.5/pixel_size/2, 1)
            self.advanced_params['Bin'].setText(f"{bin_f}")
        elif index == 1:
            self.advanced_params['Angle Step File'].setText(f'{self.file_path}/C1_delta3_mirror.txt')
            self.advanced_params['Psi Step (degree)'].setText('4')
            self.advanced_params['Highest Resolution (Å)'].setText('6')
            bin_f = np.round(5.5/pixel_size/2, 1)
            self.advanced_params['Bin'].setText(f"{bin_f}")
        elif index == 2:
            self.advanced_params['Angle Step File'].setText(f'{self.file_path}/C1_delta4_mirror.txt')
            self.advanced_params['Psi Step (degree)'].setText('5')
            self.advanced_params['Highest Resolution (Å)'].setText('8')
            bin_f = np.round(7.5/pixel_size/2, 1)
            self.advanced_params['Bin'].setText(f"{bin_f}")
            '''

    def update_parameters(self, parse_result):
        """更新界面显示"""
        params = parse_result['optics_params']
        mapping = {
            "rlnMicrographPixelSize": "Pixel Size (Å)",
            "rlnVoltage": "Voltage (kV)",
            "rlnSphericalAberration": "Cs (mm)"
        }
        
        for star_key, display_key in mapping.items():
            value = params.get(star_key, "")
            value = f"{float(value):.2f}"
            self.manual_inputs[display_key].setText(value)
        self.manual_inputs["Number of micrographs"].setText(str(parse_result['micrographs_count']))
        self.manual_inputs["Last Image"].setText(str(parse_result['micrographs_count']))

        self.manual_inputs["Micrographs Directory"].setText(str(parse_result['micrographs_dir'])+'/')

        # 显示统计信息
        count = parse_result.get('micrographs_count', 0)
        self.log_message(f"Found {count} micrographs!")

    def save_log_to_file(self):
        """保存日志到文件"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存日志", "", "日志文件 (*.log);;所有文件 (*)"
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"Log is saved to: {filename}")
            except Exception as e:
                self.log_error(f"Log saving fails: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InSituSPA()
    window.show()
    sys.exit(app.exec())