from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QButtonGroup, QLabel
from PyQt6.QtCore import Qt
import pynvml

class GpuSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select GPU")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumSize(100, 400)
        
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.gpu_mem_usable = []
        
        # 说明标签
        label = QLabel("Choose your GPU(s)")
        layout.addWidget(label)
        
        # GPU选择区域
        self.gpu_group = QButtonGroup(self)
        self.gpu_group.setExclusive(False)  # 允许多选
        
        gpu_layout = QVBoxLayout()
        self.gpu_boxes = []
        for i in range(8):  # 0-7号GPU
            box = QCheckBox(f"GPU {i}")
            box.setChecked(i == 0)  # 默认选择GPU 0
            self.gpu_group.addButton(box, i)
            self.gpu_boxes.append(box)
            gpu_layout.addWidget(box)
        
        layout.addLayout(gpu_layout)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(btn_ok)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
        self.update_gpu_usage()
    
    def get_selected_gpus(self):
        """获取选中的GPU列表"""
        selected_list = []
        gpu_mem_list = []
        for i in range(8):
            if self.gpu_boxes[i].isChecked():
                selected_list.append(i)
                gpu_mem_list.append(self.gpu_mem_usable[i])
        return selected_list, gpu_mem_list
    
    def get_selected_gpus_str(self):
        """获取选中的GPU字符串表示（逗号分隔）"""
        return ",".join(str(i) for i in self.get_selected_gpus())

    def update_gpu_usage(self):
        """更新GPU使用情况显示"""
        try:
            pynvml.nvmlInit()
            
            for i in range(8):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # 格式化使用信息
                    gpu_util = f"{util.gpu}%"
                    mem_util = f"{mem_info.used//1024**2}/{mem_info.total//1024**2}MB"
                    
                    # 更新复选框文本
                    self.gpu_boxes[i].setText(f"GPU {i} (used: {gpu_util}, memory: {mem_util})")
                    self.gpu_mem_usable.append((mem_info.total-mem_info.used)//1024**2)
                    
                    # 根据使用情况设置颜色
                    if util.gpu > 70:
                        self.gpu_boxes[i].setStyleSheet("color: red;")
                    elif util.gpu > 30:
                        self.gpu_boxes[i].setStyleSheet("color: orange;")
                    else:
                        self.gpu_boxes[i].setStyleSheet("")
                
                except pynvml.NVMLError:
                    self.gpu_boxes[i].setText(f"GPU {i} (inaccessible)")
                    self.gpu_boxes[i].setEnabled(False)
                    self.gpu_boxes[i].setCheckState(Qt.CheckState.Unchecked)
            
            pynvml.nvmlShutdown()
        
        except ImportError:
            # pynvml不可用，不做处理
            pass