import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt6.QtCore import QTimer, pyqtSignal, QObject

class ProcessManager(QObject):
    """进程管理器，负责并行执行任务"""
    
    # 信号定义
    task_started = pyqtSignal(str)  # 任务名称
    task_finished = pyqtSignal(str, int)  # 任务名称, 退出码
    task_output = pyqtSignal(str, str)  # 任务名称, 输出内容
    all_tasks_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.running_processes = {}  # {task_name: subprocess.Popen}
        #self.task_gpu_mapping = {}   # {task_name: gpu_id}
        self.executor = None
        self.is_running = False
        
    def start_parallel_tasks(self, tasks, n_g):
        """并行执行多个任务，每个任务分配一个GPU"""
        if not tasks or not gpu_ids:
            return
            
        self.is_running = True
        self.running_processes.clear()
        #self.task_gpu_mapping.clear()
        
        '''
        # 为每个任务分配GPU
        gpu_assignments = []
        for i, task in enumerate(tasks):
            gpu_id = gpu_ids[i % len(gpu_ids)]  # 循环分配GPU
            gpu_assignments.append((task, gpu_id))
            self.task_gpu_mapping[task['name']] = gpu_id
        '''
        
        # 使用线程池并行执行
        self.executor = ThreadPoolExecutor(max_workers=n_g)
        
        n_start = tasks["name"].split("part")[1]
        # 提交所有任务
        future_to_task = {}
        for i in range(n_g):
            tasks["name"] = f"isSPA part{n_start+i}"
            tasks["params"] = [f"config{n_start+i}"]
            future = self.executor.submit(self.run_single_task, tasks)
            future_to_task[future] = task['name']
            
        # 启动监控线程
        monitor_thread = threading.Thread(
            target=self.monitor_tasks, 
            args=(future_to_task,)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def run_single_task(self, task):
        """在单个GPU上运行任务"""
        task_name = task['name']
        #n_t = tasks["name"].split("part")[1]
        
        # 发送任务开始信号
        self.task_started.emit(task_name)
        
        try:
            # 构建命令，添加GPU参数
            cmd = self.build_command(task)
            
            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并标准输出和错误输出
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 保存进程引用
            self.running_processes[task_name] = process
            
            # 读取输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.task_output.emit(task_name, output.strip())
            
            # 获取退出码
            return_code = process.poll()
            
            # 发送任务完成信号
            self.task_finished.emit(task_name, return_code)
            
            return return_code
            
        except Exception as e:
            self.task_output.emit(task_name, f"错误: {str(e)}")
            self.task_finished.emit(task_name, -1)
            return -1
    
    def build_command(self, task):
        """构建命令，添加GPU参数"""
        cmd = [task['command'], task['params']]
        
        return cmd
    
    def monitor_tasks(self, future_to_task):
        """监控所有任务的完成状态"""
        # 等待所有任务完成
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                return_code = future.result()
                # 输出已经在run_single_task中处理
            except Exception as e:
                self.task_output.emit(task_name, f"任务异常: {str(e)}")
        
        # 所有任务完成后发送信号
        self.is_running = False
        self.all_tasks_finished.emit()
    
    def kill_all_tasks(self):
        """终止所有运行中的任务"""
        for task_name, process in self.running_processes.items():
            try:
                process.terminate()  # 尝试正常终止
                time.sleep(2)  # 等待2秒
                if process.poll() is None:  # 如果进程还在运行
                    process.kill()  # 强制终止
                self.task_output.emit(task_name, "任务已被终止")
            except Exception as e:
                self.task_output.emit(task_name, f"终止任务时出错: {str(e)}")
        
        # 关闭执行器
        if self.executor:
            self.executor.shutdown(wait=False)
        
        self.is_running = False
        self.running_processes.clear()
        #self.task_gpu_mapping.clear()