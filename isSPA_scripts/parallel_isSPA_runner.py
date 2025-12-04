#!/usr/bin/env python3

import subprocess
import concurrent.futures
import time
import os
import sys
from datetime import datetime

def run_single_task(config_name, task_id):
    """
    运行单个任务的函数
    
    参数:
        config_name: 配置基础名称
        task_id: 任务ID
    
    返回:
        tuple: (任务ID, 是否成功, 输出信息)
    """
    # 构建完整的配置名称
    num = int(config_name.split('config')[1])
    full_config_name = f"config{num+task_id} > config{num+task_id}_log"
    
    # 构建要执行的命令
    command = f"isSPA {full_config_name}"
    
    print(f"Task {task_id+1}: {command}")
    start_time = time.time()
    
    try:
        # 执行命令
        process = subprocess.Popen(
            command,
            shell=True,
            #stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE
            #capture_output=True,
            text=True
            #timeout=3600  # 1小时超时，可根据需要调整
        )
        
        stdout, stderr = process.communicate()
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 检查执行结果
        if process.returncode == 0:
            print(f"Task {task_id+1} finished (Total time: {execution_time:.2f} s)")
            return (task_id, True, f"成功 - 耗时: {execution_time:.2f}秒")
        else:
            print(f"Task {task_id} fails (Total time: {execution_time:.2f} s)")
            error_msg = f"失败 - 退出码: {process.returncode}, 错误: {process.stderr.strip()}"
            return (task_id, False, error_msg)
            
    except subprocess.TimeoutExpired:
        print(f"任务 {task_id} 执行超时")
        return (task_id, False, "超时")
    except Exception as e:
        print(f"任务 {task_id} 执行异常: {str(e)}")
        return (task_id, False, f"异常: {str(e)}")

def run_parallel_tasks(base_config, num_tasks, max_workers=None):
    """
    并行运行多个任务
    
    参数:
        base_config: 配置基础名称
        num_tasks: 任务数量
        max_workers: 最大并行工作进程数 (默认使用CPU核心数)
    
    返回:
        dict: 包含所有任务执行结果的字典
    """
    # 如果没有指定最大工作进程数，使用CPU核心数
    if max_workers is None:
        max_workers = 8
    
    print(f"{num_tasks} parallel tasks")
    print(f"The base configuration file: {base_config}")
    print(f"Maximum parrallel tasks: {max_workers}")
    print(f"Begin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 50)
    
    # 存储所有任务的结果
    results = {}
    
    # 使用线程池执行器并行执行任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(run_single_task, base_config, i): i+1
            for i in range(num_tasks)
        }
        
        # 收集任务结果
        for future in concurrent.futures.as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                task_id, success, message = future.result()
                results[task_id] = {
                    'success': success,
                    'message': message
                }
            except Exception as e:
                results[task_id] = {
                    'success': False,
                    'message': f"任务执行异常: {str(e)}"
                }
    
    print("-" * 50)
    print(f"All parrallel tasks finished!")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

def print_summary(results):
    """
    打印任务执行摘要
    
    参数:
        results: 任务执行结果字典
    """
    print("\n" + "=" * 50)
    print("任务执行摘要")
    print("=" * 50)
    
    successful_tasks = [task_id for task_id, result in results.items() if result['success']]
    failed_tasks = [task_id for task_id, result in results.items() if not result['success']]
    
    print(f"总任务数: {len(results)}")
    print(f"成功任务: {len(successful_tasks)}")
    print(f"失败任务: {len(failed_tasks)}")
    
    if successful_tasks:
        print(f"成功任务ID: {sorted(successful_tasks)}")
    
    if failed_tasks:
        print(f"失败任务ID: {sorted(failed_tasks)}")
        print("\n失败任务详情:")
        for task_id in failed_tasks:
            print(f"  任务 {task_id}: {results[task_id]['message']}")

def save_results_to_file(results, filename="task_results.txt"):
    """
    将任务结果保存到文件
    
    参数:
        results: 任务执行结果字典
        filename: 输出文件名
    """
    with open(filename, 'w') as f:
        f.write(f"任务执行结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
        
        for task_id in sorted(results.keys()):
            result = results[task_id]
            status = "成功" if result['success'] else "失败"
            f.write(f"任务 {task_id}: {status} - {result['message']}\n")
    
    print(f"任务结果已保存到: {filename}")

# 主程序
if __name__ == "__main__":
    # 获取命令行参数
    if len(sys.argv) < 3:
        print("isSPA_1.1.12 by Zhang Lab at Institute of Biophysics, Chinese Academy of Sciences\nRun isSPA in parallel")
        print("Usage: python parallel_isSPA_runner.py <base config> <No. of tasks> [No. of GPUs]")
        print("e.g., python parallel_isSPA_runner.py config1 5")
        print("e.g., python parallel_isSPA_runner.py config1 10 4")
        sys.exit(1)
    
    base_config = sys.argv[1]
    
    try:
        num_tasks = int(sys.argv[2])
    except ValueError:
        print("错误: 任务数量必须是整数")
        sys.exit(1)
    
    max_workers = None
    if len(sys.argv) > 3:
        try:
            max_workers = int(sys.argv[3])
        except ValueError:
            print("错误: 最大并行数必须是整数")
            sys.exit(1)
    
    # 执行并行任务
    results = run_parallel_tasks(base_config, num_tasks, max_workers)
    
    # 打印摘要
    #print_summary(results)
    
    # 保存结果到文件
    #save_results_to_file(results)