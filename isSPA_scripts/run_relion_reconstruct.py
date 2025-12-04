import subprocess
import os

def get_max_mpi_processes():
    """获取可用的最大 MPI 进程数（逻辑 CPU 核心数 - 保留 1 个核心）"""
    max_cpus = os.cpu_count() or 1  # 默认为1防止除零错误
    return min(20, max_cpus/2)  

def run_relion_reconstruct(input_star, output_model, mpi_procs=None, ctf=False):
    """
    执行 relion_reconstruct_mpi
    :param input_star: 输入 STAR 文件路径
    :param output_model: 输出模型路径（.mrc）
    :param mpi_procs: 手动指定 MPI 进程数 (None 表示自动检测)
    """
    
    # 自动确定 MPI 进程数
    if mpi_procs is None:
        mpi_procs = int(get_max_mpi_processes())
    
    # 构造命令
    if ctf:
        cmd = [
        "mpirun",
        "-n", str(mpi_procs),
        "relion_reconstruct_mpi",
        "--i", input_star,
        "--o", output_model,
        "--ctf"
        ]
    else:
        cmd = [
            "mpirun",
            "-n", str(mpi_procs),
            "relion_reconstruct_mpi",
            "--i", input_star,
            "--o", output_model
        ]
    
    try:
        # 执行命令
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        # 检查输出
        if not os.path.isfile(output_model):
            raise RuntimeError(f"输出文件 {output_model} 未生成")
            
        print("重建成功！标准输出:")
        print(result.stdout[:1000] + "...")  # 截断长输出
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"MPI 执行失败 (退出码 {e.returncode}):")
        print("标准错误输出:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"运行时错误: {str(e)}")
        return False