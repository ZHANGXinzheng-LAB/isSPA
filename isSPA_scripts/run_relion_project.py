import subprocess
import os

def run_relion_project(input_map, ang_star, output_dir, ctf=False):
    """用python脚本运行relion_project"""
    # 构造命令
    if ctf:
        cmd = [
            "relion_project",
            "--i", input_map,
            "--ang", ang_star,
            "--o", output_dir,
            "--ctf"  
            # 根据实际情况添加更多参数
        ]
    else:
        cmd = [
            relion_project_path,
            "--i", input_star,
            "--ang", ang_star,
            "--o", output_dir
        ]
    
    try:
        # 执行命令并等待完成
        result = subprocess.run(
            cmd,
            check=True,  # 如果返回非零状态码则抛出异常
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 打印输出日志
        print("relion_project 输出:")
        print(result.stdout)
        
        # 检查输出是否包含关键文件
        if not os.path.isfile(f'{output_dir}.mrcs'):
            raise RuntimeError("输出文件未生成，请检查参数")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，退出码: {e.returncode}")
        print("错误输出:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"运行时错误: {str(e)}")
        return False