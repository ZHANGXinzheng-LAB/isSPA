import pandas as pd

def parse_star(file_path, block="data_particles"):
    """解析RELION STAR文件"""
    data = []
    current_data = None
    columns = []
    
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            
            if line.startswith(block):
                current_data = []
            elif line.startswith("loop_"):
                columns = []
            elif line.startswith("_"):
                col = line.split()[0][4:]
                columns.append(col)
            elif len(line) > 0 and current_data is not None:
                current_data.append(line.split())
                
    return pd.DataFrame(current_data, columns=columns)