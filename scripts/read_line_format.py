#!/usr/bin/env python3

def read_line_format(file):
    """自动检查文件头部的格式并读取其中的物理量"""
    count1 = 0 # 第一块诠释数据数量
    count2 = 0 # 第二块诠释数据数量
    block = 0 # 块数量
    intermediate = 0 # 间隔数
    p = 0 # 转换指示器
    spliter = 0 # 分隔符
    j = 0
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if line.split(): # 跳过空格
                if line.split()[0][0] != "_":
                    if p != 0:
                        if block > 0:
                            intermediate += 1
                            if intermediate == 1:
                                j = i
                            elif intermediate == 2:
                                if len(line.split('\t')) == count2:
                                    spliter = 1
                                elif len(line.split()) == count2:
                                    spliter = 0
                                else:
                                    spliter = 2
                                    print("The data spliter used in the star file requires manual confirmation!\n")
                                break
                        p = 0                      
                else:
                    if p != 1:
                        p = 1
                    if intermediate == 0:
                        count1 += 1
                        if block == 0:
                            block += 1
                    else:
                        count2 += 1
                        if block == 1:
                            block += 1
            # 当行数过大时，自动退出
            if i > 70:
                if len(line.split('\t')) == count1:
                    spliter = 1 # 以tab为分隔符
                elif len(line.split()) == count1:
                    spliter = 0 # 以空格为分隔符
                else:
                    spliter = 2
                    print("The data spliter used in the star file requires manual confirmation!\n")
                break
    if intermediate == 2:
        return count1, count2, j, i, spliter
    elif intermediate == 1:
        return count1, j, spliter
                
