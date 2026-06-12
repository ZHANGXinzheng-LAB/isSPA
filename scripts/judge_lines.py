#!/usr/bin/env python3


def judge_lines(inline, n=0):
    """通过_rln判断star文件中数据从哪一行开始"""
    trys = 50 # 尝试行数
    intarget = -1
    if n > 0:
        for i in range(n, trys):
            if inline[i].split():
                if inline[i].split()[0][0] != "_":
                    if intarget == 1:
                        return i
                        break
                    else:
                        continue
                else: #inline[i].split()[0][0] == "_":
                    intarget = 1
    else:
        for i in range(trys):
            if inline[i].split():
                if inline[i].split()[0][0] != "_":
                    if intarget == 1:
                        return i
                        break
                    else:
                        continue
                else: #inline[i].split()[0][0] == "_":
                    intarget = 1