#!/bin/bash

# 设置最终训练数据保存的路径
save_path="/path/to/save/data"

# 运行Python脚本并传递变量
python data_prepare/data_process.py --save_path "$save_path"