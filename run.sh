#!/bin/bash

# 获取当前时间
current_time=$(date "+%Y-%m-%d %H:%M:%S")

# 提交当前代码文件到Git并将当前时间作为备注
git add .
git commit -m "提交代码文件 - $current_time"

# 获取最新的Git提交节点号
git_hash=$(git rev-parse HEAD)

# 启动main.py并传递节点号作为参数
nohup python main.py --gitNode "$git_hash" >/dev/null 2>&1 &

PID = $!
# 获取并输出进程号
echo "Python进程已启动，PID为: $PID"

echo "Process ID: $PID" > processRecord.txt
echo "Start Time: $current_time" >> processRecord.txt

