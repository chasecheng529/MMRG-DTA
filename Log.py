import datetime
from Arguments import argparser
import os
class LogModule:
    def __init__(self):
        self.FLAGS = argparser()    
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("[%m_%d][%H_%M_%S]")  # 格式化日期时间字符串
        os.makedirs(self.FLAGS.logDir, exist_ok=True)
        self.LogFileName = f"{self.FLAGS.logDir}{formatted_datetime}.txt"  # 构建文件名
        self.InitLogModule()

    def InitLogModule(self):
        with open(self.LogFileName, "a") as file:
            # 写入一些内容到文件中
            file.write("====================Init Model====================\n")
            for argName, argValue in vars(self.FLAGS).items():
                logString = f"{argName}: {argValue}"
                file.write(logString+'\n')
            file.write("====================Logging====================\n")
        print(f"已创建日志记录文件: {self.LogFileName}")

    def LogInfoWithArgs(self, logType, **kWargs):
        if self.FLAGS.logEnable == 0:
            return
        if self.LogFileName == "":
            self.InitLogModule()
        logString = f"{logType}: "
        for key, value in kWargs.items():
            logString += f"{key}:{value}  "
        with open(self.LogFileName, "a") as file:
            # 写入一些内容到文件中
            file.write(logString+'\n')
    def LogInfoWithStr(self, logType, str):
        if self.FLAGS.logEnable == 0:
            return
        if self.LogFileName == "":
            self.InitLogModule()
        logString = f"{logType}: {str}"
        with open(self.LogFileName, "a") as file:
            # 写入一些内容到文件中
            file.write(logString+'\n')

