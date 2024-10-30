import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from aligo import Aligo, EMailConfig

if __name__ == '__main__':
    email_config = EMailConfig(
        email='gugupig@hotmail.com  ',
        # 自配邮箱
        user='kurros@88.com',
        password='YvnPh9Cgbu5qwy5v',
        host='smtp.88.com',
        port=465,
    )
    ali = Aligo(email=email_config)

import os
import time

# 初始化追踪已上传文件和等待上传文件的列表
uploaded_files = []
upload_waiting_list = []

# 自定义的文件处理事件类
class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        # 只处理 .pkl 文件
        if event.src_path.endswith(".pkl"):
            file_path = event.src_path
            if file_path not in uploaded_files and file_path not in upload_waiting_list:
                print(f"Detected new file: {file_path}")
                # 等待5秒，以防止文件正在写入过程中
                time.sleep(5)
                upload_waiting_list.append(file_path)  # 新文件加入等待上传列表
    
    def upload_file(self, file_path):
        try:
            # 模拟上传文件
            upload = ali.upload_file(file_path, parent_file_id='66d722d5d6a42108cc944fb984bff0776ebddcbc', drive_id='674838034')
            
            if upload:
                print(f"File {file_path} uploaded successfully.")
                uploaded_files.append(file_path)  # 标记为已上传
                os.remove(file_path)  # 删除本地文件
                print(f"File {file_path} deleted.")
            else:
                print(f"Failed to upload file: {file_path}")
                upload_waiting_list.append(file_path)  # 上传失败，重新加入等待列表

        except Exception as e:
            print(f"Exception occurred while uploading file {file_path}: {e}")
            upload_waiting_list.append(file_path)  # 异常情况下，重新加入等待列表

def monitor_directory(directory_to_watch):
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, directory_to_watch, recursive=False)
    observer.start()
    print(f"Started monitoring {directory_to_watch}...")

    try:
        while True:
            if upload_waiting_list:
                # 处理等待上传的文件
                file_to_upload = upload_waiting_list.pop(0)
                event_handler.upload_file(file_to_upload)
            time.sleep(1)  # 控制上传频率，防止占用过多资源
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    # 设置要监控的目录
    directory_to_watch = "/root/autodl-tmp/llama_exp/pkl"
    monitor_directory(directory_to_watch)
