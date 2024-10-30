import subprocess
import sys
import os

def run_script(script_path):
    process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        output = process.stdout.readline()
        error_output = process.stderr.readline()

        if output:
            print(output.strip())
        
        if error_output:
            print(error_output.strip())

            # 检查是否存在自定义 OOM 错误
            if "CustomOOMError" in error_output:
                print("Detected custom OOM error. Restarting the script...")

                # 终止当前脚本进程
                process.terminate()

                # 重启内核或环境
                restart_kernel()

                # 重新运行脚本
                run_script(script_path)

        if process.poll() is not None:
            break

    return process.poll()

def restart_kernel():
    # 重启 Python 内核
    print("Restarting the Python kernel...")
    
    # 如果是在Jupyter Notebook中执行：
    if "ipykernel" in sys.modules:
        from IPython import get_ipython
        get_ipython().kernel.do_shutdown(True)  # 重启内核
    else:
        # 如果是本地执行：
        os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == "__main__":
    script_to_monitor = "run_llama.py"  # 替换为你的目标脚本路径
    run_script(script_to_monitor)
