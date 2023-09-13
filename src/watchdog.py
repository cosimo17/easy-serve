import subprocess
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='config file of app')
    args = parser.parse_args()
    return args


def run_program(config_file):
    process = subprocess.Popen(['python', 'app.py', '-c', config_file])
    return process


def monitor(process, config_file, time_interval=30):
    while True:
        try:
            # check state of the process
            output = process.poll()
            if output is None:
                print("Program is running.")
            else:
                print("Program is not running. Restarting...")
                process = run_program(config_file)
        except Exception as e:
            print(f"Error monitoring program: {str(e)}")
        time.sleep(time_interval)


if __name__ == "__main__":
    args = get_args()
    config_file = args.config
    process = run_program(config_file)
    monitor(process, config_file)
