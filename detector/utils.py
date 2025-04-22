#– hỗ trợ (đo FPS, log, đọc file config).
import time
import yaml
import os


def measure_fps(interval=10):
    count = 0
    start_time = time.time()

    def update():
        nonlocal count, start_time
        count += 1
        if count >= interval:
            end_time = time.time()
            fps = count / (end_time - start_time)
            count = 0
            start_time = end_time
            return fps
        return None

    return update


def save_yaml(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f)


def load_yaml(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    return {}


def log_action(filepath, message):
    with open(filepath, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
