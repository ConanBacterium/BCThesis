import os

def find_py_files(directory):
    py_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def mlflow_log_files_in_dir(directory, mlflow):
    for path in find_py_files(directory):
        mlflow.log_artifact(path)
