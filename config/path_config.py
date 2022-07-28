import os
import sys

working_dir_path = os.getcwd().replace('config', '')
sys.path.append(
        working_dir_path
)

print(working_dir_path)
print(sys.path)