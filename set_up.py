import sys
import cx_Freeze
from cx_Freeze import setup, Executable
import google

application = 'tfApp'
includes = []
packages = ['os', 'sys','cv2','tensorflow','numpy','time','random','math','google']
excludes = []
include_files = []
bin_path_includes = ["/Users/liuyongjie/anaconda2/envs/py3.7/lib/python3.6/site-packages/tensorflow",
                    "/Users/liuyongjie/anaconda2/envs/py3.7/lib/python3.6/site-packages/google"]

bin_includes = ["/Users/liuyongjie/anaconda2/envs/py3.7/lib/python3.6/site-packages/tensorflow",
                    "/Users/liuyongjie/anaconda2/envs/py3.7/lib/python3.6/site-packages/google"]

append_path = [
        "/Users/liuyongjie/anaconda2/envs/py3.7/lib",
        "/Users/liuyongjie/anaconda2/envs/py3.7/bin",
        "/Users/liuyongjie/anaconda2/envs/py3.7/lib/python3.6/site-packages/tensorflow",
        "/Users/liuyongjie/anaconda2/envs/py3.7/lib/python3.6/site-packages/tensorflow/include/google",
        "/Users/liuyongjie/anaconda2/envs/py3.7/lib/python3.6/site-packages/google"]

path = sys.path
for item in append_path:
    path.append(item)

print (path)
build_exe_options = {
    'includes': includes,
    'include_files': include_files,
    'packages': packages,
    'excludes': excludes,
    'include_files': include_files,
    "bin_includes": bin_includes,
    "bin_path_includes": bin_path_includes,
    'path': path
}

exe = [
    Executable(script='run_this.py', base=None, targetName=application)
]


setup(name=application,
      version='1.0.0',
      description='Simple tf application',
      options={'build_exe': build_exe_options},
      executables=exe)
