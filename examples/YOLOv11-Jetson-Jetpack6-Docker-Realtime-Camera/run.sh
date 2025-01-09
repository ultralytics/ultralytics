# Set up device and display variables
V4L2_DEVICES=""
for i in $(seq 0 9); do
    if [ -e "/dev/video$i" ]; then
        V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i"
    fi
done

DISPLAY_DEVICE=""
if [ -n "$DISPLAY" ]; then
    xhost +si:localuser:root
    DISPLAY_DEVICE=" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix"
fi

# change WORKSPACE_DIR to your directory that you want to mount to the docker container
WORKSPACE_DIR="/home/user/YOLOv11-Jetson-Jetpack6-Docker-Realtime-Camera"  

# Run docker
sudo docker run -it --rm \
    --runtime nvidia \
    --ipc=host \
    -v $WORKSPACE_DIR:/ultralytics/public/ \
    $DISPLAY_DEVICE $V4L2_DEVICES \
    opencv-jetson-ultralytics:6.1