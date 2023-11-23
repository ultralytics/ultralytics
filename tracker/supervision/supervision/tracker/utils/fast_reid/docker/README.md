# Use the container

```shell script
cd docker/
# Build:
docker build -t=fastreid:v0 .
# Launch (requires GPUs)
nvidia-docker run -v server_path:docker_path --name=fastreid --net=host --ipc=host -it fastreid:v0 /bin/sh
```

## Install new dependencies

Add the following to `Dockerfile` to make persist changes.
```shell script
RUN sudo apt-get update && sudo apt-get install -y vim
```

Or run them in the container to make temporary changes.

## A more complete docker container

If you want to use a complete docker container which contains many useful tools, you can check my development environment [Dockerfile](https://github.com/L1aoXingyu/fastreid_docker)