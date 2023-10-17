[toc]

# grpc 服务端用法
启动方式:
```shell
yolo grpc 服务端配置文件路径
```

配置文件可以是 json 或者 yaml.
若为 json,格式如下,其中为默认值,实际使用写的时候只需写需要覆盖的值:
```json
{
    "detector_params": {
        "ckpt_path": "you pth file path",
        "thr": {
            "default": 0.3,
            "dog": 0.5
        },
        "change_label": {
            "wcgz_dxdk": "wcgz"
        },
        "device": "cuda:0",
        "nms": 0.5
    },
    "grpc_args": {
        "host": "0.0.0.0",
        "port": "7999",
        "max_workers": 1
    }
}
```
若为 yaml,格式如下,类似json,其中为默认值,实际使用写的时候只需写需要覆盖的值:
```yaml
detector_params:
  ckpt_path: you pth file path
  thr:
    default: 0.3
    dog: 0.5
  change_label:
    wcgz_dxdk: wcgz
  device: 'cuda:0'
  nms: 0.5
grpc_args:
  host: 0.0.0.0
  port: '7999'
  max_workers: 1
```
服务接口 proto 参见 [ultralytics/grpc_server/proto/dldetection.proto](ultralytics/grpc_server/proto/dldetection.proto)