###
 # @Author: captainfffsama
 # @Date: 2023-10-24 19:00:49
 # @LastEditors: captainfffsama tuanzhangsama@outlook.com
 # @LastEditTime: 2023-10-24 19:01:59
 # @FilePath: /ultralytics/ultralytics/grpc_server/proto/build_proto.sh
 # @Description:
###
python -m grpc_tools.protoc -I ./ --proto_path=./dldetection.proto --python_out=./new/ --pyi_out=./new/ --grpc_python_out=./new/ dldetection.proto