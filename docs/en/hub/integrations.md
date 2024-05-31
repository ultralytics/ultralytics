# Ultralytics HUB Integrations

Learn about [Ultralytics HUB](https://bit.ly/ultralytics_hub) integrations with various platforms and formats.

## Datasets

Seamlessly import your datasets in [Ultralytics HUB](https://bit.ly/ultralytics_hub) for [model training](./models.md#train-model).

After a dataset is imported in [Ultralytics HUB](https://bit.ly/ultralytics_hub), you can [train a model](./models.md#train-model) on your dataset just like you would using the [Ultralytics HUB](https://bit.ly/ultralytics_hub) datasets.

### Roboflow

You can easily filter the Roboflow datasets on the [Ultralytics HUB](https://bit.ly/ultralytics_hub) [Datasets](https://hub.ultralytics.com/datasets) page.

![Ultralytics HUB screenshot of the Datasets page with Roboflow provider filter](./images/hub_roboflow_1.jpg)

[Ultralytics HUB](https://bit.ly/ultralytics_hub) supports two types of integrations with Roboflow, [Universe](#universe) and [Workspace](#workspace).

#### Universe

##### Import

##### Remove

#### Workspace

##### Import

Navigate to the [Integrations](https://hub.ultralytics.com/settings?tab=integrations) page by clicking on the **Integrations** button in the sidebar.

Type your Roboflow Workspace private API key and click on the **Add** button.

??? tip "Tip"

    You can click on the **Get my API key** button which will redirect you to the settings of your Roboflow Workspace from where you can obtain your private API key.

![Ultralytics HUB screenshot of the Integrations page with an arrow pointing to the Integrations button in the sidebar and one to the Add button](./images/hub_roboflow_workspace_import_1.jpg)

This will connect your [Ultralytics HUB](https://bit.ly/ultralytics_hub) account with your Roboflow Workspace and make your Roboflow datasets available in [Ultralytics HUB](https://bit.ly/ultralytics_hub).

![Ultralytics HUB screenshot of the Integrations page with an arrow pointing to one of the connected workspaces](./images/hub_roboflow_workspace_import_2.jpg)

Next, [train a model](./models.md#train-model) on your dataset.

![Ultralytics HUB screenshot of the Dataset page of a Roboflow Workspace dataset with an arrow pointing to the Train Model button](./images/hub_roboflow_workspace_import_3.jpg)

##### Remove

Navigate to the [Integrations](https://hub.ultralytics.com/settings?tab=integrations) page by clicking on the **Integrations** button in the sidebar and click on the **Unlink** button of the Roboflow Workspace you want to remove.

![Ultralytics HUB screenshot of the Integrations page with an arrow pointing the Unlink button of one of the connected workspaces](./images/hub_roboflow_workspace_remove_1.jpg)

??? tip "Tip"

    You can remove a connected Roboflow Workspace directly from the Dataset page of one of the datasets from your Roboflow Workspace.

    ![Ultralytics HUB screenshot of the Dataset page of a Roboflow Workspace dataset with an arrow pointing to the remove option](./images/hub_roboflow_workspace_remove_2.jpg)

??? tip "Tip"

    You can remove a connected Roboflow Workspace directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Remove option of one of the Roboflow Workspace datasets](./images/hub_roboflow_workspace_remove_3.jpg)

## Models

### Exports

After you [train a model](./models.md#train-model), you can [export it](./models.md#deploy-model) to 13 different formats, including ONNX, OpenVINO, CoreML, TensorFlow, Paddle and many others.

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with an arrow pointing to the Export card and all formats exported](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_deploy_model_1.jpg)

The available export formats are presented in the table below.

| Format                                            | `format` Argument | Model                     | Metadata | Arguments                                                            |
| ------------------------------------------------- | ----------------- | ------------------------- | -------- | -------------------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                   | -                 | `yolov8n.pt`              | ✅       | -                                                                    |
| [TorchScript](../integrations/torchscript.md)     | `torchscript`     | `yolov8n.torchscript`     | ✅       | `imgsz`, `optimize`, `batch`                                         |
| [ONNX](../integrations/onnx.md)                   | `onnx`            | `yolov8n.onnx`            | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `batch`             |
| [OpenVINO](../integrations/openvino.md)           | `openvino`        | `yolov8n_openvino_model/` | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TensorRT](../integrations/tensorrt.md)           | `engine`          | `yolov8n.engine`          | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `batch` |
| [CoreML](../integrations/coreml.md)               | `coreml`          | `yolov8n.mlpackage`       | ✅       | `imgsz`, `half`, `int8`, `nms`, `batch`                              |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model`     | `yolov8n_saved_model/`    | ✅       | `imgsz`, `keras`, `int8`, `batch`                                    |
| [TF GraphDef](../integrations/tf-graphdef.md)     | `pb`              | `yolov8n.pb`              | ❌       | `imgsz`, `batch`                                                     |
| [TF Lite](../integrations/tflite.md)              | `tflite`          | `yolov8n.tflite`          | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TF Edge TPU](../integrations/edge-tpu.md)        | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅       | `imgsz`, `batch`                                                     |
| [TF.js](../integrations/tfjs.md)                  | `tfjs`            | `yolov8n_web_model/`      | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [PaddlePaddle](../integrations/paddlepaddle.md)   | `paddle`          | `yolov8n_paddle_model/`   | ✅       | `imgsz`, `batch`                                                     |
| [NCNN](../integrations/ncnn.md)                   | `ncnn`            | `yolov8n_ncnn_model/`     | ✅       | `imgsz`, `half`, `batch`                                             |
