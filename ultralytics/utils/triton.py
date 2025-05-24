# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import List
from urllib.parse import urlsplit

import numpy as np
from imb.triton import TritonClient


class TritonRemoteModel:
    def __init__(
        self,
        url: str,
        endpoint: str = "",
        scheme: str = "",
        max_batch_size: int = 0,
        fixed_batch: bool = True,
        is_async: bool = False,
        use_cuda_shm: bool = False,
        use_system_shm: bool = False,
        max_shm_regions: int = 0,
    ):
        """
        Initialize the TritonRemoteModel for interacting with a remote Triton Inference Server.

        Arguments may be provided individually or parsed from a collective 'url' argument of the form
        <scheme>://<netloc>/<endpoint>/<task_name>?a<arg_key>=<arg_value>&<arg_key>=<arg_value>

        Args:
            url (str): The URL of the Triton server.
            endpoint (str): The name of the model on the Triton server.
            scheme (str): The communication scheme ('http' or 'grpc').
            max_batch_size (int, optional): max batch size. Defaults to 0 (get value from triton config).
            fixed_batch (bool, optional): use fixed batch size, using padding for smaller batch. Defaults to True.
            is_async (bool, optional): async inference. Defaults to False.
            use_cuda_shm (bool, optional): use cuda shared memory. Defaults to False.
            use_system_shm (bool, optional): use system shared memory. Defaults to False.
            max_shm_regions (int, optional): max clients for shared memory. Will unregister old regions. Defaults to 0.

        Examples:
            >>> model = TritonRemoteModel(url="localhost:8000", endpoint="yolov8", scheme="http")
            >>> model = TritonRemoteModel(
            ...     url="http://localhost:8000/yolov8?use_system_shm=True&max_batch_size=8&max_shm_regions=1"
            ... )
        """
        triton_params = dict()
        if not endpoint and not scheme:  # Parse all args from URL string
            splits = urlsplit(url)
            endpoint = splits.path.strip("/").split("/", 1)[0]
            scheme = splits.scheme
            url = splits.netloc

            def convert_type(value: str):
                if value.isdigit():
                    return int(value)
                elif value in {"True", "False"}:
                    return eval(value)
                else:
                    return value

            for param in splits.query.split("&"):
                key, value = param.split("=")
                value = convert_type(value)
                triton_params[key] = value

        triton_params["max_batch_size"] = triton_params.get("max_batch_size", max_batch_size)
        triton_params["fixed_batch"] = triton_params.get("fixed_batch", fixed_batch)
        triton_params["is_async"] = triton_params.get("is_async", is_async)
        triton_params["use_cuda_shm"] = triton_params.get("use_cuda_shm", use_cuda_shm)
        triton_params["use_system_shm"] = triton_params.get("use_system_shm", use_system_shm)
        triton_params["max_shm_regions"] = triton_params.get("max_shm_regions", max_shm_regions)

        print("triton_params", triton_params)
        self.triton_client = TritonClient(url, endpoint, scheme=scheme, return_dict=False, **triton_params)

        self.metadata = None

    def __call__(self, *inputs: np.ndarray) -> List[np.ndarray]:
        """
        Call the model with the given inputs.

        Args:
            *inputs (np.ndarray): Input data to the model. Each array should match the expected shape and type
                for the corresponding model input.

        Returns:
            (List[np.ndarray]): Model outputs with the same dtype as the input. Each element in the list
                corresponds to one of the model's output tensors.

        Examples:
            >>> model = TritonRemoteModel(url="localhost:8000", endpoint="yolov8", scheme="http")
            >>> outputs = model(np.random.rand(1, 3, 640, 640).astype(np.float32))
        """
        return self.triton_client(*inputs)
