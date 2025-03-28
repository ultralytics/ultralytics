# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from collections import defaultdict
from typing import Dict, List
from urllib.parse import urlsplit

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonclient.utils.cuda_shared_memory as cudashm
from google.protobuf.json_format import MessageToJson
from tritonclient import utils


class TritonRemoteModel:
    """
    Client for interacting with a remote Triton Inference Server model.

    This class provides a convenient interface for sending inference requests to a Triton Inference Server
    and processing the responses.

    Attributes:
        endpoint (str): The name of the model on the Triton server.
        url (str): The URL of the Triton server.
        triton_client: The Triton client (either HTTP or gRPC).
        InferInput: The input class for the Triton client.
        InferRequestedOutput: The output request class for the Triton client.
        input_formats (List[str]): The data types of the model inputs.
        np_input_formats (List[type]): The numpy data types of the model inputs.
        input_names (List[str]): The names of the model inputs.
        output_names (List[str]): The names of the model outputs.
        metadata: The metadata associated with the model.

    Examples:
        Initialize a Triton client with HTTP
        >>> model = TritonRemoteModel(url="localhost:8000", endpoint="yolov8", scheme="http")
        Make inference with numpy arrays
        >>> outputs = model(np.random.rand(1, 3, 640, 640).astype(np.float32))
    """

    def __init__(
        self,
        url: str,
        endpoint: str = "",
        scheme: str = "",
        max_batch_size: int = 1,
        cuda_shm: bool = False,
        shm_region_prefix: str = "ultralytics_shm_",
    ):
        """
        Initialize the TritonRemoteModel.

        Arguments may be provided individually or parsed from a collective 'url' argument of the form
        <scheme>://<netloc>/<endpoint>/<task_name>

        Args:
            url (str): The URL of the Triton server.
            endpoint (str): The name of the model on the Triton server.
            scheme (str): The communication scheme ('http' or 'grpc').
            max_batch_size (int): The maximum batch size for the model.
            cuda_shm (bool): Whether to use CUDA shared memory for inference.
            shm_region_prefix (str): The prefix for the shared memory regions names.
        """
        parameters = dict()
        if not endpoint and not scheme:  # Parse all args from URL string
            splits = urlsplit(url)
            endpoint = splits.path.strip("/").split("/")[0]
            scheme = splits.scheme
            url = splits.netloc

            def convert_type(value):
                if value.isdigit():
                    return int(value)
                elif value == "True":
                    return True
                elif value == "False":
                    return False
                else:
                    return value

            for param in splits.query.split("&"):
                key, value = param.split("=")
                value = convert_type(value)
                parameters[key] = value

        max_batch_size = parameters.get("max_batch_size", max_batch_size)
        cuda_shm = parameters.get("cuda_shm", cuda_shm)
        shm_region_prefix = parameters.get("shm_region_prefix", shm_region_prefix)

        self.model_name = endpoint
        self.url = url
        self.scheme = scheme

        self.cuda_shm = cuda_shm
        self.shm_region_prefix = shm_region_prefix
        self.static_batch = False

        # Choose the Triton client based on the communication scheme
        self.client_type = httpclient if scheme == "http" else grpcclient
        self.triton_client = self.client_type.InferenceServerClient(url=self.url, verbose=False, ssl=False)

        self._load_model_config(max_batch_size)
        self._create_input_sample()

        self.input_shm_handles = [None for _ in range(len(self.inputs_names))]
        self.output_shm_handles = [None for _ in range(len(self.outputs_names))]

        if self.cuda_shm:
            self._fill_output_dynamic_axis()
            self._create_input_output_shm_handles()
            self._register_cuda_shm_regions()

        self.class_names = [f"class_{i}" for i in range(self.outputs_shapes[0][1] - 4)]

    def _create_input_sample(self):
        has_dynamic_shapes = any(-1 in input_shape for input_shape in self.inputs_shapes)
        assert not has_dynamic_shapes, "dynamic input shapes not supported"
        self.sample_inputs = []
        for input_shape, np_input_format in zip(self.inputs_shapes, self.np_inputs_dtypes):
            self.sample_inputs.append(np.ones(input_shape).astype(np_input_format))

    def _fill_output_dynamic_axis(self):
        has_dynamic_shapes = any(-1 in output_shape for output_shape in self.outputs_shapes)
        if has_dynamic_shapes:
            start_cuda_shm_flag = self.cuda_shm
            self.cuda_shm = False
            outputs = self.__call__(*self.sample_inputs)
            self.outputs_shapes = [list(output.shape) for output in outputs]
            self.cuda_shm = start_cuda_shm_flag

    def pad_batch(self, batch: np.ndarray):
        padding_size = self.max_batch_size - batch.shape[0]
        if padding_size > 0:
            pad = np.zeros([padding_size, *batch.shape[1:]], dtype=batch.dtype)
            batch = np.concatenate((batch, pad), axis=0)
        return batch, padding_size

    def split_on_batches(self, input_data: np.ndarray):
        batches = []
        paddings = []
        for i in range(0, len(input_data), self.max_batch_size):
            batch = input_data[i : i + self.max_batch_size]
            batches.append(batch)
            paddings.append(0)

        if self.static_batch:
            batches[-1], paddings[-1] = self.pad_batch(batches[-1])

        return batches, paddings

    def _create_batches(self, *inputs_data: np.ndarray):
        inputs_batches = dict()
        paddings = []
        for input_data, np_format, input_name in zip(inputs_data, self.np_inputs_dtypes, self.inputs_names):
            input_data = input_data.astype(np_format)
            input_batches, input_paddings = self.split_on_batches(input_data)
            if paddings == []:
                paddings = input_paddings
            inputs_batches[input_name] = input_batches
        return inputs_batches, paddings

    def _parse_io_params(self, io_params: List[Dict]):
        triton_dtypes = []
        np_dtypes = []
        shapes = []
        names = []
        for params in io_params:
            triton_dtypes.append(params["data_type"].replace("TYPE_", ""))
            np_dtypes.append(utils.triton_to_np_dtype(triton_dtypes[-1]))
            shapes.append([int(x) for x in params["dims"]])
            names.append(params["name"])

        return triton_dtypes, np_dtypes, shapes, names

    @staticmethod
    def _insert_batch_size_to_shapes(shapes: List[List], insert_batch_size: int):
        return [[insert_batch_size] + shape for shape in shapes]

    def _load_model_config(self, user_max_batch_size: int):
        if self.scheme == "grpc":
            config = self.triton_client.get_model_config(self.model_name, as_json=True)
            config = config["config"]
        else:
            config = self.triton_client.get_model_config(self.model_name)

        self.triton_inputs_dtypes, self.np_inputs_dtypes, self.inputs_shapes, self.inputs_names = self._parse_io_params(
            config["input"]
        )

        self.triton_outputs_dtypes, self.np_outputs_dtypes, self.outputs_shapes, self.outputs_names = (
            self._parse_io_params(config["output"])
        )

        not_support_dynamic_batch = config["max_batch_size"] == 0
        if not_support_dynamic_batch:
            # use batch size from config
            self.static_batch = True
            self.max_batch_size = config["input"][0]["dims"][0]
        else:
            # user can decrease max_batch_size from config
            self.max_batch_size = min(config["max_batch_size"], user_max_batch_size)
            # in config's shape has no batch size
            self.inputs_shapes = self._insert_batch_size_to_shapes(self.inputs_shapes, self.max_batch_size)
            self.outputs_shapes = self._insert_batch_size_to_shapes(self.outputs_shapes, self.max_batch_size)
        self.metadata = eval(config.get("parameters", {}).get("metadata", {}).get("string_value", "None"))

    def _generate_shm_name(self, ioname):
        return f"{self.shm_region_prefix}_{ioname}"

    def _register_cuda_shm_regions(self):
        if self.scheme == "grpc":
            regions_statuses = self.triton_client.get_cuda_shared_memory_status(as_json=True)["regions"]
            registrated_regions = [region for region in regions_statuses.keys()]
        else:
            regions_statuses = self.triton_client.get_cuda_shared_memory_status()
            registrated_regions = [region["name"] for region in regions_statuses]
        for shm_handle in self.input_shm_handles + self.output_shm_handles:
            # unregister region with same name, because byte_size may be changed
            if shm_handle._triton_shm_name in registrated_regions:
                self.triton_client.unregister_cuda_shared_memory(shm_handle._triton_shm_name)
            self.triton_client.register_cuda_shared_memory(
                shm_handle._triton_shm_name, cudashm.get_raw_handle(shm_handle), 0, shm_handle._byte_size
            )

    def _create_shm_handle(self, shape, dtype, name):
        """Create a CUDA shared memory handle for a given shape, dtype, and name."""
        byte_size = int(np.prod(shape) * np.dtype(dtype).itemsize)
        shm_name = self._generate_shm_name(name)
        return cudashm.create_shared_memory_region(shm_name, byte_size, 0)

    def _create_shm_handles_for_io(self, shapes, dtypes, names):
        """Helper method to create SHM handles for inputs or outputs."""
        return [self._create_shm_handle(shape, dtype, name) for shape, dtype, name in zip(shapes, dtypes, names)]

    def _create_input_output_shm_handles(self):
        """Create CUDA shared memory handles for both inputs and outputs."""
        self.input_shm_handles = self._create_shm_handles_for_io(
            self.inputs_shapes, self.np_inputs_dtypes, self.inputs_names
        )
        self.output_shm_handles = self._create_shm_handles_for_io(
            self.outputs_shapes, self.np_outputs_dtypes, self.outputs_names
        )

    def _create_triton_input(self, input_data: np.ndarray, input_name: str, config_input_format: str, shm_handle=None):
        infer_input = self.client_type.InferInput(input_name, input_data.shape, config_input_format)
        if self.cuda_shm:
            cudashm.set_shared_memory_region(shm_handle, [input_data])
            infer_input.set_shared_memory(shm_handle._triton_shm_name, shm_handle._byte_size)
        else:
            infer_input.set_data_from_numpy(input_data)
        return infer_input

    def _create_triton_output(self, output_name: str, binary: bool = True, shm_handle=None):
        if self.scheme == "grpc":
            infer_output = self.client_type.InferRequestedOutput(output_name)
        else:
            infer_output = self.client_type.InferRequestedOutput(output_name, binary_data=binary)
        if self.cuda_shm:
            infer_output.set_shared_memory(shm_handle._triton_shm_name, shm_handle._byte_size)
        return infer_output

    def _postprocess_triton_result(self, triton_response, padding_size):
        result = dict()
        for output_name, shm_op_handle in zip(self.outputs_names, self.output_shm_handles):
            if self.cuda_shm:
                if self.scheme == "grpc":
                    # output = triton_response.get_output(output_name, as_json=True) # WARN: bug in tritonclient library, return None
                    output = json.loads(MessageToJson(triton_response.get_output(output_name)))
                else:
                    output = triton_response.get_output(output_name)
                result[output_name] = cudashm.get_contents_as_numpy(
                    shm_op_handle,
                    utils.triton_to_np_dtype(output["datatype"]),
                    [int(x) for x in output["shape"]],
                )
            else:
                result[output_name] = triton_response.as_numpy(output_name)

            if padding_size != 0:
                result[output_name] = result[output_name][:-padding_size]

        return result

    def __call__(self, *inputs: np.ndarray):
        """
        Call the model with the given inputs.

        Args:
            *inputs (np.ndarray): Input data to the model.

        Returns:
            (List[np.ndarray]): Model outputs with the same dtype as the input.
        """
        assert len(inputs) == len(self.inputs_names), "inputs number is not equal to model inputs"
        inputs_batches, batches_paddings = self._create_batches(*inputs)

        result = defaultdict(list)
        count_batches = len(next(iter(inputs_batches.values())))

        for i_batch in range(count_batches):
            triton_inputs = []
            for input_name, config_input_format, shm_ip_handle in zip(
                self.inputs_names, self.triton_inputs_dtypes, self.input_shm_handles
            ):
                triton_input = self._create_triton_input(
                    inputs_batches[input_name][i_batch], input_name, config_input_format, shm_ip_handle
                )
                triton_inputs.append(triton_input)

            triton_outputs = []
            for output_name, shm_op_handle in zip(self.outputs_names, self.output_shm_handles):
                triton_output = self._create_triton_output(output_name, binary=True, shm_handle=shm_op_handle)
                triton_outputs.append(triton_output)

            triton_response = self.triton_client.infer(
                model_name=self.model_name, inputs=triton_inputs, outputs=triton_outputs
            )

            batch_result = self._postprocess_triton_result(triton_response, batches_paddings[i_batch])

            for output_name, output_value in batch_result.items():
                result[output_name].append(output_value)

        return [np.concatenate(result[output_name]) for output_name in self.outputs_names]
