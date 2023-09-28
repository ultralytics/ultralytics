import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


class TritonRemoteModel:

    def __init__(self, url: str, endpoint: str, scheme: str, **kwargs):
        self.endpoint = endpoint
        self.url = url

        if scheme == 'http':
            self.triton_client: httpclient.InferenceServerClient = \
                httpclient.InferenceServerClient(
                    url=self.url,
                    verbose=False,
                    ssl=False,
                )
            self.InferInput = httpclient.InferInput
            self.InferRequestedOutput = httpclient.InferRequestedOutput
            model_config = self.triton_client.get_model_config(endpoint)
        else:
            self.triton_client: grpcclient.InferenceServerClient = \
                grpcclient.InferenceServerClient(
                    url=self.url,
                    verbose=False,
                    ssl=False,
                )
            self.InferInput = grpcclient.InferInput
            self.InferRequestedOutput = grpcclient.InferRequestedOutput
            model_config = self.triton_client.get_model_config(endpoint, as_json=True)['config']

        self.input_formats = [input['data_type'] for input in model_config['input']]
        converter_str_to_np_format = {'TYPE_FP32': np.float32, 'TYPE_FP16': np.float16, 'TYPE_UINT8': np.uint8}
        self.np_input_formats = [converter_str_to_np_format[x] for x in self.input_formats]

        self.input_names = [input['name'] for input in model_config['input']]
        self.input_shapes = [input['name'] for input in model_config['input']]
        self.output_names = [output['name'] for output in model_config['output']]

    def __call__(self, *inputs) -> dict:
        infer_inputs = []
        input_format = inputs[0].dtype
        for i, input in enumerate(inputs):
            if input.dtype != self.np_input_formats[i]:
                input = input.astype(self.np_input_formats[i])
            infer_input = self.InferInput(self.input_names[i], [*input.shape],
                                          self.input_formats[i].replace('TYPE_', ''))
            infer_input.set_data_from_numpy(input)
            infer_inputs.append(infer_input)

        infer_outputs = [self.InferRequestedOutput(output_name) for output_name in self.output_names]

        outputs = self.triton_client.infer(model_name=self.endpoint, inputs=infer_inputs, outputs=infer_outputs)

        return [outputs.as_numpy(output_name).astype(input_format) for output_name in self.output_names]
