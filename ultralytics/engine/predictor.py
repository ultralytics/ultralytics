# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

1|# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
2|"""
3|Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.
4|
5|Usage - sources:
6|    $ yolo mode=predict model=yolo26n.pt source=0                               # webcam
7|                                                img.jpg                         # image
8|                                                vid.mp4                         # video
9|                                                screen                          # screenshot
10|                                                path/                           # directory
11|                                                list.txt                        # list of images
12|                                                list.streams                    # list of streams
13|                                                'path/*.jpg'                    # glob
14|                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
15|                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream
16|
17|Usage - formats:
18|    $ yolo mode=predict model=yolo26n.pt                 # PyTorch
19|                              yolo26n.torchscript        # TorchScript
20|                              yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
21|                              yolo26n_openvino_model     # OpenVINO
22|                              yolo26n.engine             # TensorRT
23|                              yolo26n.mlpackage          # CoreML (macOS-only)
24|                              yolo26n_saved_model        # TensorFlow SavedModel
25|                              yolo26n.pb                 # TensorFlow GraphDef
26|                              yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
27|                              yolo26n_paddle_model       # PaddlePaddle
28|                              yolo26n.mnn                # MNN
29|                              yolo26n_ncnn_model         # NCNN
30|                              yolo26n_imx_model          # Sony IMX
31|                              yolo26n_rknn_model         # Rockchip RKNN
32|                              yolo26n_executorch_model   # PyTorch Executorch
33|                              yolo26n_axelera_model      # Axelera AI
34|                              yolo26n_deepx_model        # DEEPX
35|                              yolo26n_qnn.onnx           # Qualcomm QNN
36|                              yolo26n.tflite             # LiteRT
37|                              yolo26n_ascend_model       # Huawei Ascend
38|"""
39|
40|from __future__ import annotations
41|
42|import platform
43|import re
44|import threading
45|from pathlib import Path
46|from typing import Any, Callable
47|
48|import cv2
49|import numpy as np
50|import torch
51|
52|from ultralytics.cfg import get_cfg, get_save_dir
53|from ultralytics.data import load_inference_source
54|from ultralytics.data.augment import LetterBox
55|from ultralytics.nn.autobackend import AutoBackend
56|from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
57|from ultralytics.utils.checks import check_imgsz, check_imshow
58|from ultralytics.utils.plotting import class_activation_map
59|from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode
60|
61|STREAM_WARNING = """
62|Inference results will accumulate in RAM unless `stream=True` is passed, which can cause out-of-memory errors for large
63|sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.
64|
65|Example:
66|    results = model(source=..., stream=True)  # generator of Results objects
67|    for r in results:
68|        boxes = r.boxes  # Boxes object for bbox outputs
69|        masks = r.masks  # Masks object for segment masks outputs
70|        probs = r.probs  # Class probabilities for classification outputs
71|"""
72|
73|
74|class BasePredictor:
75|    """A base class for creating predictors.
76|
77|    This class provides the foundation for prediction functionality, handling model setup, inference, and result
78|    processing across various input sources.
79|
80|    Attributes:
81|        args (SimpleNamespace): Configuration for the predictor.
82|        save_dir (Path): Directory to save results.
83|        done_warmup (bool): Whether the predictor has finished setup.
84|        model (torch.nn.Module): Model used for prediction.
85|        data (str | Path | None): Copy of args.data, the dataset YAML AutoBackend falls back to for class names.
86|        device (torch.device): Device used for prediction.
87|        dataset (Dataset): Dataset used for prediction.
88|        vid_writer (dict[Path, cv2.VideoWriter]): Dictionary of {save_path: video_writer} for saving video output.
89|        plotted_img (np.ndarray): Last plotted image.
90|        source_type (SimpleNamespace): Type of input source.
91|        seen (int): Number of images processed.
92|        speed (dict[str, float] | None): Per-image preprocess, inference and postprocess times in ms, once run.
93|        pixels (int | None): Mean per-image inference area in pixels, once a run completes.
94|        windows (list[str]): List of window names for visualization.
95|        batch (tuple): Current batch data.
96|        results (list[Any]): Current batch results.
97|        transforms (Callable): Image transforms for classification.
98|        callbacks (dict[str, list[Callable]]): Callback functions for different events.
99|        txt_path (Path): Path to save text results.
100|        _lock (threading.Lock): Lock for thread-safe inference.
101|
102|    Methods:
103|        preprocess: Prepare input image before inference.
104|        inference: Run inference on a given image.
105|        postprocess: Process raw predictions into structured results.
106|        predict_cli: Run prediction for command line interface.
107|        setup_source: Set up input source and inference mode.
108|        stream_inference: Stream inference on input source.
109|        setup_model: Initialize and configure the model.
110|        write_results: Write inference results to files.
111|        save_predicted_images: Save prediction visualizations.
112|        show: Display results in a window.
113|        run_callbacks: Execute registered callbacks for an event.
114|        add_callback: Register a new callback function.
115|    """
116|
117|    def __init__(
118|        self,
119|        cfg=DEFAULT_CFG,
120|        overrides: dict[str, Any] | None = None,
121|        _callbacks: dict | None = None,
122|    ):
123|        """Initialize the BasePredictor class.
124|
125|        Args:
126|            cfg (str | Path | dict | SimpleNamespace): Path to a configuration file or a configuration dictionary.
127|            overrides (dict, optional): Configuration overrides.
128|            _callbacks (dict, optional): Dictionary of callback functions.
129|        """
130|        self.args = get_cfg(cfg, overrides)
131|        self.save_dir = get_save_dir(self.args)
132|        if self.args.conf is None:
133|            self.args.conf = 0.25  # default conf=0.25
134|        self.done_warmup = False
135|        if self.args.show:
136|            self.args.show = check_imshow(warn=True)
137|
138|        # Usable if setup is done
139|        self.model = None
140|        self.data = self.args.data
141|        self.imgsz = None
142|        self.device = None
143|        self.dataset = None
144|        self.vid_writer = {}  # dict of {save_path: video_writer, ...}
145|        self.plotted_img = None
146|        self.source_type = None
147|        self.seen = 0
148|        self.speed = None  # per-image speeds, set once a run completes
149|        self.pixels = None  # mean per-image inference area, set once a run completes
150|        self.windows = []
151|        self.screen = None  # cached screen resolution (width, height) for show=True scaling
152|        self.batch = None
153|        self.results = None
154|        self.transforms = None
155|        self.callbacks = _callbacks or callbacks.get_default_callbacks()
156|        self.txt_path = None
157|        self._lock = threading.Lock()  # for automatic thread-safe inference
158|        callbacks.add_integration_callbacks(self)
159|
160|    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
161|        """Prepare input image before inference.
162|
163|        Args:
164|            im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list.
165|
166|        Returns:
167|            (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
168|        """
169|        not_tensor = not isinstance(im, torch.Tensor)
170|        if not_tensor:
171|            im = np.stack(self.pre_transform(im))
172|            if im.shape[-1] == 3:
173|                im = im[..., ::-1]  # BGR to RGB
174|            im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
175|            im = np.ascontiguousarray(im)  # contiguous
176|            im = torch.from_numpy(im)
177|
178|        im = im.to(self.device)
179|        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
180|        if not_tensor:
181|            im /= 255  # 0 - 255 to 0.0 - 1.0
182|        return im
183|
184|    def inference(self, im: torch.Tensor, *args, **kwargs):
185|        """Run inference on a given image using the specified model and arguments."""
186|        skip = self.source_type.tensor or self.args.augment or self.args.embed  # unsupported with activation maps
187|        if self.args.visualize and getattr(self.model, "base_model", True) and not skip:
188|            return class_activation_map(
189|                self.model,
190|                im,
191|                self.batch[0],
192|                self.save_dir,
193|                *args,
194|                conf=self.args.conf,
195|                classes=self.args.classes,
196|                **kwargs,
197|            )
198|        return self.model(im, *args, augment=self.args.augment, embed=self.args.embed, **kwargs)
199|
200|    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
201|        """Pre-transform input image before inference.
202|
203|        Args:
204|            im (list[np.ndarray]): List of images with shape [(H, W, 3) x N].
205|
206|        Returns:
207|            (list[np.ndarray]): List of transformed images.
208|        """
209|        same_shapes = len({x.shape for x in im}) == 1
210|        letterbox = LetterBox(
211|            self.imgsz,
212|            auto=same_shapes
213|            and self.args.rect
214|            and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) and self.model.format != "imx")),
215|            stride=self.model.stride,
216|        )
217|        return [letterbox(image=x) for x in im]
218|
219|    def postprocess(self, preds, img, orig_imgs):
220|        """Post-process predictions for an image and return them."""
221|        return preds
222|
223|    def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
224|        """Perform inference on an image or stream.
225|
226|        Args:
227|            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
228|                Source for inference.
229|            model (str | Path | torch.nn.Module, optional): Model for inference.
230|            stream (bool): Whether to stream the inference results. If True, returns a generator.
231|            *args (Any): Additional arguments for the inference method.
232|            **kwargs (Any): Additional keyword arguments for the inference method.
233|
234|        Returns:
235|            (list[ultralytics.engine.results.Results] | generator): Results objects or generator of Results objects.
236|        """
237|        self.stream = stream
238|        if stream:
239|            return self.stream_inference(source, model, *args, **kwargs)
240|        else:
241|            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Results into one
242|
243|    def predict_cli(self, source=None, model=None):
244|        """Method used for Command Line Interface (CLI) prediction.
245|
246|        This function is designed to run predictions using the CLI. It sets up the source and model, then processes the
247|        inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
248|        generator without storing results.
249|
250|        Args:
251|            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
252|                Source for inference.
253|            model (str | Path | torch.nn.Module, optional): Model for inference.
254|
255|        Notes:
256|            Do not modify this function or remove the generator. The generator ensures that no outputs are
257|            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
258|        """
259|        gen = self.stream_inference(source, model)
260|        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
261|            pass
262|
263|    def setup_source(self, source, stride: int | None = None):
264|        """Set up source and inference mode.
265|
266|        Args:
267|            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor): Source for
268|                inference.
269|            stride (int, optional): Model stride for image size checking.
270|        """
271|        self.imgsz = check_imgsz(self.args.imgsz, stride=stride or self.model.stride, min_dim=2)  # check image size
272|        self.dataset = load_inference_source(
273|            source=source,
274|            batch=self.args.batch,
275|            vid_stride=self.args.vid_stride,
276|            buffer=self.args.stream_buffer,
277|            channels=getattr(self.model, "channels", 3),
278|        )
279|        self.source_type = self.dataset.source_type
280|        if (
281|            self.source_type.stream
282|            or self.source_type.screenshot
283|            or len(self.dataset) > 1000  # many images
284|            or any(getattr(self.dataset, "video_flag", [False]))
285|        ):  # long sequence
286|            import torchvision  # noqa (import here triggers torchvision NMS use in nms.py)
287|
288|            if not getattr(self, "stream", True):  # videos
289|                LOGGER.warning(STREAM_WARNING)
290|        self.vid_writer = {}
291|
292|    @smart_inference_mode()
293|    def stream_inference(self, source=None, model=None, *args, **kwargs):
294|        """Stream inference on input source and save results to file.
295|
296|        Args:
297|            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
298|                Source for inference.
299|            model (str | Path | torch.nn.Module, optional): Model for inference.
300|            *args (Any): Additional arguments for the inference method.
301|            **kwargs (Any): Additional keyword arguments for the inference method.
302|
303|        Yields:
304|            (ultralytics.engine.results.Results): Results objects.
305|        """
306|        if self.args.verbose:
307|            LOGGER.info("")
308|
309|        # Setup model
310|        if self.model is None:
311|            self.setup_model(model)
312|        if not getattr(self.model, "base_model", True) and (
313|            unsupported := [k for k in ("augment", "embed", "visualize") if getattr(self.args, k)]
314|        ):
315|            LOGGER.warning(f"{unsupported} not supported by this model (format='{self.model.format}'), ignoring.")
316|            self.args.augment, self.args.embed, self.args.visualize = False, None, False
317|
318|        with self._lock:  # for thread-safe inference
319|            # Setup source every time predict is called
320|            self.setup_source(source if source is not None else self.args.source)
321|
322|            # Check if save_dir/ label file exists
323|            if self.args.save or self.args.save_txt:
324|                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
325|
326|            self.seen, self.speed, self.pixels, self.windows, self.batch = 0, None, None, [], None
327|            px = 0  # inference pixels summed per image, so a mixed-shape source averages rather than reports its last
328|            profilers = (
329|                ops.Profile(device=self.device),
330|                ops.Profile(device=self.device),
331|                ops.Profile(device=self.device),
332|            )
333|            self.run_callbacks("on_predict_start")
334|            for batch in self.dataset:
335|                self.batch = batch
336|                self.run_callbacks("on_predict_batch_start")
337|                paths, im0s, s = self.batch
338|
339|                # Preprocess
340|                with profilers[0]:
341|                    im = self.preprocess(im0s)
342|
343|                if not self.done_warmup:
344|                    self.model.warmup(im=im)
345|                    self.done_warmup = True
346|
347|                # Inference
348|                with profilers[1]:
# Resolve overrides using base args snapshot to prevent argument sticking
resolved_args = get_cfg(self._base_args, args)
preds = self.inference(im, *args, **resolved_args)
350|                    if self.args.embed:
351|                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
352|                        continue
353|
354|                # Postprocess
355|                with profilers[2]:
356|                    self.results = self.postprocess(preds, im, im0s)
357|                self.run_callbacks("on_predict_postprocess_end")
358|
359|                # Visualize, save, write results
360|                n = len(im0s)
361|                try:
362|                    for i in range(n):
363|                        self.seen += 1
364|                        px += im.shape[2] * im.shape[3]
365|                        self.results[i].speed = {
366|                            "preprocess": profilers[0].dt * 1e3 / n,
367|                            "inference": profilers[1].dt * 1e3 / n,
368|                            "postprocess": profilers[2].dt * 1e3 / n,
369|                        }
370|                        if (
371|                            self.args.verbose
372|                            or self.args.save
373|                            or self.args.save_txt
374|                            or self.args.save_crop
375|                            or self.args.show
376|                        ):
377|                            s[i] += self.write_results(i, Path(paths[i]), im, s)
378|                except StopIteration:
379|                    break
380|
381|                # Print batch results
382|                if self.args.verbose:
383|                    LOGGER.info("\n".join(s))
384|
385|                self.run_callbacks("on_predict_batch_end")
386|                yield from self.results
387|
388|            # Final results, under the lock: seen is reset by every run, so reading it outside could divide this run's
389|            # profilers by a concurrent run's count. px and profilers are locals and are already private to this run.
390|            if seen := self.seen:
391|                t = tuple(x.t / seen * 1e3 for x in profilers)  # speeds per image
392|                self.speed = dict(zip(("preprocess", "inference", "postprocess"), t))
393|                self.pixels = round(px / seen)  # mean area, pairing with speeds that are themselves per-image means
394|                if self.args.verbose:
395|                    LOGGER.info(
396|                        f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
397|                        f"{(min(self.args.batch, seen), getattr(self.model, 'channels', 3), *im.shape[2:])}" % t
398|                    )
399|
400|        # Release assets
401|        for v in self.vid_writer.values():
402|            if isinstance(v, cv2.VideoWriter):
403|                v.release()
404|
405|        if self.args.show:
406|            cv2.destroyAllWindows()  # close any open windows
407|
408|        if self.args.save or self.args.save_txt or self.args.save_crop:
409|            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
410|            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
411|            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
412|        self.run_callbacks("on_predict_end")
413|
414|    def setup_model(self, model, verbose: bool = True):
415|        """Initialize YOLO model with given parameters and set it to evaluation mode.
416|
417|        Args:
418|            model (str | Path | torch.nn.Module): Model to load or use.
419|            verbose (bool): Whether to print verbose output.
420|        """
421|        if hasattr(model, "end2end"):
422|            if self.args.end2end is not None:
423|                model.end2end = self.args.end2end
424|            if model.end2end:
425|                # Keep head top-k >= 300 so `classes` filtering in NMS sees all candidates before `max_det` truncation
426|                model.set_head_attr(max_det=max(self.args.max_det, 300), agnostic_nms=self.args.agnostic_nms)
427|        self.model = AutoBackend(
428|            model=model or self.args.model,
429|            device=select_device(self.args.device, verbose=verbose),
430|            dnn=self.args.dnn,
431|            data=self.args.data,
432|            fp16=self.args.quantize == 16,
433|            fuse=True,
434|            verbose=verbose,
435|        )
436|
437|        self.device = self.model.device  # update device
438|        self.args.quantize = 16 if self.model.fp16 else None  # record actual inference precision
439|        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
440|            self.args.imgsz = self.model.imgsz  # reuse imgsz from export metadata
441|        self.model.eval()
442|        # channels_last (NHWC) is CUDA-only and native-PyTorch-only: lossless and Tensor-Core friendly there, wrong
443|        # on MPS, no CPU gain, and only a native nn.Module has weights to convert.
444|        channels_last = self.args.channels_last and self.device.type == "cuda" and self.model.format == "pt"
445|        if self.args.channels_last and not channels_last:
446|            LOGGER.warning(
447|                f"'channels_last=True' applies only to native PyTorch models on CUDA, ignoring for "
448|                f"format='{self.model.format}' on '{self.device.type}'."
449|            )
450|        if channels_last:
451|            self.model.to(memory_format=torch.channels_last)
452|        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)
# Snapshot base args after model setup to avoid kwargs persistence across calls
self._base_args = {**DEFAULT_CFG_DICT, **self.overrides, "quantize": self.args.quantize, "imgsz": self.args.imgsz}

# For end‑to‑end models, refresh head attributes each call
if hasattr(self.model, "set_head_attr"):
    self.model.set_head_attr(max_det=getattr(self.args, "max_det", 100), agnostic_nms=self.args.agnostic_nms)
453|
454|    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
455|        """Write inference results to a file or directory.
456|
457|        Args:
458|            i (int): Index of the current image in the batch.
459|            p (Path): Path to the current image.
460|            im (torch.Tensor): Preprocessed image tensor.
461|            s (list[str]): List of result strings.
462|
463|        Returns:
464|            (str): String with result information.
465|        """
466|        string = ""  # print string
467|        if len(im.shape) == 3:
468|            im = im[None]  # expand for batch dim
469|        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
470|            string += f"{i}: "
471|            frame = self.dataset.count
472|        else:
473|            match = re.search(r"frame (\d+)/", s[i])
474|            frame = int(match[1]) if match else None  # None if frame undetermined
475|
476|        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
477|        string += "{:g}x{:g} ".format(*im.shape[2:])
478|        result = self.results[i]
479|        result.save_dir = self.save_dir.__str__()  # used in other locations
480|        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"
481|
482|        # Add predictions to image
483|        if self.args.save or self.args.show:
484|            self.plotted_img = result.plot(
485|                line_width=self.args.line_width,
486|                boxes=self.args.show_boxes,
487|                conf=self.args.show_conf,
488|                labels=self.args.show_labels,
489|            )
490|
491|        # Save results
492|        if self.args.save_txt:
493|            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
494|        if self.args.save_crop:
495|            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
496|        if self.args.show:
497|            self.show(str(p))
498|        if self.args.save:
499|            self.save_predicted_images(self.save_dir / p.name, frame)
500|
501|
Usage - sources:
    $ yolo mode=predict model=yolo26n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream

Usage - formats:
    $ yolo mode=predict model=yolo26n.pt                 # PyTorch
                              yolo26n.torchscript        # TorchScript
                              yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolo26n_openvino_model     # OpenVINO
                              yolo26n.engine             # TensorRT
                              yolo26n.mlpackage          # CoreML (macOS-only)
                              yolo26n_saved_model        # TensorFlow SavedModel
                              yolo26n.pb                 # TensorFlow GraphDef
                              yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolo26n_paddle_model       # PaddlePaddle
                              yolo26n.mnn                # MNN
                              yolo26n_ncnn_model         # NCNN
                              yolo26n_imx_model          # Sony IMX
                              yolo26n_rknn_model         # Rockchip RKNN
                              yolo26n_executorch_model   # PyTorch Executorch
                              yolo26n_axelera_model      # Axelera AI
                              yolo26n_deepx_model        # DEEPX
                              yolo26n_qnn.onnx           # Qualcomm QNN
                              yolo26n.tflite             # LiteRT
                              yolo26n_ascend_model       # Huawei Ascend
"""

from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.plotting import class_activation_map
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode

STREAM_WARNING = """
Inference results will accumulate in RAM unless `stream=True` is passed, which can cause out-of-memory errors for large
sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    """A base class for creating predictors.

    This class provides the foundation for prediction functionality, handling model setup, inference, and result
    processing across various input sources.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (torch.nn.Module): Model used for prediction.
        data (str | Path | None): Copy of args.data, the dataset YAML AutoBackend falls back to for class names.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_writer (dict[Path, cv2.VideoWriter]): Dictionary of {save_path: video_writer} for saving video output.
        plotted_img (np.ndarray): Last plotted image.
        source_type (SimpleNamespace): Type of input source.
        seen (int): Number of images processed.
        speed (dict[str, float] | None): Per-image preprocess, inference and postprocess times in ms, once run.
        pixels (int | None): Mean per-image inference area in pixels, once a run completes.
        windows (list[str]): List of window names for visualization.
        batch (tuple): Current batch data.
        results (list[Any]): Current batch results.
        transforms (Callable): Image transforms for classification.
        callbacks (dict[str, list[Callable]]): Callback functions for different events.
        txt_path (Path): Path to save text results.
        _lock (threading.Lock): Lock for thread-safe inference.

    Methods:
        preprocess: Prepare input image before inference.
        inference: Run inference on a given image.
        postprocess: Process raw predictions into structured results.
        predict_cli: Run prediction for command line interface.
        setup_source: Set up input source and inference mode.
        stream_inference: Stream inference on input source.
        setup_model: Initialize and configure the model.
        write_results: Write inference results to files.
        save_predicted_images: Save prediction visualizations.
        show: Display results in a window.
        run_callbacks: Execute registered callbacks for an event.
        add_callback: Register a new callback function.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        """Initialize the BasePredictor class.

        Args:
            cfg (str | Path | dict | SimpleNamespace): Path to a configuration file or a configuration dictionary.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # dict of {save_path: video_writer, ...}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.speed = None  # per-image speeds, set once a run completes
        self.pixels = None  # mean per-image inference area, set once a run completes
        self.windows = []
        self.screen = None  # cached screen resolution (width, height) for show=True scaling
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """Prepare input image before inference.

        Args:
            im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list.

        Returns:
            (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR to RGB
            im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """Run inference on a given image using the specified model and arguments."""
        skip = self.source_type.tensor or self.args.augment or self.args.embed  # unsupported with activation maps
        if self.args.visualize and getattr(self.model, "base_model", True) and not skip:
            return class_activation_map(
                self.model,
                im,
                self.batch[0],
                self.save_dir,
                *args,
                conf=self.args.conf,
                classes=self.args.classes,
                **kwargs,
            )
        return self.model(im, *args, augment=self.args.augment, embed=self.args.embed, **kwargs)

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Pre-transform input image before inference.

        Args:
            im (list[np.ndarray]): List of images with shape [(H, W, 3) x N].

        Returns:
            (list[np.ndarray]): List of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) and self.model.format != "imx")),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """Post-process predictions for an image and return them."""
        return preds

    def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
        """Perform inference on an image or stream.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            stream (bool): Whether to stream the inference results. If True, returns a generator.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Returns:
            (list[ultralytics.engine.results.Results] | generator): Results objects or generator of Results objects.
        """
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Results into one

    def predict_cli(self, source=None, model=None):
        """Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes the
        inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.

        Notes:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """
        gen = self.stream_inference(source, model)
        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
            pass

    def setup_source(self, source, stride: int | None = None):
        """Set up source and inference mode.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor): Source for
                inference.
            stride (int, optional): Model stride for image size checking.
        """
        self.imgsz = check_imgsz(self.args.imgsz, stride=stride or self.model.stride, min_dim=2)  # check image size
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
            channels=getattr(self.model, "channels", 3),
        )
        self.source_type = self.dataset.source_type
        if (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # long sequence
            import torchvision  # noqa (import here triggers torchvision NMS use in nms.py)

            if not getattr(self, "stream", True):  # videos
                LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Stream inference on input source and save results to file.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Yields:
            (ultralytics.engine.results.Results): Results objects.
        """
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if self.model is None:
            self.setup_model(model)
        if not getattr(self.model, "base_model", True) and (
            unsupported := [k for k in ("augment", "embed", "visualize") if getattr(self.args, k)]
        ):
            LOGGER.warning(f"{unsupported} not supported by this model (format='{self.model.format}'), ignoring.")
            self.args.augment, self.args.embed, self.args.visualize = False, None, False

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            self.seen, self.speed, self.pixels, self.windows, self.batch = 0, None, None, [], None
            px = 0  # inference pixels summed per image, so a mixed-shape source averages rather than reports its last
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for batch in self.dataset:
                self.batch = batch
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                if not self.done_warmup:
                    self.model.warmup(im=im)
                    self.done_warmup = True

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        px += im.shape[2] * im.shape[3]
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                        if (
                            self.args.verbose
                            or self.args.save
                            or self.args.save_txt
                            or self.args.save_crop
                            or self.args.show
                        ):
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

            # Final results, under the lock: seen is reset by every run, so reading it outside could divide this run's
            # profilers by a concurrent run's count. px and profilers are locals and are already private to this run.
            if seen := self.seen:
                t = tuple(x.t / seen * 1e3 for x in profilers)  # speeds per image
                self.speed = dict(zip(("preprocess", "inference", "postprocess"), t))
                self.pixels = round(px / seen)  # mean area, pairing with speeds that are themselves per-image means
                if self.args.verbose:
                    LOGGER.info(
                        f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                        f"{(min(self.args.batch, seen), getattr(self.model, 'channels', 3), *im.shape[2:])}" % t
                    )

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.show:
            cv2.destroyAllWindows()  # close any open windows

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose: bool = True):
        """Initialize YOLO model with given parameters and set it to evaluation mode.

        Args:
            model (str | Path | torch.nn.Module): Model to load or use.
            verbose (bool): Whether to print verbose output.
        """
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if model.end2end:
                # Keep head top-k >= 300 so `classes` filtering in NMS sees all candidates before `max_det` truncation
                model.set_head_attr(max_det=max(self.args.max_det, 300), agnostic_nms=self.args.agnostic_nms)
        self.model = AutoBackend(
            model=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.quantize == 16,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # update device
        self.args.quantize = 16 if self.model.fp16 else None  # record actual inference precision
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.args.imgsz = self.model.imgsz  # reuse imgsz from export metadata
        self.model.eval()
        # channels_last (NHWC) is CUDA-only and native-PyTorch-only: lossless and Tensor-Core friendly there, wrong
        # on MPS, no CPU gain, and only a native nn.Module has weights to convert.
        channels_last = self.args.channels_last and self.device.type == "cuda" and self.model.format == "pt"
        if self.args.channels_last and not channels_last:
            LOGGER.warning(
                f"'channels_last=True' applies only to native PyTorch models on CUDA, ignoring for "
                f"format='{self.model.format}' on '{self.device.type}'."
            )
        if channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        """Write inference results to a file or directory.

        Args:
            i (int): Index of the current image in the batch.
            p (Path): Path to the current image.
            im (torch.Tensor): Preprocessed image tensor.
            s (list[str]): List of result strings.

        Returns:
            (str): String with result information.
        """
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # None if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # Add predictions to image
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
            )

        # Save results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)

        return string

    def save_predicted_images(self, save_path: Path, frame: int = 0):
        """Save video predictions as mp4/avi or images as jpg at specified path.

        Args:
            save_path (Path): Path to save the results.
            frame (int): Frame number for video mode.
        """
        im = self.plotted_img

        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = self.save_dir / f"{save_path.stem}_frames"  # save frames to a separate directory
            if save_path not in self.vid_writer:  # new video
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # Save video
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(str(save_path.with_suffix(".jpg")), im)  # save to JPG for best support

    def show(self, p: str = ""):
        """Display an image in a window."""
        im = self.plotted_img
        if platform.system() in {"Linux", "Windows"} and p not in self.windows:  # macOS scales natively
            self.windows.append(p)
            name = p.encode("unicode_escape").decode()  # match patched cv2.imshow window name
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize and scaling
            h, w = im.shape[:2]
            try:  # size window to fit screen once on creation if image larger than screen resolution
                if self.screen is None:
                    root = __import__("tkinter").Tk()
                    root.withdraw()  # hide the empty Tk window
                    self.screen = 0.9 * root.winfo_screenwidth(), 0.9 * root.winfo_screenheight()  # 0.9 taskbar margin
                    root.destroy()
                r = min(self.screen[0] / w, self.screen[1] / h, 1.0)
                cv2.resizeWindow(name, max(1, int(w * r)), max(1, int(h * r)))  # (width, height)
            except Exception:
                cv2.resizeWindow(name, w, h)
        cv2.imshow(p, im)
        if cv2.waitKey(300 if self.dataset.mode == "image" else 1) & 0xFF == ord("q"):  # 300ms if image; else 1ms
            raise StopIteration

    def run_callbacks(self, event: str):
        """Run all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func: Callable):
        """Add a callback function for a specific event."""
        self.callbacks[event].append(func)
