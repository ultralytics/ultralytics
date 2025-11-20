# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import datetime
import gc
import multiprocessing as mp
import os
import queue
import socket
import sys
import time
import uuid
from contextlib import closing
from typing import List, Optional

import psutil
import torch

from sam3.logger import get_logger

logger = get_logger(__name__)


class Sam3VideoPredictor:
    # a global dictionary that holds all inference states for this model (key is session_id)
    _ALL_INFERENCE_STATES = {}

    def __init__(
        self,
        checkpoint_path=None,
        bpe_path=None,
        has_presence_token=True,
        geo_encoder_use_img_cross_attn=True,
        strict_state_dict_loading=True,
        async_loading_frames=False,
        video_loader_type="cv2",
        apply_temporal_disambiguation: bool = True,
    ):
        self.async_loading_frames = async_loading_frames
        self.video_loader_type = video_loader_type
        from sam3.model_builder import build_sam3_video_model

        self.model = (
            build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                bpe_path=bpe_path,
                has_presence_token=has_presence_token,
                geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
                strict_state_dict_loading=strict_state_dict_loading,
                apply_temporal_disambiguation=apply_temporal_disambiguation,
            )
            .cuda()
            .eval()
        )

    @torch.inference_mode()
    def handle_request(self, request):
        """Dispatch a request based on its type."""
        request_type = request["type"]
        if request_type == "start_session":
            return self.start_session(
                resource_path=request["resource_path"],
                session_id=request.get("session_id", None),
            )
        elif request_type == "add_prompt":
            return self.add_prompt(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
                text=request.get("text", None),
                points=request.get("points", None),
                point_labels=request.get("point_labels", None),
                bounding_boxes=request.get("bounding_boxes", None),
                bounding_box_labels=request.get("bounding_box_labels", None),
                obj_id=request.get("obj_id", None),
            )
        elif request_type == "remove_object":
            return self.remove_object(
                session_id=request["session_id"],
                obj_id=request["obj_id"],
                is_user_action=request.get("is_user_action", True),
            )
        elif request_type == "reset_session":
            return self.reset_session(session_id=request["session_id"])
        elif request_type == "close_session":
            return self.close_session(session_id=request["session_id"])
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    @torch.inference_mode()
    def handle_stream_request(self, request):
        """Dispatch a stream request based on its type."""
        request_type = request["type"]
        if request_type == "propagate_in_video":
            yield from self.propagate_in_video(
                session_id=request["session_id"],
                propagation_direction=request.get("propagation_direction", "both"),
                start_frame_idx=request.get("start_frame_index", None),
                max_frame_num_to_track=request.get("max_frame_num_to_track", None),
            )
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    def start_session(self, resource_path, session_id=None):
        """
        Start a new inference session on an image or a video. Here `resource_path`
        can be either a path to an image file (for image inference) or an MP4 file
        or directory with JPEG video frames (for video inference).

        If `session_id` is defined, it will be used as identifier for the
        session. If it is not defined, the start_session function will create
        a session id and return it.
        """
        # get an initial inference_state from the model
        inference_state = self.model.init_state(
            resource_path=resource_path,
            async_loading_frames=self.async_loading_frames,
            video_loader_type=self.video_loader_type,
        )
        if not session_id:
            session_id = str(uuid.uuid4())
        self._ALL_INFERENCE_STATES[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
        }
        logger.debug(
            f"started new session {session_id}; {self._get_session_stats()}; "
            f"{self._get_torch_and_gpu_properties()}"
        )
        return {"session_id": session_id}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: Optional[str] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        bounding_boxes: Optional[List[List[float]]] = None,
        bounding_box_labels: Optional[List[int]] = None,
        obj_id: Optional[int] = None,
    ):
        """Add text, box and/or point prompt on a specific video frame."""
        logger.debug(
            f"add prompt on frame {frame_idx} in session {session_id}: "
            f"{text=}, {points=}, {point_labels=}, "
            f"{bounding_boxes=}, {bounding_box_labels=}"
        )
        session = self._get_session(session_id)
        inference_state = session["state"]

        frame_idx, outputs = self.model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text,
            points=points,
            point_labels=point_labels,
            boxes_xywh=bounding_boxes,
            box_labels=bounding_box_labels,
            obj_id=obj_id,
        )
        return {"frame_index": frame_idx, "outputs": outputs}

    def remove_object(
        self,
        session_id: str,
        obj_id: int,
        is_user_action: bool = True,
    ):
        """Remove an object from tracking."""
        logger.debug(
            f"remove object {obj_id} in session {session_id}: " f"{is_user_action=}"
        )
        session = self._get_session(session_id)
        inference_state = session["state"]

        self.model.remove_object(
            inference_state=inference_state,
            obj_id=obj_id,
            is_user_action=is_user_action,
        )
        return {"is_success": True}

    def propagate_in_video(
        self,
        session_id,
        propagation_direction,
        start_frame_idx,
        max_frame_num_to_track,
    ):
        """Propagate the added prompts to get grounding results on all video frames."""
        logger.debug(
            f"propagate in video in session {session_id}: "
            f"{propagation_direction=}, {start_frame_idx=}, {max_frame_num_to_track=}"
        )
        try:
            session = self._get_session(session_id)
            inference_state = session["state"]
            if propagation_direction not in ["both", "forward", "backward"]:
                raise ValueError(
                    f"invalid propagation direction: {propagation_direction}"
                )

            # First doing the forward propagation
            if propagation_direction in ["both", "forward"]:
                for frame_idx, outputs in self.model.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=max_frame_num_to_track,
                    reverse=False,
                ):
                    yield {"frame_index": frame_idx, "outputs": outputs}
            # Then doing the backward propagation (reverse in time)
            if propagation_direction in ["both", "backward"]:
                for frame_idx, outputs in self.model.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=max_frame_num_to_track,
                    reverse=True,
                ):
                    yield {"frame_index": frame_idx, "outputs": outputs}
        finally:
            # Log upon completion (so that e.g. we can see if two propagations happen in parallel).
            # Using `finally` here to log even when the tracking is aborted with GeneratorExit.
            logger.debug(
                f"propagation ended in session {session_id}; {self._get_session_stats()}"
            )

    def reset_session(self, session_id):
        """Reset the session to its initial state (as when it's initial opened)."""
        logger.debug(f"reset session {session_id}")
        session = self._get_session(session_id)
        inference_state = session["state"]
        self.model.reset_state(inference_state)
        return {"is_success": True}

    def close_session(self, session_id):
        """
        Close a session. This method is idempotent and can be called multiple
        times on the same "session_id".
        """
        session = self._ALL_INFERENCE_STATES.pop(session_id, None)
        if session is None:
            logger.warning(
                f"cannot close session {session_id} as it does not exist (it might have expired); "
                f"{self._get_session_stats()}"
            )
        else:
            del session
            gc.collect()
            logger.info(f"removed session {session_id}; {self._get_session_stats()}")
        return {"is_success": True}

    def _get_session(self, session_id):
        session = self._ALL_INFERENCE_STATES.get(session_id, None)
        if session is None:
            raise RuntimeError(
                f"Cannot find session {session_id}; it might have expired"
            )
        return session

    def _get_session_stats(self):
        """Get a statistics string for live sessions and their GPU usage."""
        # print both the session ids and their video frame numbers
        live_session_strs = [
            f"'{session_id}' ({session['state']['num_frames']} frames)"
            for session_id, session in self._ALL_INFERENCE_STATES.items()
        ]
        session_stats_str = (
            f"live sessions: [{', '.join(live_session_strs)}], GPU memory: "
            f"{torch.cuda.memory_allocated() // 1024**2} MiB used and "
            f"{torch.cuda.memory_reserved() // 1024**2} MiB reserved"
            f" (max over time: {torch.cuda.max_memory_allocated() // 1024**2} MiB used "
            f"and {torch.cuda.max_memory_reserved() // 1024**2} MiB reserved)"
        )
        return session_stats_str

    def _get_torch_and_gpu_properties(self):
        """Get a string for PyTorch and GPU properties (for logging and debugging)."""
        torch_and_gpu_str = (
            f"torch: {torch.__version__} with CUDA arch {torch.cuda.get_arch_list()}, "
            f"GPU device: {torch.cuda.get_device_properties(torch.cuda.current_device())}"
        )
        return torch_and_gpu_str

    def shutdown(self):
        """Shutdown the predictor and clear all sessions."""
        self._ALL_INFERENCE_STATES.clear()


class Sam3VideoPredictorMultiGPU(Sam3VideoPredictor):
    def __init__(self, *model_args, gpus_to_use=None, **model_kwargs):
        if gpus_to_use is None:
            # if not specified, use only the current GPU by default
            gpus_to_use = [torch.cuda.current_device()]

        IS_MAIN_PROCESS = os.getenv("IS_MAIN_PROCESS", "1") == "1"
        if IS_MAIN_PROCESS:
            gpus_to_use = sorted(set(gpus_to_use))
            logger.info(f"using the following GPU IDs: {gpus_to_use}")
            assert len(gpus_to_use) > 0 and all(isinstance(i, int) for i in gpus_to_use)
            assert all(0 <= i < torch.cuda.device_count() for i in gpus_to_use)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = f"{self._find_free_port()}"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = f"{len(gpus_to_use)}"

        self.gpus_to_use = gpus_to_use
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank_str = f"rank={self.rank} with world_size={self.world_size}"
        self.device = torch.device(f"cuda:{self.gpus_to_use[self.rank]}")
        torch.cuda.set_device(self.device)
        self.has_shutdown = False
        if self.rank == 0:
            logger.info("\n\n\n\t*** START loading model on all ranks ***\n\n")

        logger.info(f"loading model on {self.rank_str} -- this could take a while ...")
        super().__init__(*model_args, **model_kwargs)
        logger.info(f"loading model on {self.rank_str} -- DONE locally")

        if self.world_size > 1 and self.rank == 0:
            # start the worker processes *after* the model is loaded in the main process
            # so that the main process can run torch.compile and fill the cache first
            self._start_worker_processes(*model_args, **model_kwargs)
            for rank in range(1, self.world_size):
                self.command_queues[rank].put(("start_nccl_process_group", None))
            self._start_nccl_process_group()

        if self.rank == 0:
            logger.info("\n\n\n\t*** DONE loading model on all ranks ***\n\n")

    @torch.inference_mode()
    def handle_request(self, request):
        """Dispatch a request based on its type."""
        if self.has_shutdown:
            raise RuntimeError(
                "cannot handle request after the predictor has shutdown; please create a new predictor"
            )

        # when starting a session, we need to create a session id before dispatching
        # the request to the workers
        if request["type"] == "start_session" and request.get("session_id") is None:
            request["session_id"] = str(uuid.uuid4())
        # dispatch the request to all worker processes
        if self.world_size > 1 and self.rank == 0:
            for rank in range(1, self.world_size):
                self.command_queues[rank].put((request, False))

        response = super().handle_request(request)

        if self.world_size > 1:
            torch.distributed.barrier()  # wait for all ranks to finish
        return response

    @torch.inference_mode()
    def handle_stream_request(self, request):
        """Dispatch a stream request based on its type."""
        if self.has_shutdown:
            raise RuntimeError(
                "cannot handle request after the predictor has shutdown; please create a new predictor"
            )

        # dispatch the request to all worker processes
        if self.world_size > 1 and self.rank == 0:
            for rank in range(1, self.world_size):
                self.command_queues[rank].put((request, True))

        yield from super().handle_stream_request(request)

        if self.world_size > 1:
            torch.distributed.barrier()  # wait for all ranks to finish

    def _start_worker_processes(self, *model_args, **model_kwargs):
        """Start worker processes for handling model inference."""
        world_size = self.world_size
        logger.info(f"spawning {world_size - 1} worker processes")
        # Use "spawn" (instead of "fork") for different PyTorch or CUDA context
        mp_ctx = mp.get_context("spawn")
        self.command_queues = {rank: mp_ctx.Queue() for rank in range(1, world_size)}
        self.result_queues = {rank: mp_ctx.Queue() for rank in range(1, world_size)}
        parent_pid = os.getpid()
        for rank in range(1, world_size):
            # set the environment variables for each worker process
            os.environ["IS_MAIN_PROCESS"] = "0"  # mark this as a worker process
            os.environ["RANK"] = f"{rank}"
            worker_process = mp_ctx.Process(
                target=Sam3VideoPredictorMultiGPU._worker_process_command_loop,
                args=(
                    rank,
                    world_size,
                    self.command_queues[rank],
                    self.result_queues[rank],
                    model_args,
                    model_kwargs,
                    self.gpus_to_use,
                    parent_pid,
                ),
                daemon=True,
            )
            worker_process.start()
        # revert the environment variables for the main process
        os.environ["IS_MAIN_PROCESS"] = "1"
        os.environ["RANK"] = "0"
        # wait for all the worker processes to load the model and collect their PIDs
        self.worker_pids = {}
        for rank in range(1, self.world_size):
            # a large timeout to cover potentially long model loading time due to compilation
            _, worker_pid = self.result_queues[rank].get(timeout=7200)
            self.worker_pids[rank] = worker_pid
        logger.info(f"spawned {world_size - 1} worker processes")

    def _start_nccl_process_group(self):
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size == 1:
            return

        logger.debug(f"starting NCCL process group on {rank=} with {world_size=}")
        assert not torch.distributed.is_initialized()
        # use the "env://" init method with environment variables set in start_worker_processes
        # a short 3-min timeout to quickly detect any synchronization failures
        timeout_sec = int(os.getenv("SAM3_COLLECTIVE_OP_TIMEOUT_SEC", "180"))
        timeout = datetime.timedelta(seconds=timeout_sec)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timeout,
            device_id=self.device,
        )
        # warm-up the NCCL process group by running a dummy all-reduce
        tensor = torch.ones(1024, 1024).cuda()
        torch.distributed.all_reduce(tensor)
        logger.debug(f"started NCCL process group on {rank=} with {world_size=}")

    def _find_free_port(self) -> int:
        """
        Find a free port (a random free port from 1024 to 65535 will be selected)
        https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number)
        """
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    @staticmethod
    def _worker_process_command_loop(
        rank,
        world_size,
        command_queue,
        result_queue,
        model_args,
        model_kwargs,
        gpus_to_use,
        parent_pid,
    ):
        """
        The command loop for each worker process. It listens to commands from the main process
        and executes them using the model.
        """
        logger.info(f"starting worker process {rank=} with {world_size=}")
        # verify that the environment variables are set correctly
        assert int(os.environ["IS_MAIN_PROCESS"]) == 0
        assert int(os.environ["RANK"]) == rank
        assert int(os.environ["WORLD_SIZE"]) == world_size
        # load the model in this worker process
        predictor = Sam3VideoPredictorMultiGPU(
            *model_args, gpus_to_use=gpus_to_use, **model_kwargs
        )
        logger.info(f"started worker {rank=} with {world_size=}")
        # return the worker process id to the main process for bookkeeping
        worker_pid = os.getpid()
        result_queue.put(("load_model", worker_pid))

        # wait for the command to start the NCCL process group
        request_type, _ = command_queue.get(timeout=7200)
        assert request_type == "start_nccl_process_group"
        predictor._start_nccl_process_group()

        # keep listening to commands from the main process
        while True:
            try:
                request, is_stream_request = command_queue.get(timeout=5.0)
                if request == "shutdown":
                    logger.info(f"worker {rank=} shutting down")
                    torch.distributed.destroy_process_group()
                    result_queue.put(("shutdown", True))  # acknowledge the shutdown
                    sys.exit(0)

                logger.debug(f"worker {rank=} received request {request['type']=}")
                if is_stream_request:
                    for _ in predictor.handle_stream_request(request):
                        pass  # handle stream requests in a generator fashion
                else:
                    predictor.handle_request(request)
            except queue.Empty:
                # Usually Python's multiprocessing module will shutdown all the daemon worker
                # processes when the main process exits gracefully. However, the user may kill
                # the main process using SIGKILL and thereby leaving no chance for the main process
                # to clean up its daemon child processes. So here we manually check whether the
                # parent process still exists (every 5 sec as in `command_queue.get` timeout).
                if not psutil.pid_exists(parent_pid):
                    logger.info(
                        f"stopping worker {rank=} as its parent process has exited"
                    )
                    sys.exit(1)
            except Exception as e:
                logger.error(f"worker {rank=} exception: {e}", exc_info=True)

    def shutdown(self):
        """Shutdown all worker processes."""
        if self.rank == 0 and self.world_size > 1:
            logger.info(f"shutting down {self.world_size - 1} worker processes")
            for rank in range(1, self.world_size):
                self.command_queues[rank].put(("shutdown", False))
            torch.distributed.destroy_process_group()
            for rank in range(1, self.world_size):
                self.result_queues[rank].get()  # wait for the worker to acknowledge
            logger.info(f"shut down {self.world_size - 1} worker processes")
        self.has_shutdown = True

        super().shutdown()
