# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import logging
import queue
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from time import time

import requests


class ConsoleLogger:
    """Captures console output and streams to API endpoint or local file."""

    def __init__(self, source):
        """Initialize console logger with API endpoint or local file path."""
        self.source = source
        self.is_api = isinstance(source, str) and (source.startswith("http://") or source.startswith("https://"))
        if not self.is_api:
            self.source = Path(source)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_queue = queue.Queue()
        self.active = False
        self.worker_thread = None
        self.last_line = ""
        self.last_line_time = 0
        self.last_progress_core = ""  # Track core progress content without timing

    def start_capture(self):
        """Start capturing console output and hook into Ultralytics LOGGER."""
        if self.active:
            return

        self.active = True
        sys.stdout = self._ConsoleCapture(self.original_stdout, self._queue_log)
        sys.stderr = self._ConsoleCapture(self.original_stderr, self._queue_log)

        # Also hook Ultralytics LOGGER
        self._hook_ultralytics_logger()

        self.worker_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.worker_thread.start()

    def _hook_ultralytics_logger(self):
        """Hook into Ultralytics LOGGER to capture all log messages."""
        try:
            # Get the Ultralytics logger and add our handler
            ultralytics_logger = logging.getLogger("ultralytics")

            # Create custom handler that queues messages
            class LogHandler(logging.Handler):
                def __init__(self, callback):
                    super().__init__()
                    self.callback = callback

                def emit(self, record):
                    msg = self.format(record)
                    self.callback(msg + "\n")

            handler = LogHandler(self._queue_log)
            ultralytics_logger.addHandler(handler)
        except Exception:
            pass  # Don't break if logger hook fails

    def stop_capture(self):
        """Stop capturing console output."""
        if not self.active:
            return

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.active = False
        self.log_queue.put(None)  # Signal worker to stop

    def _queue_log(self, text):
        """Queue console text for streaming with progress bar deduplication."""
        if self.active and text.strip():
            current_time = time()

            # Handle progress bars and carriage returns
            if "\r" in text:
                text = text.split("\r")[-1]

            # Split multiline messages into individual lines
            lines = text.split("\n")

            for line in lines:
                line = line.rstrip()
                if line:
                    # Check if line already has timestamp
                    has_timestamp = line.startswith("[") and "]" in line[:30]

                    # Check for tqdm progress bar pattern - any line with percentage and it/s
                    is_progress = "it/s" in line and "%|" in line

                    # Skip all progress bars except 100% completion
                    if is_progress:
                        if not re.search(r"100%\|[#\s]*\|", line):
                            continue
                        # For 100% progress bars, dedupe by core content (strip timing)
                        progress_core = re.sub(r"\[\d+:\d+<[\d:,]+\s*[\d.]+it/s\]", "", line).strip()
                        if progress_core == self.last_progress_core:
                            continue
                        self.last_progress_core = progress_core

                    # Regular deduplication for non-progress lines
                    elif line == self.last_line and current_time - self.last_line_time < 0.1:
                        continue

                    self.last_line = line
                    self.last_line_time = current_time

                    # Add timestamp if not already present
                    if has_timestamp:
                        self.log_queue.put(f"{line}\n")
                    else:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.log_queue.put(f"[{timestamp}] {line}\n")

    def _stream_worker(self):
        """Background worker to stream logs."""
        while self.active:
            try:
                log_text = self.log_queue.get(timeout=1)
                if log_text is None:  # Stop signal
                    break
                self._write_log(log_text)
            except queue.Empty:
                continue

    def _write_log(self, text):
        """Write log to API endpoint or local file."""
        try:
            if self.is_api:
                self._write_api(text)
            else:
                self._write_local(text)
        except Exception as e:
            # Don't break training if logging fails
            print(f"Platform logging error: {e}", file=self.original_stderr)

    def _write_api(self, text):
        """Send log to API endpoint."""
        payload = {"timestamp": datetime.now().isoformat(), "message": text.strip()}
        requests.post(self.source, json=payload, timeout=5)

    def _write_local(self, text):
        """Write log to local file."""
        self.source.parent.mkdir(parents=True, exist_ok=True)
        with self.source.open("a", encoding="utf-8") as f:
            f.write(text)
            f.flush()

    class _ConsoleCapture:
        """Captures stdout/stderr writes."""

        def __init__(self, original, callback):
            self.original = original
            self.callback = callback

        def write(self, text):
            self.original.write(text)
            self.callback(text)

        def flush(self):
            self.original.flush()
