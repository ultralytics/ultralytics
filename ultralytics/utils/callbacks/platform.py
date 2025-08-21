# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import logging
import queue
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from time import time

from ultralytics.utils import LOGGER, RANK, SETTINGS


# Global early logger for debugging - start capture immediately when module loads
_global_logger = None
if RANK in {-1, 0}:
    try:
        from pathlib import Path
        log_path = Path("ultralytics_debug.log")
        if log_path.exists():
            log_path.unlink()
    except Exception:
        pass


class ConsoleLogger:
    """Captures console output and streams to GCP bucket or local file."""
    
    def __init__(self, source):
        """Initialize console logger with GCP bucket or local file path."""
        self.source = Path(source) if not source.startswith('gs://') else source
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
            ultralytics_logger = logging.getLogger('ultralytics')
            
            # Create custom handler that queues messages
            class LogHandler(logging.Handler):
                def __init__(self, callback):
                    super().__init__()
                    self.callback = callback
                    
                def emit(self, record):
                    msg = self.format(record)
                    self.callback(msg + '\n')
            
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
            if '\r' in text:
                text = text.split('\r')[-1]
            
            # Split multiline messages into individual lines
            lines = text.split('\n')
            
            for line in lines:
                line = line.rstrip()
                if line:
                    # Check if line already has timestamp
                    has_timestamp = line.startswith('[') and ']' in line[:30]
                    
                    # Check for tqdm progress bar pattern - any line with percentage and it/s
                    is_progress = ('it/s' in line and '%|' in line)
                    
                    # Skip all progress bars except 100% completion
                    if is_progress:
                        if not re.search(r'100%\|[#\s]*\|', line):
                            continue
                        # For 100% progress bars, dedupe by core content (strip timing)
                        progress_core = re.sub(r'\[\d+:\d+<[\d:,]+\s*[\d.]+it/s\]', '', line).strip()
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
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        """Write log to GCS bucket or local file."""
        try:
            if str(self.source).startswith('gs://'):
                self._write_gcs(text)
            else:
                self._write_local(text)
        except Exception as e:
            # Don't break training if logging fails
            print(f"Platform logging error: {e}", file=self.original_stderr)
            
    def _write_gcs(self, text):
        """Write log to GCS bucket."""
        from google.cloud import storage
        
        bucket_name = str(self.source).replace('gs://', '').split('/')[0]
        blob_path = '/'.join(str(self.source).replace('gs://', '').split('/')[1:])
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Append to existing file
        existing_content = ""
        try:
            existing_content = blob.download_as_text()
        except Exception as e:
            pass  # File doesn't exist yet
            
        blob.upload_from_string(existing_content + text)
        
    def _write_local(self, text):
        """Write log to local file."""
        self.source.parent.mkdir(parents=True, exist_ok=True)
        with open(self.source, 'a', encoding='utf-8') as f:
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


def on_pretrain_routine_start(trainer):
    """Initialize and start console logging immediately at the very beginning."""
    if RANK in {-1, 0}:
        # Delete existing log and start capture FIRST
        log_path = Path("train.log")
        if log_path.exists():
            log_path.unlink()
        
        # Create and start logger immediately before any other output
        trainer.platform_logger = ConsoleLogger("train.log")
        trainer.platform_logger.start_capture()


def on_pretrain_routine_end(trainer):
    """Console capture already started in on_pretrain_routine_start."""
    pass


def on_fit_epoch_end(trainer):
    """Log epoch completion."""
    if logger := getattr(trainer, "platform_logger", None):
        pass


def on_model_save(trainer):
    """Log model save events."""
    if logger := getattr(trainer, "platform_logger", None):
        pass


def on_train_end(trainer):
    """Stop console capture and finalize logs."""
    if logger := getattr(trainer, "platform_logger", None):
        logger.stop_capture()


def on_train_start(trainer):
    """Log training start."""
    if logger := getattr(trainer, "platform_logger", None):
        pass


def on_val_start(validator):
    """Disabled - only log training."""
    pass


def on_predict_start(predictor):
    """Disabled - only log training."""
    pass


def on_export_start(exporter):
    """Disabled - only log training."""
    pass


callbacks = {
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_pretrain_routine_end": on_pretrain_routine_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_model_save": on_model_save,
    "on_train_end": on_train_end,
    "on_train_start": on_train_start,
    "on_val_start": on_val_start,
    "on_predict_start": on_predict_start,
    "on_export_start": on_export_start,
}  # always register callbacks, check platform_source in each callback
