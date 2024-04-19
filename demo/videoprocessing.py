import cv2

class DisplayVideo:
    def __init__(self, frame_queue):
        self.frame_queue = frame_queue
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                cv2.imshow("Video", frame)
                if cv2.waitKey(1) == ord('q'):
                    self.stop()

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()


class WriteVideo:
    pass

class ProcessVideo:
    pass


class VideoStream:
    def __init__(self, source=0, display=True, record=False):
        self.source = source
        self.display = display
        self.record = record

        # Colas para manejar el flujo de datos
        self.capture_queue = queue.Queue(maxsize=5)
        self.process_queue = queue.Queue(maxsize=5)
        self.display_queue = queue.Queue(maxsize=5)
        self.record_queue = queue.Queue(maxsize=5)

        # Instancias de las otras clases
        self.camera = CameraStream(src=self.source)
        self.processor = ProcessVideo(self.capture_queue, self.process_queue)
        self.tracker = TrackObjects(self.process_queue, self.display_queue, self.record_queue)
        self.display_video = DisplayVideo(self.display_queue) if self.display else None
        self.write_video = WriteVideo(self.record_queue) if self.record else None

    def start(self):
        # Iniciar las transmisiones de video y otros procesos
        self.camera.start()
        self.processor.start()
        self.tracker.start()
        if self.display:
            self.display_video.start()
        if self.record:
            self.write_video.start()

    def stop(self):
        # Detener todos los procesos y limpiar
        self.camera.stop()
        self.processor.stop()
        self.tracker.stop()
        if self.display:
            self.display_video.stop()
        if self.record:
            self.write_video.stop()