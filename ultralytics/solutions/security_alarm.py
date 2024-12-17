# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class SecurityAlarm(BaseSolution):
    def __init__(self):
        super().__init__(**kwargs)
        self.start_time = 0
        self.end_time = 0

    def send_email(self, to_email, from_email, object_detected=1):
        """Sends an email notification indicating the number of objects detected; defaults to 1 object."""
        message = MIMEMultipart()
        message["From"] = from_email
        message["To"] = to_email
        message["Subject"] = "Security Alert"
        # Add in the message body
        message_body = f"ALERT - {object_detected} objects has been detected!!"

        message.attach(MIMEText(message_body, "plain"))
        server.sendmail(from_email, to_email, message.as_string())

    def display_fps(self, im0):
        """Displays the FPS on an image `im0` by calculating and overlaying as white text on a black rectangle."""
        self.end_time = time()
        fps = 1 / round(self.end_time - self.start_time, 2)
        text = f"FPS: {int(fps)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(
            im0,
            (20 - gap, 70 - text_size[1] - gap),
            (20 + text_size[0] + gap, 70 + gap),
            (255, 255, 255),
            -1,
        )
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)\
