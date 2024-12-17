# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class SecurityAlarm(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]

    def authenticate(self, from_email, password, to_email):
        # Authenticate server for email
        import smtplib
        server = smtplib.SMTP("smtp.gmail.com: 587")
        server.starttls()
        server.login(from_email, password)
        self.to_email = to_email
        self.from_email = from_email

    def send_email(self, object_detected=5):
        """Sends an email notification indicating the number of objects detected; defaults to 5 object."""
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = "Security Alert"
        # Add in the message body
        message_body = (f"Ultralytics ALERT!!! "
                        f"{object_detected} objects has been detected!!")

        message.attach(MIMEText(message_body, "plain"))
        server.sendmail(from_email, to_email, message.as_string())

    def monitor(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Draw bounding box and counting region
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        total_det = len(self.clss)
        if total_det > self.records:  # Only send email If not sent before
            if not self.email_sent:
                send_email(total_det)
                self.email_sent = True

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
