# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import LOGGER, BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class SecurityAlarm(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]

    def authenticate(self, from_email, password, to_email):
        # Authenticate server for email
        import smtplib

        self.server = smtplib.SMTP("smtp.gmail.com: 587")
        self.server.starttls()
        self.server.login(from_email, password)
        self.to_email = to_email
        self.from_email = from_email

    def send_email(self, im0, records=5):
        """Sends an email notification with an image attachment indicating the number of objects detected."""
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        import cv2

        img_bytes = cv2.imencode(".jpg", im0)[1].tobytes()  # Encode the image as JPEG

        # Create the email
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = "Security Alert"

        # Add the text message body
        message_body = f"Ultralytics ALERT!!! " f"{records} objects have been detected!!"
        message.attach(MIMEText(message_body, "plain"))

        # Attach the image
        image_attachment = MIMEImage(img_bytes, name="ultralytics.jpg")
        message.attach(image_attachment)

        # Send the email
        try:
            self.server.send_message(message)
            LOGGER.info("✅ Email sent successfully!")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")

    def monitor(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Iterate over bounding boxes, track ids and classes index
        for box, cls in zip(self.boxes, self.clss):
            # Draw bounding box
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        total_det = len(self.clss)
        if total_det > self.records:  # Only send email If not sent before
            if not self.email_sent:
                self.send_email(im0, total_det)
                self.email_sent = True

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
