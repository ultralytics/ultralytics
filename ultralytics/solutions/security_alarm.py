# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors


class SecurityAlarm(BaseSolution):
    """
    A class to manage security alarm functionalities for real-time monitoring.

    This class extends the BaseSolution class and provides features to monitor
    objects in a frame, send email notifications when specific thresholds are
    exceeded for total detections, and annotate the output frame for visualization.

    Attributes:
       email_sent (bool): Flag to track if an email has already been sent for the current event.
       records (int): Threshold for the number of detected objects to trigger an alert.

    Methods:
       authenticate: Sets up email server authentication for sending alerts.
       send_email: Sends an email notification with details and an image attachment.
       monitor: Monitors the frame, processes detections, and triggers alerts if thresholds are crossed.

    Examples:
        >>> security = SecurityAlarm()
        >>> security.authenticate("abc@gmail.com", "1111222233334444", "xyz@gmail.com")
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = security.monitor(frame)
    """

    def __init__(self, **kwargs):
        """Initializes the SecurityAlarm class with parameters for real-time object monitoring."""
        super().__init__(**kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]
        self.server = None
        self.to_email = ""
        self.from_email = ""

    def authenticate(self, from_email, password, to_email):
        """
        Authenticates the email server for sending alert notifications.

        Args:
            from_email (str): Sender's email address.
            password (str): Password for the sender's email account.
            to_email (str): Recipient's email address.

        This method initializes a secure connection with the SMTP server
        and logs in using the provided credentials.

        Examples:
            >>> alarm = SecurityAlarm()
            >>> alarm.authenticate("sender@example.com", "password123", "recipient@example.com")
        """
        import smtplib

        self.server = smtplib.SMTP("smtp.gmail.com: 587")
        self.server.starttls()
        self.server.login(from_email, password)
        self.to_email = to_email
        self.from_email = from_email

    def send_email(self, im0, records=5):
        """
        Sends an email notification with an image attachment indicating the number of objects detected.

        Args:
            im0 (numpy.ndarray): The input image or frame to be attached to the email.
            records (int): The number of detected objects to be included in the email message.

        This method encodes the input image, composes the email message with
        details about the detection, and sends it to the specified recipient.

        Examples:
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> alarm.send_email(frame, records=10)
        """
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
        message.attach(MIMEText(message_body))

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
        """
        Monitors the frame, processes object detections, and triggers alerts if thresholds are exceeded.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed and annotated.

        This method processes the input frame, extracts detections, annotates the frame
        with bounding boxes, and sends an email notification if the number of detected objects
        surpasses the specified threshold and an alert has not already been sent.

        Returns:
            (numpy.ndarray): The processed frame with annotations.

        Examples:
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> processed_frame = alarm.monitor(frame)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Iterate over bounding boxes, track ids and classes index
        for box, cls in zip(self.boxes, self.clss):
            # Draw bounding box
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        total_det = len(self.clss)
        if total_det > self.records and not self.email_sent:  # Only send email If not sent before
            self.send_email(im0, total_det)
            self.email_sent = True

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
