import smtplib
from email.mime.text import MIMEText

#from main import process_image

subject = "Email Subject"
body = "This is the body of the text message"
sender = "fruitdetectionapp@gmail.com"
recipients = ["elad10101234@gmail.com"]
password = "dsgh doia zynk keof"


def send_email(subject, body, sender, recipients, password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
       smtp_server.login(sender, password)
       smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Message sent!")

#def up(image):
#    image = cv2.imread(image)
#    # Process image using YOLO
#    processed_image = process_image(image)
#    cv2.imshow("Processed Image", processed_image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#up('oo.jpg')

