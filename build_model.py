import os
import smtplib
from email.mime.text import MIMEText
import subprocess

def retrain_model():
    # Run model.py script
    subprocess.run(['python', 'model.py'])

# Function to send notification emails
def send_notification_emails(emails_collection):
    # Retrieve emails from MongoDB
    emails = emails_collection.find()
    emails.distinct("email")
    for email in emails:
        # Send email to user
        send_email(email['email'])

def send_email(email):
    subject = "Email Subject"
    body = "This is the body of the text message"
    sender = "fruitdetectionapp@gmail.com"
    recipients = [email]
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

    send_email(subject, body, sender, recipients, password)



def build_model(emails_collection):
    # Replace this with your actual model building and training code
    global fruit_categories
    retrain_model()
    send_notification_emails(emails_collection)
    directory = "images"
    fruit_categories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    fruit_categories = [word.replace("_", " ") for word in fruit_categories]
