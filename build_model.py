import os
import smtplib
from email.mime.text import MIMEText
import subprocess

def retrain_model():
    # Run model.py script
    try:
        subprocess.run([r'C:\Users\elad1\PycharmProjects\model_for_app2\.venv\Scripts\python.exe', 'model.py'])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the model: {e}")

# Function to send notification emails
def send_notification_emails(emails_collection):
    # Retrieve emails from MongoDB
    emails = emails_collection.find()
    emails.distinct("email")
    for email in emails:
        # Send email to user
        send_email(email['email'])
        # After sending, delete the email from the collection
        emails_collection.delete_one({'_id': email['_id']})


def send_email(email):
    subject = "Email Subject"
    body = "Hi the model was retrained"
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
    directory = "train"
    fruit_categories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    fruit_categories = [word.replace("_", " ") for word in fruit_categories]
    print(len(fruit_categories))
    print(fruit_categories)
    send_notification_emails(emails_collection)
    retrain_model()


