from smtplib import SMTP_SSL as SMTP
import logging
import logging.handlers
import os
import subprocess
import argparse
from email.mime.text import MIMEText

SRC_ADDRESS = "email_sender123@163.com"
SMTP_SERVER = "smtp.163.com"
PASSWORD = "a12b34c56"

__PATH__ = os.path.abspath(__file__)
__DIR_NAME = os.path.dirname(__PATH__)

__LOG_FILE = open(os.path.join(__DIR_NAME, 'email_sender.log'), 'a')

def _send_msg(title, text, dst_address):
    title = 'By Xingjian Email Sender: ' + title
    text = 'By Xingjian Email Sender: \n' + text
    msg = MIMEText(text, 'plain')
    msg['Subject'] = title
    msg['From'] = SRC_ADDRESS
    msg['To'] = str(dst_address)
    try:
        conn = SMTP(SMTP_SERVER)
        conn.set_debuglevel(True)
        conn.login(SRC_ADDRESS, PASSWORD)
        try:
            conn.sendmail(SRC_ADDRESS, dst_address, msg.as_string())
        finally:
            conn.close()

    except Exception as exc:
        logging.error("ERROR!!!")
        logging.critical(exc)
        raise RuntimeError


def send_msg(title, text, dst_address):
    subprocess.Popen(['python3', __PATH__,
                      '--title', str(title),
                      '--text', str(text),
                      '--dst_address', str(dst_address)],
                     stdout=__LOG_FILE,
                     stderr=__LOG_FILE)


def parse_args():
    parser = argparse.ArgumentParser(description='Send email given the title and content.')
    parser.add_argument('--title', dest='title', required=True, type=str)
    parser.add_argument('--text', dest='text', required=True, type=str)
    parser.add_argument('--dst_address', dest='dst_address', required=True, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    _send_msg(title=args.title, text=args.text, dst_address=args.dst_address)
