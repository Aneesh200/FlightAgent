# email_service.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import traceback
import socket

# Import config variables
from config import (
    RECIPIENT_EMAIL, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
)

def format_flight_details_for_email(flight_details):
    """Formats flight details into a readable string for email."""
    if not flight_details or not isinstance(flight_details, dict):
        return "Error: Invalid flight details provided."

    def get(key, default="N/A"):
        val = flight_details.get(key, default)
        return default if val is None else val

    price_str = f"{get('price'):.2f} {get('currency', 'USD')}" if isinstance(get('price'), (int, float)) else get('price')
    passenger_str = f"{get('passenger_count', 'N/A')}"
    class_str = f"{get('class_preference', 'Any')}"

    body = f"Flight Booking Request:\n"
    body += f"-------------------------\n"
    body += f"Offer ID: {get('offer_id')}\n"
    body += f"Airline: {get('airline')} ({get('airline_code')})\n"
    body += f"Price: {price_str}\n"
    body += f"Type: {get('type')}\n"
    body += f"Passengers: {passenger_str}\n"
    body += f"Class: {class_str}\n\n"

    body += f"Outbound Leg:\n"
    body += f"  From: {get('origin')} -> To: {get('destination')}\n"
    body += f"  Departs: {get('departure_time')}\n"
    body += f"  Arrives: {get('arrival_time')}\n"
    body += f"  Duration: {get('duration')}\n  Stops: {get('stops')}\n\n"

    if get('is_round_trip'):
        body += f"Return Leg:\n"
        body += f"  From: {get('destination')} -> To: {get('origin')}\n"
        body += f"  Departs: {get('return_departure_time')}\n"
        body += f"  Arrives: {get('return_arrival_time')}\n"
        body += f"  Duration: {get('return_duration')}\n  Stops: {get('return_stops')}\n\n"

    body += f"-------------------------\n"
    body += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n"
    body += f"Note: This email confirms your request details. Proceed with actual booking via airline/agent.\n"
    return body

def send_flight_details_email(flight_details, recipient_address=RECIPIENT_EMAIL):
    """Sends flight details via email using SMTP config or local sendmail."""
    if not flight_details: return False, "No flight details to send."
    if not recipient_address or recipient_address == "YOUR_RECIPIENT_EMAIL@example.com":
        return False, "Recipient email address is not configured."

    subject = f"Flight Booking Request - {flight_details.get('origin', 'N/A')} to {flight_details.get('destination', 'N/A')}"
    body = format_flight_details_for_email(flight_details)
    sender_address = SMTP_USER if SMTP_USER and SMTP_USER != "YOUR_SMTP_EMAIL@example.com" else "flight-assistant@localhost"

    msg = MIMEMultipart()
    msg['From'] = sender_address
    msg['To'] = recipient_address
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Check if full SMTP config is available
        use_smtp = SMTP_HOST and SMTP_PORT and SMTP_USER and \
                   SMTP_USER != "YOUR_SMTP_EMAIL@example.com" and \
                   SMTP_PASSWORD and SMTP_PASSWORD != "YOUR_APP_PASSWORD_OR_KEY"

        if use_smtp:
            port = int(SMTP_PORT)
            print(f"Connecting to SMTP: {SMTP_HOST}:{port} as {SMTP_USER}")
            server = None
            if port == 465: server = smtplib.SMTP_SSL(SMTP_HOST, port, timeout=30)
            else:
                server = smtplib.SMTP(SMTP_HOST, port, timeout=30)
                server.ehlo(); server.starttls(); server.ehlo()

            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(sender_address, recipient_address, msg.as_string())
            server.quit()
            print(f"Email sent successfully to {recipient_address} via SMTP.")
            return True, "Email sent successfully via SMTP."
        else:
            print("SMTP not fully configured. Attempting local sendmail...")
            try:
                with smtplib.SMTP('localhost') as server:
                    server.sendmail(sender_address, recipient_address, msg.as_string())
                print(f"Email sent successfully to {recipient_address} via local sendmail.")
                return True, "Email sent successfully via local sendmail."
            except (ConnectionRefusedError, FileNotFoundError, smtplib.SMTPSenderRefused) as local_e:
                error_msg = f"Local sendmail failed: {local_e}. Ensure local MTA is running/configured."
                print(f"Email Sending Error: {error_msg}")
                return False, error_msg
            except Exception as local_e:
                error_msg = f"Failed to send email via local sendmail: {local_e}"
                print(f"Email Sending Error: {error_msg}")
                traceback.print_exc()
                return False, error_msg

    except (smtplib.SMTPAuthenticationError, smtplib.SMTPServerDisconnected,
            smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused, smtplib.SMTPException,
            socket.gaierror, socket.timeout) as e:
        error_msg = f"SMTP Error ({type(e).__name__}): {e}"
        print(f"Email Sending Error: {error_msg}")
        traceback.print_exc()
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during email sending: {e}"
        print(f"Email Sending Error: {error_msg}")
        traceback.print_exc()
        return False, error_msg