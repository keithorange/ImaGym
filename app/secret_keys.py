import os
import json
import random
import string
import time
import datetime
from calendar import monthrange

import random
import string
import datetime


def generate_random_key():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=4)) + '_' + \
           ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))


def generate_keys(start_year, end_year, monthly_filename, yearly_filename):
    with open(monthly_filename, 'w') as monthly_file, open(yearly_filename, 'w') as yearly_file:
        for year in range(start_year, end_year + 1):
            # Yearly key generation
            year_first_day = datetime.datetime(year, 1, 1)
            yearly_key = generate_random_key()
            yearly_file.write(
                f"{int(year_first_day.timestamp())}:{yearly_key}\n")

            # Monthly key generation
            for month in range(1, 13):
                month_first_day = datetime.datetime(year, month, 1)
                monthly_key = generate_random_key()
                monthly_file.write(
                    f"{int(month_first_day.timestamp())}:{monthly_key}\n")


def is_monthly_key_valid(key, filename):
    current_time = time.time()
    with open(filename, 'r') as file:
        for line in file:
            timestamp, stored_key = line.strip().split(':')
            if stored_key == key:
                month_start = datetime.datetime.fromtimestamp(int(timestamp))
                # Calculate the timestamp for the next month
                if month_start.month == 12:
                    next_month = month_start.replace(
                        year=month_start.year + 1, month=1)
                else:
                    next_month = month_start.replace(
                        month=month_start.month + 1)
                next_month_timestamp = next_month.timestamp()

                if month_start.timestamp() <= current_time < next_month_timestamp:
                    return True
    return False


def is_yearly_key_valid(key, filename):
    current_time = time.time()
    with open(filename, 'r') as file:
        for line in file:
            timestamp, stored_key = line.strip().split(':')
            if stored_key == key:
                year_start = datetime.datetime.fromtimestamp(int(timestamp))
                # Calculate the timestamp for the same month next year
                next_year = year_start.replace(year=year_start.year + 1)
                next_year_timestamp = next_year.timestamp()

                if year_start.timestamp() <= current_time < next_year_timestamp:
                    return True
    return False


def get_monthly_key_for_timestamp(timestamp, monthly_filename):
    target_date = datetime.datetime.fromtimestamp(timestamp)

    with open(monthly_filename, 'r') as file:
        for line in file:
            file_timestamp, key = line.strip().split(':')
            month_start = datetime.datetime.fromtimestamp(int(file_timestamp))
            if target_date.year == month_start.year and target_date.month == month_start.month:
                return key, 'monthly'

    return None, None


def get_yearly_key_for_timestamp(timestamp, yearly_filename):
    target_date = datetime.datetime.fromtimestamp(timestamp)

    with open(yearly_filename, 'r') as file:
        for line in file:
            file_timestamp, key = line.strip().split(':')
            year_start = datetime.datetime.fromtimestamp(int(file_timestamp))
            if target_date.year == year_start.year:
                return key, 'yearly'

    return None, None


def get_current_key(monthly_filename, yearly_filename):
    current_time = time.time()
    monthly_key, monthly_key_type = get_monthly_key_for_timestamp(
        current_time, monthly_filename)
    yearly_key, yearly_key_type = get_yearly_key_for_timestamp(
        current_time, yearly_filename)

    return {
        'monthly': {'key': monthly_key, 'type': monthly_key_type},
        'yearly': {'key': yearly_key, 'type': yearly_key_type}
    }


# # Example usage:
# current_keys = get_current_key('monthly_keys.txt', 'yearly_keys.txt')
# print(
#     f"Monthly Key: {current_keys['monthly']['key']}, Type: {current_keys['monthly']['type']}")
# print(
#     f"Yearly Key: {current_keys['yearly']['key']}, Type: {current_keys['yearly']['type']}")


# # Example usage:
# timestamp = 1672527600  # Replace with your desired timestamp
# key, key_type = get_key_for_timestamp(
#     timestamp, 'monthly_keys.txt', 'yearly_keys.txt')
# print(f"Key: {key}, Type: {key_type}")
# current_key, current_key_type = get_current_key(
#     'monthly_keys.txt', 'yearly_keys.txt')
# print(f"Current Key: {current_key}, Type: {current_key_type}")
funny_keys = [
    "LOLZ_4U", "YODA_BEST", "GROOT_TALK", "CATZ_MEOW",
    "DOGE_MOON", "42_ANSWER", "SNAKE_PSKT", "FLY_RNWAY",
    "NOM_NOMS", "JAVA_WHO", "PANDA_BAM", "GOTTA_FAST"
]


def is_funny_key(key):
    return key in funny_keys


# key_to_check = "LOLZ_4U"  # Replace with the key to check
# if is_funny_key(key_to_check):
#     print(f"The key '{key_to_check}' is a funny key and always unlocks!")
# else:
#     print(f"The key '{key_to_check}' is not a funny key.")


"""

a system that tracks and maintains the login status of users using a text file, you can follow these steps:

Validate User Login: When a user logs in, check the validity of the key they provide (either monthly or yearly). If the key is valid, proceed to record the login session.

Record Login Session: Save the initial timestamp, the duration for which the login is valid, the projected exit timestamp, and the key used. Store this information in a JSON format for easier reading and writing.

Maintain Login Status: Regularly check (e.g., during bootup) if the current time has passed the exit timestamp. If so, update the login status to "inactive" while preserving the session data.

Update Login File: When logging out (either manually or automatically after the expiry of the key), update the file to reflect the change in status.
"""

LOGIN_STATUS_FILE = 'login_status.json'


def record_login_session(key, duration_hours):
    current_time = time.time()
    exit_time = current_time + duration_hours * 3600

    if not os.path.exists(LOGIN_STATUS_FILE):
        login_data = {}
    else:
        with open(LOGIN_STATUS_FILE, 'r') as file:
            login_data = json.load(file)

    # Check if key has already been used
    if key in login_data.get('used_keys', []):
        return None  # Key already used

    login_data.update({
        "initial_timestamp": current_time,
        "duration_hours": duration_hours,
        "exit_timestamp": exit_time,
        "key_used": key,
        "active": True,
        # Add key to used keys list
        "used_keys": login_data.get('used_keys', []) + [key]
    })

    with open(LOGIN_STATUS_FILE, 'w') as file:
        json.dump(login_data, file)

    return login_data


def check_and_update_login_status():
    if not os.path.exists(LOGIN_STATUS_FILE):
        return {"status": "No login data available"}

    with open(LOGIN_STATUS_FILE, 'r') as file:
        login_data = json.load(file)

    current_time = time.time()
    if login_data['active'] and current_time >= login_data['exit_timestamp']:
        login_data['active'] = False
        with open(LOGIN_STATUS_FILE, 'w') as file:
            json.dump(login_data, file)
        return {"status": "Logged out due to key expiration"}

    return {"status": "Active" if login_data['active'] else "Inactive", "data": login_data}


# # Example usage
# # Assume the user logs in with a valid key
# valid_key = "ValidKey123"
# duration_hours = 24  # For example, 24 hours
# login_session = record_login_session(valid_key, duration_hours)
# print(f"Login session recorded: {login_session}")

# # To check and update login status
# status = check_and_update_login_status()
# print(f"Login status: {status}")
class AccountManager:
    def __init__(self, monthly_keys_file="monthly_keys.txt", yearly_keys_file="yearly_keys.txt", login_status_file="login_status.txt"):
        self.monthly_keys_file = monthly_keys_file
        self.yearly_keys_file = yearly_keys_file
        self.login_status_file = login_status_file

    def validate_premium_key(self, key):
        if not os.path.exists(LOGIN_STATUS_FILE):
            login_data = {}
        else:
            with open(LOGIN_STATUS_FILE, 'r') as file:
                login_data = json.load(file)

        # Check if key has already been used
        if key in login_data.get('used_keys', []):
            return False  # Key already used

        # Check if key is valid (monthly or yearly)
        if is_monthly_key_valid(key, self.monthly_keys_file) or \
           is_yearly_key_valid(key, self.yearly_keys_file) or \
           is_funny_key(key):
            return True
        return False

    def record_login_session(self, key):
        # Determine duration based on key type
        if is_monthly_key_valid(key, self.monthly_keys_file):
            duration_hours = 24 * 30
        elif is_yearly_key_valid(key, self.yearly_keys_file):
            duration_hours = 24 * 365
        elif is_funny_key(key):
            duration_hours = 24 * 365 * 100  # 100 years for funny keys
        else:
            return None  # Invalid key

        return record_login_session(key, duration_hours)

    def check_and_update_login_status(self):
        return check_and_update_login_status()

    def is_premium_active(self):
        if not os.path.exists(LOGIN_STATUS_FILE):
            return False

        with open(LOGIN_STATUS_FILE, 'r') as file:
            login_data = json.load(file)

        current_time = time.time()
        return login_data.get('active', False) and current_time < login_data.get('exit_timestamp', 0)
