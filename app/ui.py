import asyncio
import datetime
import logging
import random
import re
import subprocess
import sys
import json
import os
import threading
import time
import io
import cv2
import numpy as np
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QScrollArea, QMessageBox, QHBoxLayout, QCheckBox, QSpinBox, QSlider, QDialog, QLineEdit, QProgressBar, QGridLayout
from PySide2.QtCore import Qt, QSize, QTimer
from PySide2.QtGui import QPixmap, QIcon, QFont
from PySide2.QtWidgets import QSystemTrayIcon, QMenu
from PySide2.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PySide2.QtWidgets import QMainWindow, QLabel
from PySide2.QtGui import QPixmap
from PySide2.QtCore import QUrl
from PySide2.QtGui import QColor, QPalette, QPainter, QBrush
import math
from math import sqrt
from math import ceil
import os
import requests
from urllib.parse import urlparse
# Add to your existing imports
from PySide2.QtWidgets import QComboBox, QLineEdit

from PySide2.QtWidgets import QSpacerItem, QSizePolicy
from PySide2.QtCore import Signal

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from PIL.ImageQt import ImageQt
from tqdm import tqdm

import gpt_flipper
from gpt_flipper import DEFAULT_FULL_ON_VOLUME, FlipperConfig, FlipperScriptArgs
import qasync

from edit_image import apply_global_transparency, apply_random_effects_cv, ensure_3_channels
from secret_keys import AccountManager
from utils import linearly_scaled_random

import os
import asyncio
import aiohttp
import aiofiles
from tqdm.asyncio import tqdm_asyncio
import re


IMAGES_PATH = 'images'


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# class AsyncImageDownloader:
#     def __init__(self, concurrent_downloads=30):
#         self.concurrent_downloads = concurrent_downloads

#     async def download_and_save_images(self, image_directory, number_of_images):
#         if not os.path.exists(image_directory):
#             os.makedirs(image_directory)

#         images_downloaded = 0
#         page_number = 1
#         limit = 100  # Maximum number of images per API page

#         # Set up tqdm for total progress tracking
#         pbar = tqdm_asyncio(total=number_of_images, desc="Downloading images")

#         async with aiohttp.ClientSession() as session:
#             while images_downloaded < number_of_images:
#                 images_to_fetch = min(
#                     limit, number_of_images - images_downloaded)
#                 list_url = f'https://picsum.photos/v2/list?page={
#                     page_number}&limit={images_to_fetch}'

#                 try:
#                     async with session.get(list_url) as response:
#                         if response.status == 200:
#                             images_info = await response.json()
#                             tasks = [self.download_and_save_image(session, image_directory, image_info['download_url'], pbar)
#                                      for image_info in images_info if images_downloaded < number_of_images]

#                             # Process tasks in batches of 'self.concurrent_downloads'
#                             for i in range(0, len(tasks), self.concurrent_downloads):
#                                 batch = tasks[i:i+self.concurrent_downloads]
#                                 await asyncio.gather(*batch)

#                             images_downloaded += len(tasks)
#                         else:
#                             logging.error(
#                                 f"Failed to fetch image list from {list_url}")

#                 except Exception as e:
#                     logging.error(f"Error fetching image list: {e}")

#                 page_number += 1

#         pbar.close()

#     async def download_and_save_image(self, session, image_directory, image_url, pbar):
#         filename = self.generate_filename_from_url(image_url)
#         file_path = os.path.join(image_directory, filename)

#         if not os.path.exists(file_path):
#             try:
#                 async with session.get(image_url) as response:
#                     if response.status == 200:
#                         content = await response.read()
#                         async with aiofiles.open(file_path, 'wb') as file:
#                             await file.write(content)
#                             logging.info(f"Downloaded and saved: {image_url}")
#                     else:
#                         logging.error(
#                             f"Failed to download image from {image_url}")
#                         pass
#             except Exception as e:
#                 logging.error(f"Error downloading image: {e}")
#                 pass
#             finally:
#                 pbar.update(1)

#     @staticmethod
#     def generate_filename_from_url(url):
#         sanitized_url = re.sub(r'[^a-zA-Z0-9]', '_', url)
#         return f"{sanitized_url}.jpg"

class PremiumWindow(QDialog):
    def __init__(self, account_manager, parent=None):
        super().__init__(parent)
        self.account_manager = account_manager
        self.setWindowTitle("Premium Account")
        self.layout = QVBoxLayout(self)
        self.setStyleSheet("background-color: grey;")

        self.premium_status = self.check_premium_status()
        self.setup_ui_based_on_status()

    def setup_ui_based_on_status(self):
        # Clear existing widgets
        for i in reversed(range(self.layout.count())):
            widget_to_remove = self.layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        # Add widgets based on premium status
        if self.premium_status['active']:
            status_label = QLabel(
                f"Status: Active - Expires on {self.premium_status['expiration_date'].strftime('%B %Y')}")
            status_label.setStyleSheet("font-weight: bold;")
            self.layout.addWidget(status_label)
        else:
            self.layout.addWidget(QLabel("Status: Inactive"))
            self.key_input = QLineEdit(self)
            self.layout.addWidget(self.key_input)
            activate_button = QPushButton("Activate", self)
            activate_button.setStyleSheet(
                "background-color: #007BFF; color: white; border-radius: 5px;")
            activate_button.clicked.connect(self.activate_premium)
            self.layout.addWidget(activate_button)

        self.website_link = QLabel(
            "<a href='http://yourwebsite.com'>Purchase Premium on our website</a>", self)
        self.website_link.setOpenExternalLinks(True)
        self.layout.addWidget(self.website_link)

    def activate_premium(self):
        secret_key = self.key_input.text()
        if self.account_manager.validate_premium_key(secret_key):
            login_data = self.account_manager.record_login_session(secret_key)
            self.update_premium_status_ui(login_data)
        else:
            QMessageBox.warning(self, "Activation Failed",
                                "Invalid Premium Key")

    def update_premium_status_ui(self, login_data):
        # Update the premium status and UI
        self.premium_status = self.check_premium_status()
        self.setup_ui_based_on_status()

    def check_premium_status(self):
        status = self.account_manager.check_and_update_login_status()
        if status['status'] == "Active":
            expiration_date = datetime.datetime.fromtimestamp(
                status['data']['exit_timestamp'])
            return {
                'active': True,
                'duration': f"{status['data']['duration_hours']} hours",
                'activation_date': datetime.datetime.fromtimestamp(status['data']['initial_timestamp']),
                'expiration_date': expiration_date
            }
        return {'active': False, 'duration': 'N/A', 'activation_date': None, 'expiration_date': None}


class WallpaperView(QMainWindow):
    clicked = Signal()

    def __init__(self, shared_state=None, use_offline_images=False, parent=None):
        super().__init__(parent)
        self.network_manager = QNetworkAccessManager(self)
        self.network_manager.finished.connect(self.onImageDownloaded)
        self.shared_state = shared_state
        self.use_offline_images = use_offline_images
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ImaGym Overlay  ENSURE ON TOP OF VIDEO')
        self.imageLabel = QLabel(self)  # Create a QLabel to display images
        self.setCentralWidget(self.imageLabel)

        # Set the background color to black
        self.setStyleSheet("background-color: black;")

        self.resizeToScreen()

        # set initial opacity
        self.setWindowOpacity(0.3)

    def mousePressEvent(self, event):
        # Emit the clicked signal when the widget is clicked
        self.clicked.emit()
        super().mousePressEvent(event)  # Call the parent class's mousePressEvent

    def resizeToScreen(self):
        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width(), screen_size.height())
        self.imageLabel.setFixedSize(screen_size.width(), screen_size.height())

    def fetchNewImage(self):
        if self.use_offline_images:
            self.loadRandomLocalImage()
        else:
            url = 'https://picsum.photos/{width}/{height}'.format(
                width=self.width(), height=self.height())
            self.network_manager.get(QNetworkRequest(QUrl(url)))

    def loadRandomLocalImage(self, image_directory='blocking_images'):
        # Load a random image from the local directory
        if os.path.exists(image_directory):
            images = [img for img in os.listdir(image_directory) if os.path.isfile(
                os.path.join(image_directory, img))]
            if images:
                random_image = random.choice(images)
                image_path = os.path.join(image_directory, random_image)
                self.processAndDisplayImage(image_path)
            else:
                print("No images found in the directory.")
        else:
            print("Image directory not found.")

    def processAndDisplayImage(self, image_path):
        # Load the image and apply effects
        pil_image = Image.open(image_path)
        self.applyEffectsAndDisplay(self.convertPilToOpenCV(pil_image))

    def apply_random_effects(image, num_effects=-1):
        # Apply random effects to this image
        # If num_effects is -1, apply 1 to all effects
        apply_random_effects_cv(image, num_effects)

    def applyEffectsAndDisplay(self, open_cv_image):
        # Apply effects and display the image
        n_effects = random.randint(5, 10)
        try:
            open_cv_image = apply_random_effects_cv(
                open_cv_image, num_effects=n_effects)
            # Convert back to Qt Image and set it to the label
            pil_image = Image.fromarray(
                open_cv_image[:, :, ::-1])  # Convert back to PIL
            qt_image = ImageQt(pil_image)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.imageLabel.size(), Qt.KeepAspectRatioByExpanding)
            self.imageLabel.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error applying effects: {e}")

    def convertPilToOpenCV(self, pil_image):
        # Convert PIL image to OpenCV format
        open_cv_image = np.array(pil_image)
        open_cv_image = ensure_3_channels(open_cv_image)
        return open_cv_image[:, :, ::-1].copy()  # Convert from RGB to BGR

    def onImageDownloaded(self, reply):
        statusCode = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
        if statusCode == 302:
            newUrl = reply.attribute(
                QNetworkRequest.RedirectionTargetAttribute)
            self.network_manager.get(QNetworkRequest(newUrl))
            return
        if reply.error():
            print(f"Network error: {reply.errorString()}")
            return
        data = reply.readAll()
        if not data:
            print("Received empty data")
            return

        pixmap = QPixmap()
        if pixmap.loadFromData(data):
            pil_image = Image.open(io.BytesIO(data))
            self.applyEffectsAndDisplay(self.convertPilToOpenCV(pil_image))
        else:
            print("Failed to load image data.")


CONFIG_FILE = "configurations.json"


class ConfigManager:
    def __init__(self):
        self.configurations = self.load_configs()
        self.check_and_create_default_config()

    def load_configs(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as file:
                return json.load(file)
        return {}

    def check_and_create_default_config(self):
        if 'Mild' not in self.configurations:
            self.configurations['Mild'] = {
                'max_volume': 80,
                'cognitive_load': 50,
                'audio_full_on_chance': 60,  # reversed from backend cuz lazy
                'video_full_on_chance': 80,
                'session_length': 45,
                'flip_audio': True,
                'flip_video': True,
                'opaque_overlay': False,
                'black_overlay': False,
            }
            self.save_configs()
        if 'Intense' not in self.configurations:
            self.configurations['Intense'] = {
                'max_volume': 90,
                'cognitive_load': 95,
                'audio_full_on_chance': 80,
                'video_full_on_chance': 100,
                'session_length': 10,
                'flip_audio': True,
                'flip_video': True,
                'opaque_overlay': False,
                'black_overlay': False,
            }
            self.save_configs()

    def save_configs(self):
        with open(CONFIG_FILE, "w") as file:
            json.dump(self.configurations, file)

    def add_config(self, name, config):
        self.configurations[name] = config
        self.save_configs()

    def get_config(self, name):
        return self.configurations.get(name, None)

    def delete_config(self, name):
        if name in self.configurations:
            del self.configurations[name]
            self.save_configs()


class EndSessionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("End Session")

        # Create and configure layout
        layout = QVBoxLayout(self)

        # Add message label
        message_label = QLabel("Do you want to end the session?")
        layout.addWidget(message_label)

        # Add buttons and their layout
        buttons_layout = QHBoxLayout()

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        # End Session button with custom style
        end_session_button = QPushButton("End Session")
        end_session_button.clicked.connect(self.accept)
        end_session_button.setStyleSheet("""
            QPushButton {
                background-color: grey;
                color: white;
                border-radius: 10px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #a9a9a9; /* slightly lighter grey for hover effect */
            }
        """)

        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(end_session_button)

        layout.addLayout(buttons_layout)


class WallpaperViewSettingsWindow(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WallpaperView Settings")

        self.directory_label = QLabel("Directory")
        self.directory_input = QLineEdit()

        self.download_section_label = QLabel("Download Wallpaper Images")
        self.number_of_images_input = QSpinBox()
        self.download_button = QPushButton("Download")
        self.download_progress_bar = QProgressBar()

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.directory_label)
        self.layout.addWidget(self.directory_input)
        self.layout.addWidget(self.download_section_label)
        self.layout.addWidget(self.number_of_images_input)
        self.layout.addWidget(self.download_button)
        self.layout.addWidget(self.download_progress_bar)

        self.setLayout(self.layout)


class TimeTracker:
    def __init__(self, file_path='time_tracker.json', max_daily_time=60 * 30):
        self.file_path = file_path
        self.max_daily_time = max_daily_time  # 30 minutes
        self.data = self.load_data()
        self.current_session_start = None
        self.cleanup_sessions()

    def load_data(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                return json.load(file)
        else:
            return {"sessions": []}

    def save_data(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.data, file)

    def start_session(self):
        self.current_session_start = datetime.datetime.now().isoformat()

    def end_session(self):
        if self.current_session_start:
            end_time = datetime.datetime.now()
            session = {
                "start": self.current_session_start,
                "end": end_time.isoformat()
            }
            self.data['sessions'].append(session)
            self.current_session_start = None  # Reset the start time
            self.save_data()

    def get_total_time_used(self):
        self.cleanup_sessions()
        total_time = 0
        for session in self.data['sessions']:
            start = datetime.datetime.fromisoformat(session['start'])
            end = datetime.datetime.fromisoformat(session['end'])
            total_time += (end - start).total_seconds()
        return total_time

    def check_time_remaining(self):
        self.cleanup_sessions()
        total_time = self.get_total_time_used()
        return max(self.max_daily_time - total_time, 0)

    def cleanup_sessions(self):
        """Remove sessions older than 24 hours."""
        twenty_four_hours_ago = datetime.datetime.now() - datetime.timedelta(days=1)
        self.data['sessions'] = [session for session in self.data['sessions']
                                 if datetime.datetime.fromisoformat(session['end']) > twenty_four_hours_ago]
        self.save_data()

    def get_total_time_used_today(self):
        today = datetime.datetime.now().date()
        total_time_today = 0
        for session in self.data['sessions']:
            start = datetime.datetime.fromisoformat(session['start'])
            end = datetime.datetime.fromisoformat(session['end'])
            if start.date() == today or end.date() == today:
                # Adjust start and end times if they are not entirely within today
                if start.date() != today:
                    start = datetime.datetime.combine(today, datetime.time.min)
                if end.date() != today:
                    end = datetime.datetime.combine(today, datetime.time.max)
                total_time_today += (end - start).total_seconds()
        return total_time_today


class OutOfTimeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Out of Time")
        self.layout = QVBoxLayout(self)

        # Set the message
        self.message_label = QLabel(
            "<h2>Out of Time</h2>"
            "<p>Unlock premium today for Unlimited Use</p>"
            "<ul>"
            "<li>Full Customization</li>"
            "<li>Control Audio and Video Settings Individually</li>"
            "<li>Use Fully-Black Overlay, Fully-Opaque Overlay, and More!</li>"
            "<li>Custom Volume and Video Control</li>"
            "<li>Maximum Imagination Training</li>"
            "</ul>"
        )
        self.message_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.message_label)

        # Add button to close the dialog or go to the purchase page
        self.purchase_button = QPushButton("Upgrade to Premium")
        self.purchase_button.clicked.connect(self.on_purchase_clicked)
        self.layout.addWidget(self.purchase_button)

    def on_purchase_clicked(self):
        # Implement the logic to redirect to the purchase page or another relevant action
        self.accept()  # Close the dialog for now


class MainUI(QMainWindow):
    def __init__(self, USE_OFFLINE_IMAGES=False):
        super().__init__()
        self.setWindowTitle('ImaGym')
        # Set window geometry: x-position, y-position, width, height

    # Set window width and height
        self.width = 400
        self.height = 500
        self.resize(self.width, self.height)
        # Center the window on the screen
        self.centerWindow()

        self.flipper_task = None  # To hold the asyncio task for flipper

        self.initTrayIcon()
        self.updateTrayIconMenu(False)  # Initial state: not flipping

        self.shared_state = gpt_flipper.SharedState()
        # self.wallpaper_view = WallpaperView(shared_state=self.shared_state)

        self.USE_OFFLINE_IMAGES = USE_OFFLINE_IMAGES

        self.spacing_spacers = []  # Store the spacers for easy access

        # New attributes for tracking configurations
        self.current_config = None
        self.initial_config = None
        self.config_modified = False

        # manage account
        self.account_manager = AccountManager()

        self.initUI()

        # tracking free time usage
        self.time_tracker = TimeTracker()
        # Timer for updating premium status
        self.update_premium_status_timer = QTimer(self)
        self.update_premium_status_timer.timeout.connect(
            self.updatePremiumStatus)
        self.update_premium_status_timer.start(
            10000)  # Update every 10 seconds

        # Timer for updating remaining time today
        self.update_remaining_time_timer = QTimer(self)
        self.update_remaining_time_timer.timeout.connect(
            self.updateTotalTimeUsedToday)
        self.update_remaining_time_timer.start(1000)  # Update every second

        # track to see if paid premium
        self.premium_status_timer = QTimer(self)
        self.premium_status_timer.timeout.connect(self.updatePremiumStatus)
        self.premium_status_timer.start(10000)  # 10 seconds interval

        # Check premium status immediately on start
        self.updatePremiumStatus()

    def updateRemainingTimeDisplay(self):
        # Check if premium is active
        premium_active = self.account_manager.is_premium_active()

        if not premium_active:
            remaining_time_s = self.time_tracker.check_time_remaining()
            minutes, seconds = divmod(remaining_time_s, 60)
            self.feeStatusLabel.setText(
                f"{int(minutes):02d}:{int(seconds):02d} left today")
        else:
            # If premium is active, display "Premium Activated"
            self.feeStatusLabel.setText("Premium Activated")

    def centerWindow(self):
        # Get the screen resolution from the app
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # Calculate the x and y coordinates
        x = (screen_width - self.width) // 2
        y = (screen_height - self.height) // 2

        # Set the geometry
        self.setGeometry(x, y, self.width, self.height)
        # Attribute to track if the current config has been modified

    def load_initial_config(self):
        default_config_name = 'Mild'
        if default_config_name in self.config_manager.configurations:
            index = self.config_dropdown.findText(default_config_name)
            self.config_dropdown.setCurrentIndex(index)
            self.load_config(index)

    def updateConfigUI(self):
        """Show/hide UI elements based on the config state"""
        if self.config_modified:
            self.save_config_button.show()
            self.config_name_edit.show()
            self.delete_config_button.hide()
        else:
            self.save_config_button.hide()
            self.config_name_edit.hide()
            self.delete_config_button.show()

    def initTrayIcon(self):
        self.tray_icon = QSystemTrayIcon(
            QIcon(f'{IMAGES_PATH}/tray_icon.png'), self)
        self.tray_menu = QMenu()

        # self.show_controller_action = self.tray_menu.addAction("Show Controller")
        # self.show_controller_action.triggered.connect(self.toggleControllerTopLevel)

        self.toggle_flip_action = self.tray_menu.addAction("Start Flipping")
        self.toggle_flip_action.triggered.connect(self.toggleFlipping)

        self.exit_action = self.tray_menu.addAction("Exit")
        self.exit_action.triggered.connect(app.quit)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.show()

    def toggleControllerTopLevel(self):
        self.show()

    def updateUIButtons(self):
        if self.isPlaying:
            self.play_button.setIcon(QIcon(f'{IMAGES_PATH}/pause.png'))
            self.play_button.setStyleSheet("background-color: yellow;")
            self.stop_button.setVisible(True)
        else:
            self.play_button.setIcon(QIcon(f'{IMAGES_PATH}/play.png'))
            self.play_button.setStyleSheet("background-color: green;")
            self.stop_button.setVisible(False)

    def toggleFlipping(self):
        if self.isPlaying:
            self.stopFlipping()
        else:
            self.startFlipping()

    def updateTrayIconMenu(self, is_flipping):
        if is_flipping:
            self.toggle_flip_action.setText("Stop Flipping")
            self.tray_icon.setIcon(QIcon(f'{IMAGES_PATH}/stop_tray_icon.png'))
        else:
            self.toggle_flip_action.setText("Start Flipping")
            self.tray_icon.setIcon(QIcon(f'{IMAGES_PATH}/tray_icon.png'))

    def initPremiumButton(self):
        # Add Premium Button
        self.premium_button = QPushButton('Premium', self)
        self.premium_button.clicked.connect(self.open_premium_window)
        self.main_layout.addWidget(self.premium_button)

    def open_premium_window(self):
        # Assuming you have initialized account_manager somewhere in MainUI
        # For example, self.account_manager = AccountManager(...)
        self.premium_window = PremiumWindow(self.account_manager, self)
        self.premium_window.show()

    def initPremiumStatusBar(self):
        self.statusBar = QWidget()
        self.statusLayout = QHBoxLayout(self.statusBar)

        # Add a stretch to push content to the right
        self.statusLayout.addStretch(1)

        # Status label for Fee
        self.feeStatusLabel = QLabel("Fee: x minutes left today")
        self.feeStatusLabel.setStyleSheet(
            "color: rgba(255, 255, 255, 0.5);")  # 50% transparent text
        self.statusLayout.addWidget(self.feeStatusLabel)

        self.statusLayout.setContentsMargins(
            0, 0, 10, 0)  # Right margin of 10px

        self.main_layout.addWidget(self.statusBar)

    def updatePremiumStatus(self):
        premium_active = self.account_manager.is_premium_active()
        self.updatePremiumStatusBar(premium_active)

    def updatePremiumStatusBar(self, premium_active):
        if premium_active:
            self.feeStatusLabel.setText("Premium Activated")
            self.feeStatusLabel.setStyleSheet(
                "color: rgba(255, 255, 255, 1);")  # Opaque text
            self.premium_button.hide()  # Hide the button
        else:
            remaining_time_s = self.time_tracker.check_time_remaining()
            minutes, seconds = divmod(remaining_time_s, 60)
            self.feeStatusLabel.setText(
                f"Free {int(minutes):02d}:{int(seconds):02d} left today")
            self.premium_button.show()

    def checkAndUpdatePremiumStatus(self):
        premium_active = self.account_manager.is_premium_active()
        self.updatePremiumStatusBar(premium_active)

    def updateTotalTimeUsedToday(self):
        total_time_today = self.time_tracker.get_total_time_used_today()

        hours, remainder = divmod(total_time_today, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.totalTimeUsedTodayLabel.setText(
            f"{int(hours):02d}:{int(minutes):02d} used today")

    def initTimeUsage(self):
        # Inside the __init__ method of MainUI
        self.totalTimeUsedTodayLabel = QLabel("00:00 used today", self)
        self.totalTimeUsedTodayLabel.setStyleSheet(
            "color: rgba(255, 255, 255, 0.5);")  # 50% transparent text

        # Layout for aligning totalTimeUsedTodayLabel to the right
        timeUsageLayout = QHBoxLayout()
        timeUsageLayout.addStretch(1)  # Push label to the right
        timeUsageLayout.addWidget(self.totalTimeUsedTodayLabel)
        timeUsageLayout.setContentsMargins(0, 0, 10, 0)  # Right margin of 10px

        # Add layout to main_layout instead of the label directly
        self.main_layout.addLayout(timeUsageLayout)

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.initMenuBar()
        self.initBrandingAndNav()
        self.initConfigDropdown()

        self.initBasicSettings()
        self.initAdvancedSettings()
        self.initControlButtons()
        self.initTimeUsage()

        self.initPremiumButton()
        self.initPremiumStatusBar()
        self.updatePremiumStatus()

        self.initDisclaimer()
        self.setStyleSheet(self.getStyleSheet())
        self.addSpacingSpacers()
        # # Adjust spacing and margins

        self.addSpacingSpacers()
        # self.main_layout.update()
        self.load_initial_config()

    def initDisclaimer(self):
        logoPixmap = QPixmap(f'{IMAGES_PATH}/logo.png').scaled(40,
                                                               40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo = QLabel()
        logo.setPixmap(logoPixmap)

        disclaimerPixmap = QPixmap(f'{IMAGES_PATH}/warning.png').scaled(
            500/2, 150/2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        disclaimer = QLabel()
        disclaimer.setPixmap(disclaimerPixmap)
        self.main_layout.addWidget(disclaimer, alignment=Qt.AlignHCenter)

    def initMenuBar(self):
        # Add a menu option for the configuration manager
        self.menu_bar = self.menuBar()  # If you don't have a menu bar yet, add this line
        self.config_menu = self.menu_bar.addMenu("Settings")

        # self.show_online_ai_settings_action = self.config_menu.addAction("Online AI Settings")
        # self.show_online_ai_settings_action.triggered.connect(self.openOnlineAISettings)

        # self.wallpaper_view_settings_action = self.config_menu.addAction("WallpaperView Settings")
        # self.wallpaper_view_settings_action.triggered.connect(self.openWallpaperViewSettings)

        # self.history_menu_action = self.config_menu.addAction("Show History")
        # self.history_menu_action.triggered.connect(self.showHistory)

        self.account_menu_action = self.config_menu.addAction("Account")
        self.account_menu_action.triggered.connect(self.openAccount)

        self.help_menu_action = self.config_menu.addAction("Help")
        self.help_menu_action.triggered.connect(self.openHelp)

    def openWallpaperViewSettings(self):
        self.settings_window = WallpaperViewSettingsWindow()
        self.settings_window.show()

    def openHelp(self):
        # Logic to handle the help action
        pass

    def openAccount(self):
        # Logic to handle the account action
        pass

    def addSpacingSpacers(self):
        # Clear existing spacers
        for spacer in self.spacing_spacers:
            self.main_layout.removeItem(spacer)
        self.spacing_spacers.clear()

        # Determine which sections are visible
        sections = [self.basic_settings_container,
                    self.advanced_settings_container, self.control_container]
        visible_sections = [
            section for section in sections if section.isVisible()]

        # Add spacers between visible sections
        spacer_index = 1  # Start index for inserting spacers
        for section in visible_sections:
            # Add spacer before each section except the first one
            if spacer_index > 1:
                spacer = QSpacerItem(
                    20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
                self.main_layout.insertSpacerItem(spacer_index, spacer)
                self.spacing_spacers.append(spacer)
                spacer_index += 2  # Increase index to skip the section widget
            else:
                spacer_index = 2  # Next spacer should be after the first section

        self.main_layout.update()

    def initBrandingAndNav(self):
        # Branding
        logoPixmap = QPixmap(f'{IMAGES_PATH}/logo.png').scaled(40,
                                                               40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo = QLabel()
        logo.setPixmap(logoPixmap)

        title = QLabel('ImaGym')
        titleFont = QFont()
        titleFont.setPointSize(24)
        title.setProperty("title", True)  # Set property for title styling
        title.setAlignment(Qt.AlignCenter)  # Center align the title

        # Layout for logo and title
        logoTitleLayout = QHBoxLayout()
        logoTitleLayout.addWidget(logo)
        logoTitleLayout.addWidget(title)
        logoTitleLayout.setAlignment(Qt.AlignCenter)  # Center align the title

        # logoTitleLayout.addStretch()  # Add stretch to push logo and title to the left
        # If you need a spacer to control space after title
        spacer = QSpacerItem(
            40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        logoTitleLayout.addSpacerItem(spacer)

        logoTitleLayout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Container for the logo and title
        logoTitleContainer = QWidget()
        logoTitleContainer.setLayout(logoTitleLayout)

        self.main_layout.addWidget(
            logoTitleContainer, alignment=Qt.AlignHCenter)

    def initConfigDropdown(self):
        self.config_dropdown = QComboBox(self)
        self.config_dropdown.setEditable(False)
        self.config_manager = ConfigManager()
        self.config_dropdown.addItems(
            list(self.config_manager.configurations.keys()))
        self.config_dropdown.setInsertPolicy(QComboBox.NoInsert)
        self.config_dropdown.currentIndexChanged.connect(self.load_config)

        self.config_name_edit = QLineEdit(self)
        self.config_name_edit.setPlaceholderText("Enter config name")
        self.config_name_edit.hide()

        # ... [Rest of your code] ...

        self.save_config_button = QPushButton(self)
        self.save_config_button.setIcon(QIcon(f'{IMAGES_PATH}/save_icon.png'))
        self.save_config_button.clicked.connect(self.save_config)
        self.save_config_button.hide()
        self.save_config_button.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Set this to the size of your image plus padding
        self.save_config_button.setFixedSize(80, 80)
        self.save_config_button.setStyleSheet(
            "QPushButton { border-radius: 5px; margin-top: 40px;}")

        self.delete_config_button = QPushButton(self)
        self.delete_config_button.setIcon(
            QIcon(f'{IMAGES_PATH}/delete_icon.png'))
        self.delete_config_button.clicked.connect(self.delete_config)
        self.delete_config_button.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Set this to the size of your image plus padding
        self.delete_config_button.setFixedSize(80, 80)
        self.delete_config_button.setStyleSheet(
            "QPushButton { border-radius: 5px; margin-top: 50px;}")

    # ... [Rest of your code] ...

        # ... [Rest of your code] ...

        # Create a vertical layout for the dropdown and the input text
        self.dropdown_layout = QVBoxLayout()
        self.dropdown_layout.addWidget(QLabel("Configuration:"))
        self.dropdown_layout.addWidget(self.config_dropdown)
        self.dropdown_layout.addWidget(self.config_name_edit)

        # ... [Rest of your code] ...

        # Create a grid layout for the buttons
        self.buttons_layout = QGridLayout()
        self.buttons_layout.addWidget(self.save_config_button, 1, 1)
        self.buttons_layout.addWidget(self.delete_config_button, 1, 2)

        # ... [Rest of your code] ...

        # Create a horizontal layout for the dropdown and buttons layouts
        self.dropdown_buttons_layout = QHBoxLayout()

        # Add the dropdown and buttons layouts to the horizontal layout
        self.dropdown_buttons_layout.addLayout(self.dropdown_layout)
        self.dropdown_buttons_layout.addLayout(self.buttons_layout)

        # Add the horizontal layout to the main layout
        self.main_layout.addLayout(self.dropdown_buttons_layout)

    def apply_config(self, config):
        # Apply configuration values to UI elements
        for key, value in config.items():
            if hasattr(self, key):
                ui_element = getattr(self, key)

                # Handling different types of UI elements
                if isinstance(ui_element, QSlider):
                    ui_element.setValue(value)

                elif isinstance(ui_element, QSpinBox):
                    ui_element.setValue(value)
                # ... add more conditions as needed for other types

    def collect_current_config(self):
        # Collect current configuration from UI elements
        config = {}
        # Get the name of the active configuration from the dropdown
        active_config_name = self.config_dropdown.currentText()

        # Load the keys of the active configuration
        for key in self.config_manager.get_config(active_config_name).keys():
            # Check if the attribute exists
            if hasattr(self, key):
                ui_element = getattr(self, key)
                value = None
                if isinstance(ui_element, QSlider):
                    value = ui_element.value()

                elif isinstance(ui_element, QSpinBox):
                    value = ui_element.value()

                elif isinstance(ui_element, QCheckBox):
                    value = ui_element.isChecked()
                else:
                    print(f"Warning: Unknown UI element type for key {key}")
                    raise NotImplementedError

                # Debugging print
                print(f"Key: {key}, Value: {value}")

                # Validate value
                if value is None:
                    print(
                        f"Warning: No value found for key {key}. Setting default value.")
                    value = 0  # Replace with an appropriate default value

                config[key] = value
        return config

    def onConfigDropdownChanged(self, index):
        # update flag
        self.load_config(index)

    def load_config(self, index):
        config_name = self.config_dropdown.itemText(index)
        config = self.config_manager.get_config(config_name)
        if config:
            self.apply_config(config)
            self.initial_config = config
            self.current_config = config
            self.checkForConfigChanges()

    def save_config(self):
        name = self.config_name_edit.text()
        if name:
            self.current_config = self.collect_current_config()
            self.config_manager.add_config(name, self.current_config)
            self.initial_config = self.current_config.copy()
            self.refresh_config_list()
            self.config_name_edit.clear()
            self.checkForConfigChanges()

    def delete_config(self):
        name = self.config_dropdown.currentText()
        self.config_manager.delete_config(name)
        self.refresh_config_list()
        self.initial_config = None
        self.checkForConfigChanges()

    def refresh_config_list(self):
        self.config_dropdown.clear()
        self.config_dropdown.addItems(
            list(self.config_manager.configurations.keys()))

    def checkForConfigChanges(self):
        current_config = self.collect_current_config()
        if current_config != self.initial_config:
            self.config_modified = True
            print(
                f"Config modified: {self.config_modified} - {current_config} - {self.initial_config}")
        else:
            self.config_modified = False
        self.updateConfigUI()

    def initControlButtons(self):
        self.control_layout = QHBoxLayout()

        # Play/Pause Button
        self.play_button = QPushButton('', self)
        self.play_button.setIcon(QIcon(f'{IMAGES_PATH}/play.png'))
        self.play_button.clicked.connect(self.togglePlayPause)
        self.play_button.setStyleSheet(
            "background-color: green; border-radius: 15px; ")
        self.play_button.setIconSize(QSize(30, 30))
        self.play_button.setFixedSize(300, 40)

        # Stop Button
        self.stop_button = QPushButton('', self)
        self.stop_button.setIcon(QIcon(f'{IMAGES_PATH}/stop.png'))
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stopPlayback)
        self.stop_button.setStyleSheet(
            "background-color: red; border-radius: 15px;")
        self.stop_button.setIconSize(QSize(30, 30))
        self.stop_button.setFixedSize(200, 40)

        # self.control_layout.addStretch()
        self.control_layout.addWidget(self.play_button)
        self.control_layout.addWidget(self.stop_button)

        # Create a container QWidget and set the layout
        self.control_container = QWidget()
        self.control_container.setLayout(self.control_layout)

        # Add the container widget to the main layout
        self.main_layout.addWidget(
            self.control_container, alignment=Qt.AlignBottom)

        self.isPlaying = False

    def togglePlayPause(self):
        if not self.isPlaying:
            self.play_button.setIcon(QIcon(f'{IMAGES_PATH}/pause.png'))
            # Yellow background for pause
            self.play_button.setStyleSheet("background-color: yellow;")
            self.stop_button.setVisible(True)
            self.startFlipping()
        else:
            self.play_button.setIcon(QIcon(f'{IMAGES_PATH}/play.png'))
            self.play_button.setStyleSheet(
                "background-color: green;")  # Green background for play
            self.stopFlipping()  # Stop window flipping

    def get_flipper_args(self):
        # Assuming the attributes are named the same as the config keys
        # random_chance = self.random_chance.slider.value() if hasattr(self, 'random_chance') else None
        audio_full_on_chance = self.audio_full_on_chance.value(
        ) if hasattr(self, 'audio_full_on_chance') else None
        video_full_on_chance = self.video_full_on_chance.value(
        ) if hasattr(self, 'video_full_on_chance') else None

        max_volume = self.max_volume.value() if hasattr(self, 'max_volume') else None
        cognitive_load = self.cognitive_load.value() if hasattr(
            self, 'cognitive_load') else None
        session_length = self.session_length.value() if hasattr(
            self, 'session_length') else None

        # Advanced Settings with Enable checkbox button
        # average_hold_time = self.average_hold_time_slider.value() / 1 if hasattr(self, 'average_hold_time_slider') and self.average_hold_time_checkbox.isChecked() else None
        # average_intensity = self.average_intensity_slider.value() / 1 if hasattr(self, 'average_intensity_slider') and self.average_intensity_checkbox.isChecked() else None
        # frequency_av_changes = self.av_change_frequency_slider.value() / 1 if hasattr(self, 'av_change_frequency_slider') and self.av_change_frequency_checkbox.isChecked() else None

        # # Feetch Online GPT checkbox
        # use_online_ai = self.online_ai_checkbox.isChecked()

        # Check if the checkboxes are checked
        flip_audio = self.flip_audio.isChecked() if hasattr(self, 'flip_audio') else True
        flip_video = self.flip_video.isChecked() if hasattr(self, 'flip_video') else True
        opaque_overlay = self.opaque_overlay.isChecked(
        ) if hasattr(self, 'opaque_overlay') else False
        black_overlay = self.black_overlay.isChecked(
        ) if hasattr(self, 'black_overlay') else False

        return FlipperScriptArgs(
            # reversed from backkend (100-x) cuz lazy
            audio_full_on_chance=(100-audio_full_on_chance),
            video_full_on_chance=(100-video_full_on_chance),
            max_volume=max_volume,
            cognitive_load=cognitive_load,
            session_length=session_length,
            flip_audio=flip_audio,  # Add the new arguments
            flip_video=flip_video,
            opaque_overlay=opaque_overlay,
            black_overlay=black_overlay,
        )

    def onWallpaperViewClicked(self):
        dialog = EndSessionDialog(self)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            print("Session ended")  # Replace with your end session logic
            self.stopFlipping()
        else:
            # Replace with logic to continue session
            print("Session continuation")

    def startFlipping(self):

        if not self.isPlaying:
            # check if free usage left
            if not self.account_manager.is_premium_active():
                remaining_time_s = self.time_tracker.check_time_remaining()
                if remaining_time_s <= 0:
                    # OUT OF TIME - NO PREMIUM!
                    # DISPLAY MESSAGE
                    # Display out-of-time message
                    out_of_time_dialog = OutOfTimeDialog(self)
                    out_of_time_dialog.exec_()

                    # return no time to continue
                    return
            # time left, flip
            self.time_tracker.start_session()
            self.isPlaying = True
            self.updateUIButtons()
            self.updateTrayIconMenu(True)
            self.hide()  # hide main controleler ui view

            args = self.get_flipper_args()
            history = gpt_flipper.load_history()

            self.wallpaper_view = WallpaperView(
                shared_state=self.shared_state, use_offline_images=self.USE_OFFLINE_IMAGES)
            self.wallpaper_view.clicked.connect(self.onWallpaperViewClicked)
            # bring to top
            self.wallpaper_view.show()

            # Initialize wallpaper_view if not done already
            if not self.shared_state.initialized:
                self.wallpaper_view.initUI()
                if not args.black_overlay:
                    QTimer.singleShot(1, self.wallpaper_view.fetchNewImage)
                self.shared_state.initialized = True

            # Schedule the start_flipper task

            self.shared_state.shutdown_flag = False  # Reset the shutdown flag

            def onFlipperDone():
                # Handle the future when it's done
                self._cancelFlipperTaskAndSubprocesses()
                self.stopFlipping()

            self.flipper_task = asyncio.create_task(
                gpt_flipper.start_flipper(
                    args,
                    history,
                    self.shared_state,
                    self.wallpaper_view,
                    on_exit_callback=onFlipperDone
                ))

    def _cancelFlipperTaskAndSubprocesses(self):

        # Cancel the flipping task
        if self.flipper_task and not self.flipper_task.done():
            self.flipper_task.cancel()

        # return to full_on state
        audio_task = asyncio.create_task(
            gpt_flipper.set_volume(1, DEFAULT_FULL_ON_VOLUME, 0.1))

        gpt_flipper.terminate_subprocesses()

    def stopFlipping(self):
        if self.isPlaying:

            # After user stops using the app
            self.time_tracker.end_session()

            # Shutdown flipping
            self.shared_state.shutdown_flag = True
            self.isPlaying = False
            self.updateUIButtons()
            self.updateTrayIconMenu(False)

            self._cancelFlipperTaskAndSubprocesses()

            # Close the WallpaperView window
            if self.wallpaper_view is not None:
                self.wallpaper_view.close()
                self.wallpaper_view = None  # Reset the wallpaper view

            self.show()  # Show the main UI again

        self.shared_state.initialized = False

    def stopPlayback(self):
        # Reset to play icon, hide stop button, and change color back to green
        self.play_button.setIcon(QIcon(f'{IMAGES_PATH}/play.png'))
        self.play_button.setStyleSheet(
            "background-color: green; border-radius: 15px;")  # Reset background to green
        self.stop_button.setVisible(False)
        self.isPlaying = False
        # Add any additional logic for stopping the playback here

    def initBasicSettings(self):
        # Create a container for basic settings
        self.basic_settings_container = QWidget()
        # Use QVBoxLayout for vertical alignment
        basic_settings_layout = QVBoxLayout(self.basic_settings_container)

        # # Add a title for the section
        # basic_settings_title = QLabel('Flipper Setup')
        # basic_settings_title.setFont(QFont('Arial', 16, QFont.Bold))
        # basic_settings_layout.addWidget(basic_settings_title)

        # self.random_chance = DualCategorySlider("Random", "Smart (A.I. GPT)")
        # basic_settings_layout.addWidget(self.random_chance)
        # self.random_chance.valueChanged.connect(self.checkForConfigChanges)

        # # Add Fetch New AI Configurations Checkbox
        # self.online_ai_checkbox = QCheckBox("Use Online A.I.")
        # basic_settings_layout.addWidget(self.online_ai_checkbox)
        # Add Cognitive Load Slider
        self.cognitive_load_value_label = QLabel("50%")
        self.cognitive_load = QSlider(Qt.Horizontal)
        self.cognitive_load.setRange(1, 100)
        self.cognitive_load.setValue(50)
        self.cognitive_load.valueChanged.connect(
            lambda value: (self.onSliderValueChanged('Difficulty', self.cognitive_load, 1),
                           self.cognitive_load_value_label.setText(f'{value}%')))
        cognitive_load_layout = QHBoxLayout()
        cognitive_load_layout.addWidget(QLabel("Difficulty"))

        cognitive_load_layout.addWidget(self.cognitive_load)
        cognitive_load_layout.addWidget(self.cognitive_load_value_label)
        basic_settings_layout.addLayout(cognitive_load_layout)

        # add basic settings container to main layout
        self.main_layout.addWidget(self.basic_settings_container)

    def initAdvancedSettings(self):
        # Toggle Checkbox for showing/hiding advanced settings
        self.toggle_advanced_settings = QCheckBox(
            "Show Advanced Settings", self)
        self.toggle_advanced_settings.stateChanged.connect(
            self.toggleAdvancedSettings)
        self.toggle_advanced_settings.setObjectName("advancedSettingsLabel")

        self.main_layout.addWidget(self.toggle_advanced_settings)

        # Create a container for advanced settings and set its layout
        self.advanced_settings_container = QWidget()
        self.advanced_settings_layout = QVBoxLayout(
            self.advanced_settings_container)

        # Add checkboxes for opaque overlay and black overlay
        self.opaque_overlay = QCheckBox('Opaque Overlay')
        self.opaque_overlay.setChecked(False)
        self.black_overlay = QCheckBox('Black Overlay')
        self.black_overlay.setChecked(False)

        # Add checkbox layout to advanced settings
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.opaque_overlay)
        checkbox_layout.addWidget(self.black_overlay)
        self.advanced_settings_layout.addLayout(checkbox_layout)

        audio_settings_layout = QHBoxLayout()
        self.flip_audio = QCheckBox('Audiation     ')
        self.flip_audio.setChecked(True)

        # Add DualCategorySlider for Video Flipping/Not Flipping
        self.audio_full_on_chance_label = QLabel("50%")
        self.audio_full_on_chance = QSlider(Qt.Horizontal)
        self.audio_full_on_chance.setRange(1, 100)
        self.audio_full_on_chance.setValue(50)

        def on_audio_full_on_chance_value_changed(value):
            self.audio_full_on_chance_label.setText(f'{value}%')
            self.checkForConfigChanges()

        self.audio_full_on_chance.valueChanged.connect(
            on_audio_full_on_chance_value_changed)

        audio_full_on_chance_layout = QHBoxLayout()

        audio_full_on_chance_layout.addWidget(self.audio_full_on_chance)
        audio_full_on_chance_layout.addWidget(self.audio_full_on_chance_label)

        audio_settings_layout.addWidget(self.flip_audio)
        audio_settings_layout.addLayout(audio_full_on_chance_layout)
        self.advanced_settings_layout.addLayout(audio_settings_layout)

        video_settings_layout = QHBoxLayout()
        self.flip_video = QCheckBox('Visualization')
        self.flip_video.setChecked(True)

        self.video_full_on_chance_label = QLabel("50%")
        self.video_full_on_chance = QSlider(Qt.Horizontal)
        self.video_full_on_chance.setRange(1, 100)
        self.video_full_on_chance.setValue(50)

        def on_video_full_on_chance_value_changed(value):
            self.video_full_on_chance_label.setText(f'{value}%')
            self.checkForConfigChanges()

        self.video_full_on_chance.valueChanged.connect(
            on_video_full_on_chance_value_changed)

        video_full_on_chance_layout = QHBoxLayout()

        video_full_on_chance_layout.addWidget(self.video_full_on_chance)
        video_full_on_chance_layout.addWidget(self.video_full_on_chance_label)

        video_settings_layout.addWidget(self.flip_video)
        video_settings_layout.addLayout(video_full_on_chance_layout)
        self.advanced_settings_layout.addLayout(video_settings_layout)

        # Add Session Length Spinbox
        self.session_length = QSpinBox()
        self.session_length.setRange(1, 120)
        self.session_length.setValue(60)
        session_length_layout = QHBoxLayout()
        session_length_layout.addWidget(QLabel('Session Length'))
        session_length_layout.addWidget(self.session_length)
        self.advanced_settings_layout.addLayout(session_length_layout)

       # Add Volume Control Slider
        self.max_volume_label = QLabel("90%")
        self.max_volume = QSlider(Qt.Horizontal)
        self.max_volume.setRange(0, 100)
        self.max_volume.setValue(90)

        def on_max_volume_value_changed(value):
            self.max_volume_label.setText(f'{value}%')
            self.onSliderValueChanged('Max Volume', self.max_volume, 1)

        self.max_volume.valueChanged.connect(on_max_volume_value_changed)

        max_volume_layout = QHBoxLayout()
        max_volume_layout.addWidget(QLabel('Max Volume    '))
        max_volume_layout.addWidget(self.max_volume)
        max_volume_layout.addWidget(self.max_volume_label)
        self.advanced_settings_layout.addLayout(max_volume_layout)

        # Add the advanced settings container to the main layout
        self.main_layout.addWidget(self.advanced_settings_container)

        # Initially hide the advanced settings
        self.advanced_settings_container.setVisible(False)

    def onSliderValueChanged(self, label, slider, scaling_factor):
        scaled_value = slider.value() / scaling_factor
        # print(f"{label}: {scaled_value}")  # Update this to reflect changes in the UI as needed
        self.checkForConfigChanges()  # Check for configuration changes

    def toggleAdvancedSettings(self):
        show_advanced = self.toggle_advanced_settings.isChecked()
        self.advanced_settings_container.setVisible(show_advanced)
        self.addSpacingSpacers()  # Recalculate spacers for even spacing
        self.main_layout.update()  # Force layout update

    def getStyleSheet(self):
        return '''
        * {
        font-family: 'Roboto'; /* Change the global font to Roboto */
        font-size: 16px; /* Increase the global font size by 4px */
        }
        QLabel {
            color: white;
        }
        QLabel[title="true"] { /* Special styling for titles */
            font-size: 22px; /* Increase the font size of titles */
        }
        #advancedSettingsLabel { /* Specific style for 'Show Advanced Settings' label */
            font-size: 14px; /* Even larger font size for this specific label */
            text-decoration: underline;
            
        }
        QMainWindow {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #d147a3, /* Darker shade of pink */
                                          stop:1 #5e239d); /* Darker shade of purple */
    }
        QPushButton {
            background-color: #007BFF;
            border: none;
            color: white;
            text-align: center;
            margin: 4px 2px;
            border-radius: 10px;
        }
        QPushButton:hover {
            background-color: white;
            color: black;
        }
        QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff1493, stop:1 #9400d3); /* Handle gradient: dark pink to purple */
        border: 1px solid #5c5c5c;
        width: 18px;  /* Width of the handle */
        height: 18px;  /* Height of the handle */
        margin: -8px -8px;  /* Adjust margin for circular handle */
        border-radius: 8px;  /* Circular shape */
        }
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: rgba(177, 177, 177, 0.5);  /* Semi-transparent background */
            border-radius: 4px;  /* Rounded edges */
            margin: 2px 0;
        }
        QCheckBox {
            font-size: 16px;  /* Matching global font size */

        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
        }

        QMessageBox QPushButton {
            background-color: #FF5733;
            color: white;
        }
        QMessageBox QPushButton:hover {
            background-color: #C70039;
            color: white;
        }
        QPushButton {
            border: none;
            background-color: transparent;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: rgba(255, 255, 255, 0.2);
        }
        QLineEdit {
            background-color: rgba(42, 47, 58, 0.7);  /* Semi-transparent background */
            border: none;
            color: white;
            padding: 5px;
            border-radius: 5px;
        }
        QSpinBox {
            background-color: rgba(42, 47, 58, 0.7);  /* Semi-transparent background */
            border: none;
            color: white;
            padding: 5px;
            border-radius: 5px;
        }
        QComboBox {
            background-color: rgba(42, 47, 58, 0.5);  /* Background with opacity */
            border: none;
            color: white;
            padding: 5px;
            border-radius: 5px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: rgba(42, 47, 58, 0.5);  /* Item background with opacity */
            color: white;
            selection-background-color: #007BFF;  /* Highlight color for selected item */
            min-height: 20px;  /* Adjust as needed for item height */
        }
        '''

# Usage


# async def main():
#     downloader = AsyncImageDownloader(concurrent_downloads=3)
#     await downloader.download_and_save_images('wallpaper_images', 10000)

# if __name__ == '__main__':
#     asyncio.run(main())


if __name__ == '__main__':
    # Setup asyncio event loop and app
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Use  Downloaded wallpaper view images offiline mode
    USE_OFFLINE_IMAGES = True

    main_ui = MainUI(USE_OFFLINE_IMAGES=USE_OFFLINE_IMAGES)
    app.setStyleSheet(main_ui.getStyleSheet())
    main_ui.show()

    with loop:
        sys.exit(loop.run_forever())
