"""
Configuration file for Crowd Panic Prediction System
"""

# Crowd Density Standards
SAFE_DENSITY = 0.4
MAX_DENSITY = 0.67

# Alert Thresholds
WARNING_THRESHOLD = 0.75
CRITICAL_THRESHOLD = 1.0

# Model Settings - UPDATED FOR BETTER DETECTION
YOLO_MODEL = 'yolov8s.pt'  # Changed from 'yolov8n.pt' - better for crowds
CONFIDENCE_THRESHOLD = 0.25  # Lowered from 0.5 - more sensitive
PERSON_CLASS_ID = 0

# UI Settings
APP_TITLE = "Crowd Monitor"
APP_ICON = "👥"

# Default Settings
DEFAULT_VENUE_AREA = 500

# Colors (BGR for OpenCV)
COLOR_SAFE = (76, 175, 80)
COLOR_WARNING = (255, 152, 0)
COLOR_CRITICAL = (244, 67, 54)

# Video Processing
FRAME_SKIP = 2
MAX_VIDEO_LENGTH = 300