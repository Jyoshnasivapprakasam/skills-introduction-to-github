"""
Utility functions for Crowd Panic Prediction System
"""

import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from config import *


@st.cache_resource
def load_detection_model():
    """Load YOLO model (cached)"""
    try:
        model = YOLO(YOLO_MODEL)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def estimate_venue_capacity(area_sqm=None):
    """Calculate safe and maximum crowd capacity"""
    if area_sqm is None or area_sqm <= 0:
        area_sqm = DEFAULT_VENUE_AREA
    
    safe_capacity = int(area_sqm * SAFE_DENSITY)
    max_capacity = int(area_sqm * MAX_DENSITY)
    
    return {
        'area': area_sqm,
        'safe_capacity': safe_capacity,
        'max_capacity': max_capacity
    }


def count_people_in_frame(model, frame):
    """Count number of people in a single frame - IMPROVED VERSION"""
    try:
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Run detection with LOWER confidence for crowded scenes
        results = model(frame, verbose=False, conf=0.25, iou=0.5)
        
        person_count = 0
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id == PERSON_CLASS_ID:
                        person_count += 1
        
        return person_count
        
    except Exception as e:
        print(f"Detection error: {e}")
        return 0


def get_alert_status(current_count, safe_capacity, max_capacity):
    """Determine alert level based on current crowd count"""
    percentage = (current_count / max_capacity) * 100 if max_capacity > 0 else 0
    
    if current_count >= max_capacity:
        return {
            'status': 'CRITICAL',
            'status_emoji': '🚨',
            'color': 'critical',
            'color_bgr': COLOR_CRITICAL,
            'percentage': percentage,
            'message': 'Overcrowding Detected',
            'recommendations': [
                'Stop all entries immediately',
                'Activate emergency protocols',
                'Deploy security personnel',
                'Open emergency exits',
                'Alert emergency services'
            ]
        }
    
    elif current_count >= safe_capacity:
        return {
            'status': 'WARNING',
            'status_emoji': '⚠️',
            'color': 'warning',
            'color_bgr': COLOR_WARNING,
            'percentage': percentage,
            'message': 'Approaching Capacity',
            'recommendations': [
                'Reduce entry rate',
                'Increase monitoring',
                'Prepare dispersal plan',
                'Alert security team',
                'Monitor crowd flow'
            ]
        }
    
    else:
        return {
            'status': 'SAFE',
            'status_emoji': '✅',
            'color': 'safe',
            'color_bgr': COLOR_SAFE,
            'percentage': percentage,
            'message': 'Normal Operations',
            'recommendations': [
                'Continue monitoring',
                'Maintain regular checks'
            ]
        }


def annotate_frame(frame, count, alert_info):
    """Add text annotations to video frame"""
    annotated = frame.copy()
    
    # Add semi-transparent background for text
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (500, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    
    # Add count
    cv2.putText(annotated, f"Count: {count}", (15, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    
    # Add status
    status_text = f"{alert_info['status_emoji']} {alert_info['status']}"
    cv2.putText(annotated, status_text, (15, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, alert_info['color_bgr'], 3)
    
    # Add percentage
    cv2.putText(annotated, f"{alert_info['percentage']:.1f}% of max", 
                (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, alert_info['color_bgr'], 2)
    
    return annotated


def draw_detections(frame, model):
    """Draw bounding boxes around detected people - IMPROVED VERSION"""
    try:
        height, width = frame.shape[:2]
        
        # Resize if needed
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            resized = cv2.resize(frame, (new_width, new_height))
        else:
            resized = frame.copy()
            scale = 1.0
        
        # Detect with lower confidence
        results = model(resized, verbose=False, conf=0.25, iou=0.5)
        
        annotated = resized.copy()
        person_count = 0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    
                    if class_id == PERSON_CLASS_ID:
                        person_count += 1
                        
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Draw box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence
                        label = f"{conf:.2f}"
                        cv2.putText(annotated, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add count overlay
        cv2.rectangle(annotated, (10, 10), (250, 60), (0, 0, 0), -1)
        cv2.putText(annotated, f"Detected: {person_count}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize back
        if width > 1280:
            annotated = cv2.resize(annotated, (width, height))
        
        return annotated
        
    except Exception as e:
        print(f"Drawing error: {e}")
        return frame