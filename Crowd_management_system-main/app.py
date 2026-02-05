import streamlit as st
import cv2
import tempfile
import time
from utils import (
    load_detection_model,
    estimate_venue_capacity,
    count_people_in_frame,
    get_alert_status,
    annotate_frame
)

# Page config
st.set_page_config(page_title="Crowd Monitor", page_icon="👥", layout="wide")

# Title
st.title("👥 Crowd Monitor")
st.caption("AI-Powered Crowd Safety System")
st.divider()

# Load model
model = load_detection_model()

if model is None:
    st.error("Failed to load model")
    st.stop()

# Session state
if 'capacity' not in st.session_state:
    st.session_state.capacity = None

if 'capacity_calculated' not in st.session_state:
    st.session_state.capacity_calculated = False

# SECTION 1: Venue Setup
st.header("📍 Venue Setup")

col1, col2 = st.columns([3, 1])

with col1:
    area = st.number_input("Venue Area (square meters)", min_value=50, value=500, step=50)

with col2:
    st.write("")
    st.write("")
    if st.button("Calculate Capacity"):
        st.session_state.capacity = estimate_venue_capacity(area)
        st.session_state.capacity_calculated = True
        # REMOVED st.rerun() - this was causing the loop

# Show capacity if calculated
if st.session_state.capacity:
    
    # Only show success message once
    if st.session_state.capacity_calculated:
        st.success("✅ Capacity calculated!")
        st.session_state.capacity_calculated = False
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Venue Area", f"{st.session_state.capacity['area']} m²")
    col2.metric("Safe Capacity", f"{st.session_state.capacity['safe_capacity']} people")
    col3.metric("Max Capacity", f"{st.session_state.capacity['max_capacity']} people")
    
    st.divider()
    
    # SECTION 2: Monitoring
    st.header("📹 Live Monitoring")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_file = st.file_uploader("Upload crowd video", type=['mp4', 'avi', 'mov'])
    
    with col2:
        st.write("")
        st.write("")
        show_boxes = st.checkbox("Show detections", value=False)
    
    if video_file:
        # Save video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        
        st.divider()
        
        # Metrics placeholders
        col1, col2, col3 = st.columns(3)
        metric1 = col1.empty()
        metric2 = col2.empty()
        metric3 = col3.empty()
        
        st.write("")
        
        # Video placeholder
        video_display = st.empty()
        
        st.write("")
        
        # Alert placeholder
        alert_box = st.empty()
        
        # Recommendations placeholder
        rec_box = st.empty()
        
        # Progress
        progress = st.progress(0)
        
        # Process video
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        safe = st.session_state.capacity['safe_capacity']
        max_cap = st.session_state.capacity['max_capacity']
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 2 != 0:
                continue
            
            # Count people
            count = count_people_in_frame(model, frame)
            alert = get_alert_status(count, safe, max_cap)
            
            # Annotate (with or without boxes)
            if show_boxes:
                from utils import draw_detections
                annotated = draw_detections(frame, model)
            else:
                annotated = annotate_frame(frame, count, alert)
            
            # Update display
            metric1.metric("👥 Current Count", count)
            metric2.metric("📊 Status", f"{alert['status_emoji']} {alert['status']}")
            metric3.metric("📈 Capacity", f"{alert['percentage']:.0f}%")
            
            # Display video - FIXED
            video_display.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )
            
            # Alert message
            if alert['status'] == 'CRITICAL':
                alert_box.error(f"🚨 {alert['message']}")
            elif alert['status'] == 'WARNING':
                alert_box.warning(f"⚠️ {alert['message']}")
            else:
                alert_box.success(f"✅ {alert['message']}")
            
            # Recommendations
            if alert['status'] != 'SAFE':
                rec_text = "**💡 Recommendations:**\n"
                for i, rec in enumerate(alert['recommendations'][:3], 1):
                    rec_text += f"\n{i}. {rec}"
                rec_box.info(rec_text)
            else:
                rec_box.empty()
            
            # Progress
            prog = int((frame_count / total_frames) * 100)
            progress.progress(min(prog, 100))
            
            time.sleep(0.03)
        
        cap.release()
        progress.empty()
        st.success("✅ Video processing complete!")

else:
    st.info("👆 Please calculate venue capacity first")