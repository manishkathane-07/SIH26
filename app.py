"""
VISTA-UGV: Vision-Based Autonomous Navigation System
Smart India Hackathon (SIH26 PS 26126)

Advanced Tactical Ground Control Station (GCS) Dashboard
"""

import os
import tempfile
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.odometry import VisualOdometry
from core.perception import Perception
from core.planner import (
    build_costmap,
    command,
    draw_path,
    dynamic_window_plan,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Tactical CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VISTA-UGV | Tactical Ground Control Station",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Tech Tactical CSS
st.markdown(
    """
    <style>
        /* Import Futuristic Font */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Rajdhani:wght@500;600;700&display=swap');

        /* Root Palette */
        :root {
            --bg-dark: #0A0E17;
            --card-bg: rgba(18, 24, 38, 0.85);
            --accent-cyan: #00F0FF;
            --accent-green: #00FF66;
            --accent-orange: #FF9900;
            --accent-red: #FF0055;
            --text-muted: #8A99AD;
            --border-glow: rgba(0, 240, 255, 0.25);
        }

        /* Hide default Streamlit headers & footers */
        header[data-testid="stHeader"] { visibility: hidden; height: 0px; }
        .stDeployButton, div[data-testid="stAppDeployButton"] { display: none !important; }
        #MainMenu, footer { visibility: hidden; }

        /* Main Container Background */
        .stApp {
            background-color: var(--bg-dark);
            font-family: 'Rajdhani', sans-serif;
            color: #E2E8F0;
        }

        /* Top Header GCS Banner */
        .gcs-header {
            background: linear-gradient(135deg, rgba(18, 24, 38, 0.95), rgba(10, 14, 23, 0.95));
            border: 1px solid var(--border-glow);
            border-left: 4px solid var(--accent-cyan);
            border-radius: 8px;
            padding: 14px 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .gcs-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: 1.5px;
            color: var(--accent-cyan);
            margin: 0;
            text-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
        }
        .gcs-subtitle {
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Status Badge Pills */
        .badge-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 1px;
            margin-left: 8px;
        }
        .badge-operational {
            background: rgba(0, 255, 102, 0.15);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
        }
        .badge-warning {
            background: rgba(255, 153, 0, 0.15);
            color: var(--accent-orange);
            border: 1px solid var(--accent-orange);
        }
        .badge-alert {
            background: rgba(255, 0, 85, 0.15);
            color: var(--accent-red);
            border: 1px solid var(--accent-red);
            animation: pulse-red 1.5s infinite;
        }
        @keyframes pulse-red {
            0% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0.4); }
            70% { box-shadow: 0 0 0 8px rgba(255, 0, 85, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0); }
        }

        /* Card Panels */
        div[data-testid="stMetricValue"] {
            font-family: 'Orbitron', sans-serif !important;
            font-size: 1.5rem !important;
            color: var(--accent-cyan) !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: var(--text-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Tabs Custom Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: rgba(18, 24, 38, 0.6);
            padding: 6px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            border-radius: 6px;
            color: var(--text-muted);
            font-family: 'Orbitron', sans-serif;
            font-size: 0.82rem;
            letter-spacing: 0.8px;
            padding: 0px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--accent-cyan) !important;
            color: #0A0E17 !important;
            font-weight: 700 !important;
            box-shadow: 0 0 12px rgba(0, 240, 255, 0.4);
        }

        /* DataFrame Styling */
        .stDataFrame {
            border: 1px solid rgba(0, 240, 255, 0.2);
            border-radius: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "mission_start_time" not in st.session_state:
    st.session_state.mission_start_time = time.time()
if "event_logs" not in st.session_state:
    st.session_state.event_logs = []
if "playback_paused" not in st.session_state:
    st.session_state.playback_paused = False


# -----------------------------------------------------------------------------
# 3. Model Caching & Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def load_perception_model() -> Perception:
    """Initialize and cache the perception neural network module."""
    return Perception()


perception = load_perception_model()
if "vo_tracker" not in st.session_state:
    st.session_state.vo_tracker = VisualOdometry()

visual_odometry = st.session_state.vo_tracker


# -----------------------------------------------------------------------------
# 4. Top GCS Header Bar
# -----------------------------------------------------------------------------
def render_gcs_header(status_mode: str = "OPERATIONAL", max_risk: float = 0.0):
    elapsed_sec = int(time.time() - st.session_state.mission_start_time)
    elapsed_str = f"{elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}"

    if max_risk > 0.75:
        badge_html = '<span class="badge-pill badge-alert">CRITICAL COLLISION RISK</span>'
    elif max_risk > 0.45:
        badge_html = '<span class="badge-pill badge-warning">CAUTION - OBSTACLE DETECTED</span>'
    else:
        badge_html = '<span class="badge-pill badge-operational">ALL SYSTEMS NOMINAL</span>'

    st.markdown(
        f"""
        <div class="gcs-header">
            <div>
                <div class="gcs-title">🛡️ VISTA-UGV | GROUND CONTROL STATION</div>
                <div class="gcs-subtitle">Team TechPulse — Vision-First Autonomous Navigation System </div>
            </div>
            <div style="text-align: right;">
                <span style="font-family: Orbitron; font-size: 0.85rem; color: #8A99AD; margin-right: 12px;">MISSION CLOCK: <strong style="color: #00F0FF;">{elapsed_str}</strong></span>
                <span style="font-family: Orbitron; font-size: 0.85rem; color: #8A99AD; margin-right: 12px;">MODE: <strong style="color: #00FF66;">{status_mode}</strong></span>
                {badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_gcs_header()


# -----------------------------------------------------------------------------
# 5. Sidebar Tactical Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Mission Control")

    preset_video_path = os.path.join(os.getcwd(), "20260826_161224.mp4")
    has_preset = os.path.exists(preset_video_path)

    source_options = []
    if has_preset:
        source_options.append("Sample Demo Video (Pre-loaded)")
    source_options.extend([
        "External Mobile Camera (IP / RTSP Stream)",
        "Upload Video File",
        "Live Webcam",
    ])

    input_source = st.selectbox(
        label="Video Feed Source",
        options=source_options,
        index=0 if has_preset else 1,
    )

    mobile_cam_url = ""
    if input_source == "External Mobile Camera (IP / RTSP Stream)":
        mobile_cam_url = st.text_input(
            label="Mobile Camera Stream URL",
            value="http://192.168.1.100:8080/video",
            help="Enter HTTP/MJPEG URL (e.g. IP Webcam, DroidCam) or RTSP URL.",
        )
        st.markdown(
            """
            <div style="background: rgba(0, 240, 255, 0.05); border-left: 3px solid #00F0FF; padding: 8px 12px; border-radius: 4px; font-size: 0.8rem; margin-top: -6px; margin-bottom: 12px;">
                <strong>📱 Quick Setup Guide:</strong><br>
                1. Install <b>IP Webcam</b> or <b>DroidCam</b> on Android/iOS.<br>
                2. Connect phone & PC to the same Wi-Fi network.<br>
                3. Paste stream link (e.g. <code>http://192.168.1.50:8080/video</code>).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### 🤖 Autonomy & Planner")
    is_autonomous = st.toggle(
        label="Autonomous Mode",
        value=True,
        help="Enable dynamic window path planning and automated steering control.",
    )

    target_bias = st.select_slider(
        label="Target Steering Bias",
        options=[-0.6, -0.3, 0.0, 0.3, 0.6],
        value=0.0,
        format_func=lambda x: "LEFT" if x < -0.1 else ("RIGHT" if x > 0.1 else "CENTER"),
        help="Guide the UGV trajectory towards a virtual target offset.",
    )

    confidence_threshold = st.slider(
        label="YOLO Detection Threshold",
        min_value=0.15,
        max_value=0.85,
        value=0.30,
        step=0.05,
    )

    st.divider()

    st.markdown("### 👁️ Perception & HUD Displays")
    enable_hud = st.toggle("Tactical HUD Reticle", value=True)
    show_traversability = st.toggle("Show Traversability Overlay", value=False)
    draw_candidate_arcs = st.toggle("Draw DWA Candidate Arcs", value=True)

    st.divider()

    st.markdown("### 🔄 Odometry Controls")
    if st.button("Reset Visual Odometry", use_container_width=True):
        visual_odometry.reset()
        st.toast("Visual Odometry pose & trajectory reset.")

    playback_delay = st.slider("Playback Speed Delay (ms)", min_value=0, max_value=100, value=20, step=5)


# -----------------------------------------------------------------------------
# 6. Video Capture Source Resolution
# -----------------------------------------------------------------------------
video_capture = None
temp_file_path = None

if input_source == "Sample Demo Video (Pre-loaded)" and has_preset:
    video_capture = cv2.VideoCapture(preset_video_path)
elif input_source == "External Mobile Camera (IP / RTSP Stream)":
    if not mobile_cam_url.strip():
        st.warning("⚠️ Please enter a valid Mobile Camera Stream URL in the sidebar.")
        st.stop()
    video_capture = cv2.VideoCapture(mobile_cam_url.strip())
elif input_source == "Upload Video File":
    uploaded_file = st.file_uploader(
        label="Upload Outdoor Navigation Video (MP4 / AVI / MOV)",
        type=["mp4", "avi", "mov", "mkv"],
    )
    if not uploaded_file:
        st.info("💡 Upload a video file or select 'Sample Demo Video' from the sidebar to begin execution.")
        st.stop()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name
    temp_file.close()
    video_capture = cv2.VideoCapture(temp_file_path)
else:
    webcam_idx = st.number_input("Webcam Device Index", min_value=0, max_value=5, value=0)
    video_capture = cv2.VideoCapture(int(webcam_idx))

if not video_capture or not video_capture.isOpened():
    if input_source == "External Mobile Camera (IP / RTSP Stream)":
        st.error(f"❌ Failed to connect to Mobile Camera Stream at: `{mobile_cam_url}`\n\nPlease make sure:\n1. The streaming app (e.g. IP Webcam) is running on your phone.\n2. Both phone and computer are connected to the same Wi-Fi.\n3. The IP address and port match your app screen.")
    else:
        st.error("Failed to open video source feed.")
    st.stop()


# -----------------------------------------------------------------------------
# 7. Dashboard Layout Structure (Tabs)
# -----------------------------------------------------------------------------
tab_mission, tab_odometry, tab_perception, tab_logs = st.tabs([
    "🎯 MISSION CONTROL",
    "🗺️ TACTICAL MAP & VO",
    "📊 AI PERCEPTION & HAZARDS",
    "📜 MISSION EVENT LOG",
])

with tab_mission:
    # Top Telemetry Cards Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    metric_cmd_slot = m_col1.empty()
    metric_speed_slot = m_col2.empty()
    metric_risk_slot = m_col3.empty()
    metric_vo_slot = m_col4.empty()
    metric_fps_slot = m_col5.empty()

    st.markdown("---")

    # Feeds Layout
    feed_left, feed_right = st.columns([1.6, 1.0])
    with feed_left:
        st.markdown("##### 📹 PRIMARY CAMERA FEED [TACTICAL HUD & DWA TRAJECTORY]")
        main_stream_slot = st.empty()

    with feed_right:
        st.markdown("##### 🌋 MONOCULAR DEPTH ESTIMATION")
        depth_stream_slot = st.empty()
        st.markdown("##### 🗺️ LOCAL COSTMAP & CANDIDATE ARCS")
        costmap_stream_slot = st.empty()

with tab_odometry:
    map_col1, map_col2 = st.columns([1.5, 1.0])
    with map_col1:
        st.markdown("##### 📍 REAL-TIME 2D POSITION TRAJECTORY (VISUAL ODOMETRY)")
        plotly_map_slot = st.empty()
    with map_col2:
        st.markdown("##### 📈 TELEMETRY ANALYTICS & SPEED PROFILE")
        plotly_speed_chart_slot = st.empty()
        plotly_risk_chart_slot = st.empty()

with tab_perception:
    st.markdown("##### ⚠️ LIVE DETECTED HAZARDS & OBSTACLE BREAKDOWN")
    hazard_table_slot = st.empty()
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("##### 🟢 TRAVERSABLE FREE-SPACE REGION")
        freespace_slot = st.empty()
    with p_col2:
        st.markdown("##### 🎯 ORB FEATURE TRACKING OVERLAY")
        orb_features_slot = st.empty()

with tab_logs:
    st.markdown("##### 📋 SYSTEM EVENT TELEMETRY LOG")
    event_log_table_slot = st.empty()


# -----------------------------------------------------------------------------
# 8. Frame Processing Execution Loop
# -----------------------------------------------------------------------------
trajectory_history_log: List[Dict[str, Any]] = []
time_history: List[float] = []
speed_history: List[float] = []
risk_history: List[float] = []
last_frame_time = time.time()

frame_counter = 0

while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        # Loop video feed seamlessly if pre-loaded demo video
        if input_source == "Sample Demo Video (Pre-loaded)" and has_preset:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            break

    frame_counter += 1
    frame = cv2.resize(frame, (850, 480))
    current_time = time.time()

    # 1. Perception Inference (YOLO + Free-space + Distance)
    obstacles, free_space, annotated_frame = perception.run(
        frame,
        conf=confidence_threshold,
        enable_hud=enable_hud,
    )

    # 2. Monocular Depth Estimation Map
    depth_map = perception.estimate_depth_map(frame)

    # 3. Visual Odometry Update
    pose = visual_odometry.update(frame)

    # 4. Local Costmap & Dynamic Window Trajectory Planning
    costmap = build_costmap(free_space, obstacles)
    best_path, steering, target_speed, candidate_paths = dynamic_window_plan(costmap, target_bias=target_bias)

    # 5. Collision Risk & Control Signal Command
    max_risk = max([obs["risk"] for obs in obstacles], default=0.0)
    nav_cmd, cmd_speed = command(steering, target_speed, max_risk, pose["confidence"])

    if not is_autonomous:
        nav_cmd = "MANUAL STANDBY"
        cmd_speed = 0.0

    # 6. Render Trajectory Path onto HUD frame
    display_frame = draw_path(
        annotated_frame,
        best_path,
        candidate_paths=candidate_paths if draw_candidate_arcs else None,
        draw_candidates=draw_candidate_arcs,
    )

    # Optional Traversability Overlay
    if show_traversability:
        trav_overlay = cv2.cvtColor(free_space, cv2.COLOR_GRAY2BGR)
        trav_overlay[:, :, 1] = np.maximum(trav_overlay[:, :, 1], free_space)
        display_frame = cv2.addWeighted(display_frame, 0.70, trav_overlay, 0.30, 0.0)

    # Compute FPS
    delta = max(current_time - last_frame_time, 1e-3)
    fps = 1.0 / delta
    last_frame_time = current_time

    # Update Telemetry Metric Cards
    metric_cmd_slot.metric("NAV COMMAND", nav_cmd)
    metric_speed_slot.metric("SPEED TARGET", f"{cmd_speed:.2f} m/s")
    metric_risk_slot.metric("COLLISION RISK", f"{max_risk * 100:.0f}%")
    metric_vo_slot.metric("VO CONFIDENCE", f"{pose['confidence'] * 100:.0f}%", delta=f"{pose['inliers']} inliers")
    metric_fps_slot.metric("SYSTEM FPS", f"{fps:.1f}")

    # Render Stream Video Slots
    main_stream_slot.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    depth_stream_slot.image(cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Generate Heatmap for Costmap
    normalized_cost = cv2.normalize(costmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cost_heatmap = cv2.applyColorMap(255 - normalized_cost, cv2.COLORMAP_TURBO)
    if len(best_path) > 1:
        traj_pts = np.array(
            [(int(c / costmap.shape[1] * cost_heatmap.shape[1]), int(r / costmap.shape[0] * cost_heatmap.shape[0]))
             for r, c in best_path],
            dtype=np.int32,
        )
        cv2.polylines(cost_heatmap, [traj_pts], False, (255, 255, 255), 3)

    costmap_stream_slot.image(cv2.cvtColor(cost_heatmap, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Store telemetry trends
    rel_time = round(current_time - st.session_state.mission_start_time, 1)
    time_history.append(rel_time)
    speed_history.append(cmd_speed)
    risk_history.append(max_risk * 100)

    if len(time_history) > 60:
        time_history.pop(0)
        speed_history.pop(0)
        risk_history.pop(0)

    # Render Plotly 2D Map (Every 2 frames to optimize performance)
    if frame_counter % 2 == 0:
        hist = pose["history"]
        hx = [p[0] for p in hist]
        hy = [p[1] for p in hist]

        fig_map = go.Figure()
        # Trajectory trail
        fig_map.add_trace(go.Scatter(
            x=hx, y=hy,
            mode='lines+markers',
            name='UGV Path',
            line=dict(color='#00F0FF', width=3),
            marker=dict(size=4, color='#00FF66')
        ))
        # Current Position
        fig_map.add_trace(go.Scatter(
            x=[pose["x"]], y=[pose["y"]],
            mode='markers',
            name='Current Position',
            marker=dict(size=14, color='#FF0055', symbol='triangle-up')
        ))
        fig_map.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(10, 14, 23, 0.8)',
            plot_bgcolor='rgba(18, 24, 38, 0.9)',
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="X Position (Relative)",
            yaxis_title="Y Position (Relative)",
            height=360,
        )
        plotly_map_slot.plotly_chart(fig_map, use_container_width=True)

        # Plotly Speed Chart
        fig_speed = px.line(x=time_history, y=speed_history, labels={'x': 'Time (s)', 'y': 'Speed (m/s)'}, title="Speed Profile (m/s)")
        fig_speed.update_traces(line_color='#00FF66', line_width=2)
        fig_speed.update_layout(template="plotly_dark", paper_bgcolor='rgba(10, 14, 23, 0.8)', height=180, margin=dict(l=20, r=20, t=30, b=20))
        plotly_speed_chart_slot.plotly_chart(fig_speed, use_container_width=True)

        # Plotly Risk Chart
        fig_risk = px.area(x=time_history, y=risk_history, labels={'x': 'Time (s)', 'y': 'Risk %'}, title="Collision Risk %")
        fig_risk.update_traces(line_color='#FF0055', fillcolor='rgba(255, 0, 85, 0.2)')
        fig_risk.update_layout(template="plotly_dark", paper_bgcolor='rgba(10, 14, 23, 0.8)', height=180, margin=dict(l=20, r=20, t=30, b=20))
        plotly_risk_chart_slot.plotly_chart(fig_risk, use_container_width=True)

        # Render Hazard Table
        if obstacles:
            hazard_data = [
                {
                    "Hazard Class": obs["label"].upper(),
                    "Confidence": f"{obs['confidence']*100:.1f}%",
                    "Distance (Est)": f"{obs['distance_m']:.1f} m",
                    "Risk Index": f"{obs['risk']*100:.0f}%",
                    "Status": "AVOIDING" if obs["risk"] > 0.5 else "MONITORING"
                }
                for obs in obstacles
            ]
            hazard_df = pd.DataFrame(hazard_data)
            hazard_table_slot.dataframe(hazard_df, use_container_width=True)
        else:
            hazard_table_slot.info("No hazards currently detected in local range.")

        # Render Auxiliary Perception Views
        freespace_slot.image(free_space, caption="Binary Ground Traversability Mask", use_container_width=True)
        orb_overlay = visual_odometry.draw_features(frame)
        orb_features_slot.image(cv2.cvtColor(orb_overlay, cv2.COLOR_BGR2RGB), caption="ORB Feature Keypoints", use_container_width=True)

        # Telemetry Event Logging
        log_entry = {
            "Timestamp": time.strftime("%H:%M:%S"),
            "Command": nav_cmd,
            "Speed": round(cmd_speed, 2),
            "Risk %": round(max_risk * 100, 1),
            "Pose (X, Y)": f"({pose['x']:.2f}, {pose['y']:.2f})",
            "Heading": f"{pose['yaw']:.2f} rad",
        }
        st.session_state.event_logs.append(log_entry)
        if len(st.session_state.event_logs) > 50:
            st.session_state.event_logs.pop(0)

        event_log_table_slot.dataframe(pd.DataFrame(st.session_state.event_logs), use_container_width=True)

    # Frame Pacing Delay
    if playback_delay > 0:
        time.sleep(playback_delay / 1000.0)

# Release resources
video_capture.release()
if temp_file_path and os.path.exists(temp_file_path):
    try:
        os.remove(temp_file_path)
    except Exception:
        pass