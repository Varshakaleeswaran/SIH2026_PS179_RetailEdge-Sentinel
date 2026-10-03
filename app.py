"""
RetailEdge Sentinel - Streamlit Edge Operations Dashboard
CAMERA -> DETECT -> ANALYSE -> ALERT -> ACTION
SIH 2026 - Problem Statement ID: SIH26179 | Team: VENSMOLD AI
"""

import os
import sys
import time
from pathlib import Path
import yaml
import cv2
import numpy as np
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import RetailPipeline
from src.privacy import PrivacyGuard
from src.database import DatabaseManager
from dashboard.components import (
    inject_custom_css,
    render_header,
    render_alert_banner,
    render_kpi_cards,
    render_recommendations_list,
    render_edge_deployment_panel,
    render_privacy_panel
)
from dashboard.charts import (
    render_shelf_status_cards,
    render_risk_breakdown,
    render_metrics_trends
)

# Page configuration
st.set_page_config(
    page_title="RetailEdge Sentinel | Edge Retail Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern UI styling
inject_custom_css()

# Load configuration
@st.cache_resource
def load_settings():
    cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}

settings = load_settings()

# Initialize or retrieve pipeline in session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RetailPipeline(settings)

pipeline: RetailPipeline = st.session_state.pipeline
db: DatabaseManager = pipeline.db

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shop.png", width=64)
    st.title("Control Station")
    st.caption("RetailEdge Sentinel v1.0-MVP (SIH 2026)")

    st.subheader("Video Source Selection")
    source_type = st.radio(
        "Input Mode:",
        [
            "Recorded CCTV Dataset (Primary MVP Mode)",
            "Webcam (Live Camera 0)",
            "Custom Video File / RTSP Stream",
            "Demo Scenario Simulator"
        ],
        index=0,
        help="Recorded CCTV datasets run real-time on-device computer vision and tracking."
    )

    dataset_video_path = ""
    if source_type == "Recorded CCTV Dataset (Primary MVP Mode)":
        dataset_choice = st.selectbox(
            "Select Retail Dataset Clip:",
            [
                "🏪 Retail Store Aisle (videos/store_aisle.mp4)",
                "🚶 Customer Movement & Traffic (videos/people_cctv.mp4)",
                "🚪 Entrance & Footfall Gate (videos/entrance_footfall.mp4)",
                "🛒 Synthetic Multi-Zone Store Floor (videos/sample_store.mp4)"
            ],
            index=0
        )
        if "store_aisle" in dataset_choice:
            dataset_video_path = str(PROJECT_ROOT / "videos" / "store_aisle.mp4")
        elif "people_cctv" in dataset_choice:
            dataset_video_path = str(PROJECT_ROOT / "videos" / "people_cctv.mp4")
        elif "entrance_footfall" in dataset_choice:
            dataset_video_path = str(PROJECT_ROOT / "videos" / "entrance_footfall.mp4")
        else:
            dataset_video_path = str(PROJECT_ROOT / "videos" / "sample_store.mp4")

    elif source_type == "Custom Video File / RTSP Stream":
        dataset_video_path = st.text_input("Path or RTSP URL:", "videos/store_aisle.mp4")

    st.markdown("---")
    st.subheader("Inference Settings")
    conf_thresh = st.slider("Detection Confidence:", min_value=0.15, max_value=0.85, value=0.35, step=0.05)
    pipeline.detector.confidence_thresh = conf_thresh

    adaptive_mode = st.toggle("Adaptive Edge Inference", value=True, help="Dynamically adjusts frame skips based on scene activity.")
    pipeline.adaptive_enabled = adaptive_mode

    st.markdown("---")
    st.subheader("Interactive Shelf Controls")
    st.caption("Trigger inventory replenishment in real-time:")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("Restock Shelf B", use_container_width=True):
            pipeline.shelf_intel.replenish_shelf("SHELF_B")
            st.success("Shelf B Restocked!")
    with col_sb2:
        if st.button("Deplete Shelf B", use_container_width=True):
            pipeline.shelf_intel.set_shelf_status("SHELF_B", "EMPTY", 0)
            st.warning("Shelf B Depleted!")

    st.markdown("---")
    st.subheader("Edge Hardware Profile")
    st.info(
        "**Current Mode:** Local Laptop Edge Simulation\n\n"
        "**Target Hardware:** Qualcomm Dragonwing Edge AI Platform\n\n"
        "**Inference:** Ultralytics YOLOv8n (FP32 CPU)\n\n"
        "**Target Runtime:** Qualcomm QAIRT / QNN"
    )

    st.caption("VENSMOLD AI — SIH 2026")

# ----------------- MAIN INTERFACE -----------------
render_header()

if source_type == "Demo Scenario Simulator":
    st.markdown("""
    <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 10px 14px; margin-bottom: 15px;">
        <span style="font-weight: 700; color: #1D4ED8;">DEMO SIMULATION SCENARIO ACTIVE:</span>
        <span style="color: #1E40AF; font-size: 0.9rem;">
            Demonstrating retail state transitions, risk scoring, and staff action recommendations without video playback.
        </span>
    </div>
    """, unsafe_allow_html=True)

    demo_scenario = st.selectbox(
        "Select Demonstration Scenario:",
        [
            "Scenario A: Normal Store Flow (Balanced Footfall & Stock)",
            "Scenario B: Queue Buildup & Operational Stress (Rush Hour)",
            "Scenario C: Critical Bottleneck (Surge + Empty Shelf B)"
        ],
        index=1
    )

    if "Scenario A" in demo_scenario:
        sim_shopper = {
            "footfall": 42,
            "occupancy": 3,
            "entrance_occupancy": 1,
            "shelf_occupancy": 1,
            "billing_occupancy": 1,
            "average_dwell_time": 35.2,
            "traffic_level": "LOW",
            "zone_breakdown": {"ENTRANCE": 1, "SHELF": 1, "BILLING": 1}
        }
        sim_shelf = {
            "shelf_status": "AVAILABLE",
            "low_shelves": 0,
            "empty_shelves": 0,
            "replenishment_required": False,
            "shelves": {
                "SHELF_A": {"name": "Fresh Produce / Beverages", "status": "AVAILABLE", "fill_percentage": 90},
                "SHELF_B": {"name": "Snacks & Packaged Goods", "status": "AVAILABLE", "fill_percentage": 82},
                "SHELF_C": {"name": "Personal Care & Household", "status": "AVAILABLE", "fill_percentage": 95}
            },
            "estimation_label": "Simulated Demo Inventory State"
        }
        sim_queue = {
            "queue_length": 1,
            "queue_level": "LOW",
            "queue_growth": 0,
            "growth_trend": "STABLE",
            "estimated_wait_minutes": 0.7,
            "waiting_time_trend": "STABLE",
            "disclaimer": "Simulated Metric"
        }
    elif "Scenario B" in demo_scenario:
        sim_shopper = {
            "footfall": 126,
            "occupancy": 9,
            "entrance_occupancy": 2,
            "shelf_occupancy": 3,
            "billing_occupancy": 4,
            "average_dwell_time": 54.8,
            "traffic_level": "HIGH",
            "zone_breakdown": {"ENTRANCE": 2, "SHELF": 3, "BILLING": 4}
        }
        sim_shelf = pipeline.shelf_intel.process(shelf_occupancy=3)
        sim_queue = {
            "queue_length": 7,
            "queue_level": "HIGH",
            "queue_growth": 4,
            "growth_trend": "INCREASING",
            "estimated_wait_minutes": 4.7,
            "waiting_time_trend": "INCREASING",
            "disclaimer": "Simulated Metric"
        }
    else:
        sim_shopper = {
            "footfall": 158,
            "occupancy": 14,
            "entrance_occupancy": 3,
            "shelf_occupancy": 4,
            "billing_occupancy": 7,
            "average_dwell_time": 68.0,
            "traffic_level": "HIGH",
            "zone_breakdown": {"ENTRANCE": 3, "SHELF": 4, "BILLING": 7}
        }
        sim_shelf = {
            "shelf_status": "CRITICAL_EMPTY",
            "low_shelves": 1,
            "empty_shelves": 1,
            "replenishment_required": True,
            "shelves": {
                "SHELF_A": {"name": "Fresh Produce / Beverages", "status": "LOW", "fill_percentage": 30},
                "SHELF_B": {"name": "Snacks & Packaged Goods", "status": "EMPTY", "fill_percentage": 0},
                "SHELF_C": {"name": "Personal Care & Household", "status": "AVAILABLE", "fill_percentage": 80}
            },
            "estimation_label": "Simulated Demo Inventory State"
        }
        sim_queue = {
            "queue_length": 9,
            "queue_level": "HIGH",
            "queue_growth": 5,
            "growth_trend": "INCREASING",
            "estimated_wait_minutes": 6.0,
            "waiting_time_trend": "INCREASING",
            "disclaimer": "Simulated Metric"
        }

    store_state = pipeline.state_engine.evaluate(sim_shopper, sim_shelf, sim_queue)
    risk_metrics = pipeline.risk_engine.compute_risk(sim_shopper, sim_shelf, sim_queue)
    recommendations = pipeline.rec_engine.generate_recommendations(store_state, risk_metrics, sim_shelf)

    render_alert_banner(store_state, risk_metrics)
    render_kpi_cards(sim_shopper, sim_queue, sim_shelf, risk_metrics)

    c_left, c_right = st.columns([1.1, 0.9])
    with c_left:
        st.subheader("Live CCTV Video Feed (Annotated)")
        sample_path = PROJECT_ROOT / "videos" / "store_aisle.mp4"
        if sample_path.exists():
            cap = cv2.VideoCapture(str(sample_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                annotated, _ = pipeline.process_frame(frame)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Camera 01: Retail Floor Zones & Anonymous Tracking HUD")
        
        st.subheader("Shopper Intelligence Analytics")
        st.markdown(f"""
        - **Total Verified Footfall:** `{sim_shopper['footfall']}` shoppers
        - **Current Active Occupancy:** `{sim_shopper['occupancy']}` shoppers
        - **Store Traffic Level:** `{sim_shopper['traffic_level']}`
        - **Average Dwell Time:** `{sim_shopper['average_dwell_time']}` seconds
        - **Zone Occupancies:** Entrance: `{sim_shopper['entrance_occupancy']}` | Shelf Aisle: `{sim_shopper['shelf_occupancy']}` | Billing Counter: `{sim_shopper['billing_occupancy']}`
        """)

    with c_right:
        st.subheader("Queue Intelligence (Billing Counter)")
        q_len = sim_queue["queue_length"]
        q_growth = sim_queue["queue_growth"]
        q_trend = sim_queue["growth_trend"]
        q_wait = sim_queue["estimated_wait_minutes"]
        q_lvl = sim_queue["queue_level"]

        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #334155;">Queue Severity Level:</span>
                <span style="font-weight: 700; color: {'#EF4444' if q_lvl=='HIGH' else ('#F59E0B' if q_lvl=='MEDIUM' else '#10B981')};">{q_lvl}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #334155;">Active Customers in Queue:</span>
                <span style="font-weight: 700;">{q_len}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #334155;">Queue Growth Trend:</span>
                <span style="font-weight: 700;">{q_trend} ({'+' if q_growth > 0 else ''}{q_growth})</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-weight: 600; color: #334155;">Est. Wait Time:</span>
                <span style="font-weight: 700;">~{q_wait} mins <span style="font-size: 0.75rem; color: #94A3B8;">({sim_queue['disclaimer']})</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_risk_breakdown(risk_metrics)

    render_shelf_status_cards(sim_shelf)
    render_recommendations_list(recommendations)

else:
    # ----------------- RECORDED CCTV DATASET / WEBCAM PIPELINE -----------------
    video_source = 0 if source_type == "Webcam (Live Camera 0)" else dataset_video_path

    # Stream Controls
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.5, 1, 1])
    with c_ctrl1:
        run_stream = st.checkbox("▶️ Run Continuous Edge Pipeline", value=True)
    with c_ctrl2:
        loop_video = st.checkbox("🔁 Loop Video File", value=True)
    with c_ctrl3:
        frame_limit = st.selectbox("Process Batch:", [50, 100, 200, "Continuous"], index=1)

    frame_placeholder = st.empty()
    alert_placeholder = st.empty()
    kpi_placeholder = st.empty()

    col_vid_left, col_vid_right = st.columns([1.2, 0.8])
    with col_vid_left:
        sub_shopper_placeholder = st.empty()
    with col_vid_right:
        sub_queue_placeholder = st.empty()
        risk_placeholder = st.empty()

    shelf_placeholder = st.empty()
    rec_placeholder = st.empty()

    # Open video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error(
            f"❌ Unable to open video source: `{video_source}`.\n\n"
            "Please check if the file path is correct, or switch to **Demo Scenario Simulator** in the sidebar."
        )
    else:
        max_frames = 100000 if frame_limit == "Continuous" else int(frame_limit)
        frame_idx = 0

        while cap.isOpened() and run_stream and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                if loop_video and isinstance(video_source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    break

            annotated_frame, telemetry = pipeline.process_frame(frame)
            frame_idx += 1

            if telemetry:
                # Render live video frame
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb,
                    use_container_width=True,
                    caption=f"Camera 01: Retail CCTV Pipeline | FPS: {telemetry['fps']} | Latency: {telemetry['inference_time_ms']}ms | Active Tracks: {telemetry['active_tracks_count']}"
                )

                # Update live banner & KPIs
                with alert_placeholder.container():
                    render_alert_banner(telemetry["store_state"], telemetry["risk"])
                with kpi_placeholder.container():
                    render_kpi_cards(telemetry["shopper"], telemetry["queue"], telemetry["shelf"], telemetry["risk"])

                with sub_shopper_placeholder.container():
                    st.subheader("Shopper Intelligence Analytics")
                    st.markdown(f"""
                    - **Total Verified Footfall:** `{telemetry['shopper']['footfall']}` shoppers
                    - **Current Active Occupancy:** `{telemetry['shopper']['occupancy']}` shoppers
                    - **Store Traffic Level:** `{telemetry['shopper']['traffic_level']}`
                    - **Average Dwell Time:** `{telemetry['shopper']['average_dwell_time']}` seconds
                    - **Active Tracks:** `{telemetry['active_tracks_count']}` anonymous profiles
                    """)

                with sub_queue_placeholder.container():
                    st.subheader("Queue Intelligence (Billing Counter)")
                    q = telemetry["queue"]
                    q_lvl = q["queue_level"]
                    st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="font-weight: 600;">Queue Severity:</span>
                            <span style="font-weight: 700; color: {'#EF4444' if q_lvl=='HIGH' else ('#F59E0B' if q_lvl=='MEDIUM' else '#10B981')};">{q_lvl}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="font-weight: 600;">Customers in Queue:</span>
                            <span style="font-weight: 700;">{q['queue_length']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: 600;">Est. Wait Time:</span>
                            <span style="font-weight: 700;">~{q['estimated_wait_minutes']} mins</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with risk_placeholder.container():
                    render_risk_breakdown(telemetry["risk"])

                with shelf_placeholder.container():
                    render_shelf_status_cards(telemetry["shelf"])

                with rec_placeholder.container():
                    render_recommendations_list(telemetry["recommendations"])

            time.sleep(0.015)

        cap.release()

# ----------------- COMMON HISTORICAL & ARCHITECTURE PANELS -----------------
st.markdown("---")
history = db.get_metrics_history(limit=30)
render_metrics_trends(history)

st.markdown("---")
render_edge_deployment_panel()

st.markdown("---")
render_privacy_panel(PrivacyGuard.get_privacy_status())

# ----------------- RECENT EVENTS TABLE (SQLite) -----------------
st.markdown("---")
st.subheader("Edge Event Ledger (SQLite Database)")
st.caption("Locally stored aggregate audit events. No personal identity or video frames stored.")
recent_events = db.get_recent_events(limit=10)
if recent_events:
    st.table(recent_events)
else:
    st.info("No high-severity operational events logged yet. System running smoothly.")
