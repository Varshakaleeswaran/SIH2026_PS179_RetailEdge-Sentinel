"""
RetailEdge Sentinel - Streamlit Dashboard UI Components
Provides styled UI widgets, status badges, KPI cards, alerts, recommendations, and hardware panels.
"""

import streamlit as st
from typing import Dict, Any, List

def inject_custom_css():
    """Inject modern, high-contrast, professional retail operations theme CSS."""
    st.markdown("""
    <style>
        /* Main background & typography */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0px;
            letter-spacing: -0.5px;
        }
        .main-subtitle {
            font-size: 1.05rem;
            color: #64748B;
            margin-top: 2px;
            margin-bottom: 18px;
            font-weight: 500;
        }
        /* Top Status Pills Bar */
        .status-bar-container {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            padding: 10px 16px;
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            margin-bottom: 20px;
            align-items: center;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .status-online { background-color: #DCFCE7; color: #166534; }
        .status-env { background-color: #E0E7FF; color: #3730A3; }
        .status-target { background-color: #FEF3C7; color: #92400E; }
        .status-privacy { background-color: #F1F5F9; color: #334155; }

        /* KPI Cards */
        .kpi-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 14px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 12px;
        }
        .kpi-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 2.0rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.1;
        }
        .kpi-subtext {
            font-size: 0.8rem;
            color: #94A3B8;
            margin-top: 4px;
        }

        /* Recommendation cards */
        .rec-card {
            background-color: #FFFFFF;
            border-left: 5px solid #3B82F6;
            border-top: 1px solid #E2E8F0;
            border-right: 1px solid #E2E8F0;
            border-bottom: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .rec-priority-high { border-left-color: #EF4444; }
        .rec-priority-medium { border-left-color: #F59E0B; }
        .rec-priority-low { border-left-color: #10B981; }

        .rec-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .rec-badge {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .rec-action {
            font-weight: 600;
            color: #1E293B;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }
        .rec-reason {
            font-size: 0.82rem;
            color: #64748B;
        }

        /* Banner Alerts */
        .alert-banner {
            padding: 10px 16px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.92rem;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .alert-green { background-color: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
        .alert-amber { background-color: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
        .alert-orange { background-color: #FFF7ED; color: #9A3412; border: 1px solid #FED7AA; }
        .alert-red { background-color: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render top brand header and deployment telemetry badges."""
    st.markdown('<div class="main-header">RetailEdge Sentinel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Privacy-Preserving Edge Intelligence for Retail Operations | '
        '<b>CAMERA → DETECT → ANALYSE → ALERT → ACTION</b></div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="status-bar-container">
        <div class="status-pill status-online">● SYSTEM ONLINE</div>
        <div class="status-pill status-env">DEV ENV: Windows Laptop (Local Simulation)</div>
        <div class="status-pill status-target">TARGET: Qualcomm Dragonwing Edge AI</div>
        <div class="status-pill status-privacy">PRIVACY: Anonymous Edge Processing (No Facial Rec)</div>
    </div>
    """, unsafe_allow_html=True)

def render_alert_banner(store_state: Dict[str, Any], risk_metrics: Dict[str, Any]):
    """Render high-visibility operational banner based on store state and risk level."""
    state = store_state.get("overall_state", "NORMAL")
    risk_lvl = risk_metrics.get("risk_level", "LOW")
    score = risk_metrics.get("risk_score", 0.0)

    if risk_lvl == "CRITICAL" or state == "CRITICAL_RETAIL_STATE":
        css_class = "alert-red"
        icon = "🚨"
        title = f"CRITICAL RETAIL ALERT (Risk Score: {score}/100)"
        desc = store_state.get("summary", "Immediate floor management intervention required.")
    elif risk_lvl == "HIGH" or state in ["OPERATIONAL_STRESS", "QUEUE_BUILDUP"]:
        css_class = "alert-orange"
        icon = "⚠️"
        title = f"HIGH OPERATIONAL STRESS (Risk Score: {score}/100)"
        desc = store_state.get("summary", "Checkout bottleneck or stock depletion detected.")
    elif risk_lvl == "MEDIUM" or state == "LOW_STOCK":
        css_class = "alert-amber"
        icon = "🔔"
        title = f"ATTENTION REQUIRED (Risk Score: {score}/100)"
        desc = store_state.get("summary", "Moderate queue or stock levels approaching minimum threshold.")
    else:
        css_class = "alert-green"
        icon = "✓"
        title = f"OPERATIONS NOMINAL (Risk Score: {score}/100)"
        desc = "Customer flow, checkout throughput, and shelf inventory within optimal operational limits."

    st.markdown(f"""
    <div class="alert-banner {css_class}">
        <span>{icon}</span>
        <div>
            <b>{title}</b> — <span style="font-weight: 400;">{desc}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_cards(shopper: Dict[str, Any], queue: Dict[str, Any], shelf: Dict[str, Any], risk: Dict[str, Any]):
    """Render top 5 KPI cards in responsive columns."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Cumulative Footfall</div>
            <div class="kpi-value">{shopper.get('footfall', 0)}</div>
            <div class="kpi-subtext">Total verified entries</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        occ = shopper.get('occupancy', 0)
        traf = shopper.get('traffic_level', 'LOW')
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Store Occupancy</div>
            <div class="kpi-value">{occ}</div>
            <div class="kpi-subtext">Traffic Level: <b>{traf}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        q_len = queue.get('queue_length', 0)
        q_trend = queue.get('growth_trend', 'STABLE')
        trend_arrow = "▲" if q_trend == "INCREASING" else ("▼" if q_trend == "DECREASING" else "■")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Checkout Queue</div>
            <div class="kpi-value">{q_len}</div>
            <div class="kpi-subtext">Trend: <b>{trend_arrow} {q_trend}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        low_s = shelf.get('low_shelves', 0)
        emp_s = shelf.get('empty_shelves', 0)
        status_txt = f"{low_s} Low, {emp_s} Empty" if (low_s + emp_s) > 0 else "All Stocked"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Shelf Attention</div>
            <div class="kpi-value">{low_s + emp_s}</div>
            <div class="kpi-subtext">{status_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        score = risk.get('risk_score', 0.0)
        lvl = risk.get('risk_level', 'LOW')
        color = risk.get('color_hex', '#2ECC71')
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {color};">
            <div class="kpi-title">Operational Risk</div>
            <div class="kpi-value" style="color: {color};">{int(score)}<span style="font-size: 1.1rem; color: #94A3B8;">/100</span></div>
            <div class="kpi-subtext">Priority: <b>{lvl}</b></div>
        </div>
        """, unsafe_allow_html=True)

def render_recommendations_list(recommendations: List[Dict[str, Any]]):
    """Render actionable recommendation cards."""
    st.subheader("Action Recommendations for Floor Staff")
    if not recommendations:
        st.info("No active recommendations. Store operations nominal.")
        return

    for rec in recommendations:
        priority = rec.get("priority", "LOW")
        p_class = f"rec-priority-{priority.lower()}"
        b_color = rec.get("badge_color", "#3B82F6")
        
        st.markdown(f"""
        <div class="rec-card {p_class}">
            <div class="rec-header">
                <span class="rec-badge" style="background-color: {b_color}22; color: {b_color};">
                    {priority} PRIORITY
                </span>
                <span style="font-size: 0.78rem; color: #64748B;">{rec.get('issue', '')}</span>
            </div>
            <div class="rec-action">{rec.get('action', '')}</div>
            <div class="rec-reason"><b>Reason:</b> {rec.get('reason', '')}</div>
        </div>
        """, unsafe_allow_html=True)

def render_edge_deployment_panel():
    """Render clear, honest deployment comparison between local laptop simulation and Qualcomm Dragonwing."""
    st.subheader("Edge Hardware & Architecture Roadmap")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 14px 18px;">
            <div style="font-weight: 700; color: #1E293B; font-size: 1.05rem; margin-bottom: 8px;">
                Current Execution (MVP Development)
            </div>
            <ul style="font-size: 0.88rem; color: #334155; margin-bottom: 0; padding-left: 20px;">
                <li><b>Environment:</b> Windows 11 PC / Laptop</li>
                <li><b>Inference Engine:</b> <code>LocalInferenceEngine</code> (CPU / Direct PyTorch)</li>
                <li><b>Model Weights:</b> Ultralytics YOLOv8 nano (FP32)</li>
                <li><b>Status:</b> Fully working, locally validated simulation</li>
                <li><b>Storage:</b> Local SQLite event & metric store</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #EFF6FF; border: 1px solid #93C5FD; border-radius: 8px; padding: 14px 18px;">
            <div style="font-weight: 700; color: #1E3A8A; font-size: 1.05rem; margin-bottom: 8px;">
                Target Deployment Platform (Production Roadmap)
            </div>
            <ul style="font-size: 0.88rem; color: #1E40AF; margin-bottom: 0; padding-left: 20px;">
                <li><b>Hardware:</b> Qualcomm Dragonwing-compatible Edge AI Platform</li>
                <li><b>Runtime:</b> Qualcomm AI Runtime (QAIRT / QNN / SNPE)</li>
                <li><b>Optimization:</b> Qualcomm AI Hub & AIMET (INT8 Quantization)</li>
                <li><b>Container:</b> Qualcomm DLC (Deep Learning Container)</li>
                <li><b>Status:</b> Interface stub defined (<code>QualcommInferenceEngine</code>)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_privacy_panel(privacy_status: Dict[str, Any]):
    """Render privacy compliance guarantees."""
    st.subheader("Privacy Protection Architecture")
    st.markdown("""
    RetailEdge Sentinel adheres strictly to <b>Privacy-by-Design</b> principles. Video streams are analyzed purely in volatile edge memory.
    """)
    for g in privacy_status.get("guarantees", []):
        st.markdown(f"**✓ {g['item']}**: {g['details']}")
