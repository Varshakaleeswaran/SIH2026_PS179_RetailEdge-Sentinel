"""
RetailEdge Sentinel - Dashboard Chart Visualizations
Builds clean, lightweight visual analytics for Streamlit:
- Real-time occupancy & queue historical trends
- Operational risk factor breakdown
- Zone distribution
- Shelf stock fill gauges
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

def render_shelf_status_cards(shelf_metrics: Dict[str, Any]):
    """Render status indicators and fill percentage bars for monitored shelves."""
    st.subheader("Inventory & Shelf Intelligence")
    st.caption(f"Status: {shelf_metrics.get('estimation_label', 'Prototype Shelf Availability Estimation')}")
    
    shelves = shelf_metrics.get("shelves", {})
    cols = st.columns(len(shelves) if shelves else 1)

    for idx, (sid, info) in enumerate(shelves.items()):
        with cols[idx]:
            status = info.get("status", "AVAILABLE")
            fill = int(info.get("fill_percentage", 100))
            name = info.get("name", sid)

            if status == "AVAILABLE":
                color = "#10B981" # Green
                badge_bg = "#ECFDF5"
            elif status == "LOW":
                color = "#F59E0B" # Amber
                badge_bg = "#FFFBEB"
            else:
                color = "#EF4444" # Red
                badge_bg = "#FEF2F2"

            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;">
                <div style="font-weight: 600; color: #1E293B; font-size: 0.9rem; margin-bottom: 6px;">{name}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.78rem; font-weight: 700; background: {badge_bg}; color: {color}; padding: 2px 8px; border-radius: 12px;">
                        {status}
                    </span>
                    <span style="font-size: 0.85rem; font-weight: 700; color: {color};">{fill}% Fill</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(fill / 100.0)

def render_risk_breakdown(risk_metrics: Dict[str, Any]):
    """Render risk score components and root-cause reasons."""
    st.subheader("Operational Risk Intelligence")
    
    score = risk_metrics.get("risk_score", 0.0)
    level = risk_metrics.get("risk_level", "LOW")
    reasons = risk_metrics.get("reasons", [])
    components = risk_metrics.get("components", {})

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; text-align: center;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #64748B;">COMPOSITE RISK INDEX</div>
            <div style="font-size: 2.5rem; font-weight: 800; color: {risk_metrics.get('color_hex', '#333')};">{int(score)}<span style="font-size: 1.2rem; color: #94A3B8;">/100</span></div>
            <div style="font-weight: 700; color: {risk_metrics.get('color_hex', '#333')}; font-size: 1.05rem;">{level} PRIORITY</div>
        </div>
        """, unsafe_allow_html=True)

        # Component contributions
        df_comp = pd.DataFrame([
            {"Factor": "Queue Pressure (45%)", "Score": components.get("queue_score", 0)},
            {"Factor": "Shelf Depletion (30%)", "Score": components.get("inventory_score", 0)},
            {"Factor": "Shopper Traffic (25%)", "Score": components.get("traffic_score", 0)}
        ]).set_index("Factor")
        st.bar_chart(df_comp, height=160)

    with col2:
        st.markdown("**Top Contributing Factors:**")
        for r in reasons:
            st.markdown(f"- ⚠️ {r}")
        st.caption(f"Model Note: {risk_metrics.get('disclaimer', '')}")

def render_metrics_trends(metrics_history: List[Dict[str, Any]]):
    """Render time-series trend of occupancy and queue length from SQLite."""
    st.subheader("Store Flow & Queue Trends (Real-Time History)")
    if not metrics_history or len(metrics_history) < 2:
        st.caption("Collecting historical metric snapshots from edge SQLite database...")
        return

    df = pd.DataFrame(metrics_history)
    df["time"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
    df_chart = df[["time", "occupancy", "queue_length"]].set_index("time")
    df_chart.columns = ["Store Occupancy", "Checkout Queue"]
    st.line_chart(df_chart, height=220)
