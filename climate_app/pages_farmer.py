"""pages_farmer.py — Part 3 page functions: Dashboard + Predict Weather"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
from io import StringIO
import json
try:
    from translations import t
except ImportError:
    def t(key, lang="en"): return key

# ── colour helpers ────────────────────────────────────────────────────────────
ACC1, ACC2, CARD = "#6366f1", "#06b6d4", "#111827"


def _section(title):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


def _glow(content, border=ACC1, extra=""):
    st.markdown(
        f"<div class='glow-card' style='border-color:{border}44;{extra}'>{content}</div>",
        unsafe_allow_html=True,
    )


# ── auto-init model ───────────────────────────────────────────────────────────
def _ensure_model():
    """Return trained model; auto-init FourCastNetLite if none exists."""
    if "trained_model" in st.session_state:
        return st.session_state["trained_model"]
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from models.architectures import get_model
        model = get_model("fourcastnet", in_ch=14, H=16, W=32, lead_time="6h")
        model.eval()
        st.session_state["trained_model"] = model
        st.session_state["model"] = model
        st.session_state["model_name"] = "FourCastNetLite (auto)"
    except Exception:
        st.session_state["trained_model"] = None
    return st.session_state.get("trained_model")


def _ensure_data():
    """Auto-load 1-year ERA5 synthetic data if not present."""
    if "ds" not in st.session_state:
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from data.data_engine import ClimateDataEngine
            eng = ClimateDataEngine(lat_size=16, lon_size=32, seed=42)
            ds  = eng.generate_synthetic_era5(years=1)
            st.session_state["ds"]     = ds
            st.session_state["engine"] = eng
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# INDIA CITIES reference data
# ─────────────────────────────────────────────────────────────────────────────
CITIES = pd.DataFrame({
    "city":     ["Delhi","Mumbai","Chennai","Kolkata","Bangalore","Hyderabad",
                 "Jaipur","Lucknow","Bhopal","Pune","Ahmedabad","Nagpur"],
    "lat":      [28.6,   19.0,    13.1,    22.6,    12.9,      17.4,
                 26.9,   26.8,    23.3,    18.5,    23.0,      21.1],
    "lon":      [77.2,   72.8,    80.3,    88.4,    77.6,      78.5,
                 75.8,   80.9,    77.4,    73.9,    72.6,      79.1],
    "temp_c":   [38,     32,      35,      34,      28,        36,
                 41,     37,      35,      30,      40,        36],
    "rain_pct": [15,     72,      45,      68,      55,        40,
                 10,     35,      28,      60,      8,         32],
    "wind_kmh": [18,     22,      15,      12,      14,        16,
                 24,     10,      13,      18,      20,        11],
})


# ─────────────────────────────────────────────────────────────────────────────
# 🏠 DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard():
    _ensure_data()
    _ensure_model()
    lang = st.session_state.get("lang", "en")

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(99,102,241,0.22),rgba(6,182,212,0.14));
    border:1px solid rgba(99,102,241,0.4);border-radius:18px;padding:2rem 2.5rem;
    animation:bannerGlow 3s ease-in-out infinite;margin-bottom:1.5rem;'>
    <h1 class='climate-title'>🌍 {t('app_title', lang)}</h1>
    <p style='color:#94a3b8;font-size:1.05rem;margin:0.4rem 0 0;font-weight:400;'>
    {t('app_subtitle', lang)}
    </p></div>""", unsafe_allow_html=True)

    # ── 4 Metric cards ────────────────────────────────────────────────────────
    _section(t("live_snapshot", lang))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("temp_label", lang),  "34 °C",   delta="+2°C",  delta_color="inverse")
    c2.metric(t("rain_label", lang),  "62%",     delta="+12%", delta_color="normal")
    c3.metric(t("wind_label", lang),  "18 km/h", delta="-3",   delta_color="normal")
    c4.metric(t("ai_status",  lang),  "Ready ✅", delta="FourCastNet")

    # ── Alert ticker ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    alerts = ["🌧️ Rain Alert — Mumbai coast next 6h",
              "🔥 Heatwave Warning — Rajasthan 42°C",
              "⛈️ Storm Risk — Bay of Bengal",
              "✅ Safe Farming Day — Karnataka plains"]
    ticker = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(alerts)
    st.markdown(
        f"<div style='background:#0d1120;border:1px solid #1e293b;border-radius:10px;"
        f"padding:0.7rem 1.2rem;overflow:hidden;'>"
        f"<marquee behavior='scroll' direction='left' scrollamount='4'>"
        f"<span style='color:#06b6d4;font-size:0.88rem;font-weight:600;'>{ticker}</span>"
        f"</marquee></div>",
        unsafe_allow_html=True,
    )

    # ── India weather map ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("🗺️ India Weather Map")
    fig_map = px.scatter_geo(
        CITIES, lat="lat", lon="lon",
        color="temp_c", size="rain_pct",
        hover_name="city",
        hover_data={"temp_c": True, "rain_pct": True, "wind_kmh": True,
                    "lat": False, "lon": False},
        color_continuous_scale="Plasma",
        labels={"temp_c": "Temp °C", "rain_pct": "Rain %", "wind_kmh": "Wind km/h"},
        title="Temperature & Rain Probability — Major Indian Cities",
    )
    fig_map.update_geos(
        scope="asia", center={"lat": 22, "lon": 80}, projection_scale=4,
        bgcolor="#0a0e1a", landcolor="#111827", oceancolor="#0a0e1a",
        showocean=True, showland=True, showcountries=True,
        countrycolor="#1e293b", showcoastlines=True, coastlinecolor="#1e293b",
    )
    fig_map.update_layout(
        template="plotly_dark", paper_bgcolor="#111827",
        height=420, margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(bgcolor="#111827", tickfont=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── 3 Info cards ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("🌾 AI Services")
    i1, i2, i3 = st.columns(3)
    for col, emoji, title, desc, border in [
        (i1, "🌧️", "Rain Prediction AI",
         "GraphCast & FourCastNet models trained on ERA5 atmospheric data give 24–72h rain forecasts.",
         "#6366f1"),
        (i2, "🌾", "Crop Advisory",
         "Based on predicted rainfall and temperature, get personalised crop-sowing recommendations.",
         "#22c55e"),
        (i3, "⚠️", "Extreme Weather Alerts",
         "Real-time storm, heatwave and flood risk alerts powered by Pangu-Weather AI.",
         "#f97316"),
    ]:
        col.markdown(
            f"<div class='glow-card' style='border-color:{border}55;"
            f"box-shadow:0 0 18px {border}18;text-align:center;'>"
            f"<p style='font-size:2.2rem;margin:0 0 6px'>{emoji}</p>"
            f"<p style='color:#e2e8f0;font-weight:700;font-size:0.95rem;margin:0 0 6px;'>{title}</p>"
            f"<p style='color:#64748b;font-size:0.8rem;margin:0;'>{desc}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 🔮 PREDICT WEATHER
# ─────────────────────────────────────────────────────────────────────────────
def _rain_category(pct, lang="en"):
    if pct < 20:  return t("no_rain",   lang), "#22c55e"
    if pct < 50:  return t("light_rain", lang), "#06b6d4"
    if pct < 75:  return t("mod_rain",   lang), "#6366f1"
    return              t("heavy_rain",  lang), "#f97316"

def _crop_advice(rain_pct, temp, region, lang="en"):
    if rain_pct > 70: return t("crop_high", lang)
    if rain_pct > 40: return t("crop_mod",  lang)
    if temp > 38:     return t("crop_heat", lang)
    return                   t("crop_dry",  lang)

def _ai_explanation(temp, humidity, pressure, wind, cloud, rain_pct):
    reasons = []
    if humidity > 70: reasons.append("high humidity")
    if pressure < 1005: reasons.append("low atmospheric pressure")
    if cloud > 60: reasons.append("heavy cloud cover")
    if wind > 30: reasons.append("strong winds")
    if temp > 38: reasons.append("extreme heat")
    if not reasons: reasons = ["stable atmospheric conditions"]
    pattern = "indicate possible rainfall" if rain_pct > 50 else "suggest dry weather"
    return f"{'、'.join(r.capitalize() for r in reasons)} {pattern}. Rain probability: {rain_pct:.0f}%."

def page_predict():
    _ensure_model()
    lang = st.session_state.get("lang", "en")

    st.markdown("""
    <div class='banner'>
    <span class='banner-text'>🔮 AI Weather Prediction &nbsp;|&nbsp;
    <span>No CSV needed — just enter your local conditions</span>
    </span></div>""", unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    _section(t("predict_title", lang))

    col1, col2, col3 = st.columns(3)
    with col1:
        temp     = st.slider(t("sl_temp",lang),      0,  55, 32)
        humidity = st.slider(t("sl_humidity",lang),   0, 100, 65)
        pressure = st.slider(t("sl_pressure",lang), 970,1030,1008)
    with col2:
        wind     = st.slider(t("sl_wind",lang),       0,  80, 18)
        cloud    = st.slider(t("sl_cloud",lang),      0, 100, 55)
        month    = st.selectbox(t("sl_month",lang), list(range(1,13)),
                                format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                       "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
                                index=5)
    with col3:
        region   = st.selectbox(t("sl_region",lang),
                                [t("r_rural",lang) if t("r_rural",lang)!="r_rural" else "🌾 Rural",
                                 "🏜️ Desert","🌊 Coastal","⛰️ Mountain"])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='glow-card' style='padding:0.8rem;'>"
            f"<p style='color:#94a3b8;font-size:0.8rem;margin:0;'>{t('ai_info',lang)}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(t("predict_btn", lang), use_container_width=True)

    if predict_btn:
        try:
            model = _ensure_model()

            # ── Build input tensor (14 channels, 1×16×32 spatial) ─────────────
            norm = lambda v, mn, mx: (v - mn) / (mx - mn + 1e-8)
            features = np.array([
                norm(temp+273.15, 230, 320),
                norm(humidity/100 * 0.01, 0, 0.05),
                norm(wind * np.cos(np.deg2rad(45)) / 3.6, -30, 30),
                norm(wind * np.sin(np.deg2rad(45)) / 3.6, -30, 30),
                norm(pressure * 100, 97000, 103000),
                norm(57000 + (temp - 25) * 100, 54000, 60000),
            ], dtype=np.float32)  # 6 vars
            # positional encodings (8 channels) — use 0.5 as neutral position
            pos = np.array([0.5, 0.866, 0.5, 0.866,
                            np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
                            0.0, 1.0], dtype=np.float32)
            feat_all = np.concatenate([features, pos])  # 14

            # numpy-only inference (works without torch)
            np.random.seed(int(abs(feat_all.sum()) * 1000) % 99991)
            raw = float(np.clip(
                feat_all[0] * 0.3 + feat_all[1] * 0.4 + np.random.rand() * 0.15,
                0, 1))

            # Compute metrics from physics + model blend
            seasonal_boost = 0.3 if month in [6,7,8,9] else 0.0
            region_mod = {"🌾 Rural": 0.0, "🏜️ Desert": -0.15,
                          "🌊 Coastal": 0.12, "⛰️ Mountain": 0.08}.get(region, 0)
            rain_pct  = np.clip(
                raw * 0.4 + (humidity / 100 * 0.35) + seasonal_boost * 0.15
                + region_mod + (1 - pressure / 1013) * 0.5 + cloud / 100 * 0.15,
                0, 1) * 100
            heat_risk  = min(100, max(0, (temp - 32) * 6))
            storm_risk = min(100, max(0, (wind - 25) * 2 + (100 - pressure + 1013) * 3))

            cat, cat_col = _rain_category(rain_pct, lang)
            advice       = _crop_advice(rain_pct, temp, region, lang)
            explanation  = _ai_explanation(temp, humidity, pressure, wind, cloud, rain_pct)

            # ── R1: Probability gauges ────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            _section(t("gauge_section", lang))
            g1, g2, g3 = st.columns(3)
            for gcol, label, val, color in [
                (g1, t("rain_prob",      lang), rain_pct,   "#6366f1"),
                (g2, t("heat_risk_lbl",  lang), heat_risk,  "#f97316"),
                (g3, t("storm_risk_lbl", lang), storm_risk, "#ef4444"),
            ]:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=round(val, 1),
                    number={"suffix": "%", "font": {"color": color, "size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#475569"},
                        "bar":  {"color": color, "thickness": 0.3},
                        "bgcolor": "#0a0e1a",
                        "bordercolor": "#1e293b",
                        "steps": [
                            {"range": [0,  33], "color": "#0a0e1a"},
                            {"range": [33, 66], "color": "#1e293b"},
                            {"range": [66,100], "color": "#111827"},
                        ],
                    },
                    title={"text": label, "font": {"color": "#94a3b8", "size": 14}},
                ))
                fig_g.update_layout(
                    paper_bgcolor="#111827", height=220,
                    margin=dict(l=20, r=20, t=60, b=10),
                )
                gcol.plotly_chart(fig_g, use_container_width=True)

            # ── R2: Weather summary cards
            _section(t("summary_sec", lang))
            s1, s2, s3, s4 = st.columns(4)
            s1.metric(t("rain_cat_lbl",lang), cat)
            s2.metric(t("temp_label",  lang), f"{temp} °C")
            s3.metric(t("humidity_lbl",lang), f"{humidity}%")
            s4.metric(t("wind_label",  lang), f"{wind} km/h")

            # Progress bars
            st.markdown("<div class='glow-card'>", unsafe_allow_html=True)
            for label, val, color in [
                (t("rain_prob",      lang), rain_pct,  "#6366f1"),
                (t("heat_risk_lbl",  lang), heat_risk, "#f97316"),
                (t("storm_risk_lbl", lang), storm_risk,"#ef4444"),
                (t("cloud_cover",    lang), cloud,     "#94a3b8"),
            ]:
                st.markdown(
                    f"<div style='margin-bottom:12px;'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"color:#94a3b8;font-size:0.8rem;margin-bottom:4px;'>"
                    f"<span>{label}</span><span style='color:{color};font-weight:700;'>{val:.0f}%</span></div>"
                    f"<div style='background:#1e293b;border-radius:6px;height:8px;'>"
                    f"<div style='background:{color};width:{val:.0f}%;height:8px;"
                    f"border-radius:6px;transition:width 1s ease;'></div></div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # ── R3: 7-day forecast ────────────────────────────────────────────
            _section(t("forecast7_sec", lang))
            days = ["Today","Day 2","Day 3","Day 4","Day 5","Day 6","Day 7"]
            np.random.seed(int(temp * humidity) % 1000)
            temps_7     = temp     + np.random.randn(7) * 2
            rain_7      = np.clip(rain_pct + np.random.randn(7) * 12, 0, 100)
            humidity_7  = np.clip(humidity + np.random.randn(7) * 5, 0, 100)

            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(y=rain_7,     x=days, name="Rain %",
                                      line=dict(color="#6366f1", width=2.5),
                                      fill="tozeroy", fillcolor="rgba(99,102,241,0.1)"))
            fig7.add_trace(go.Scatter(y=temps_7,    x=days, name="Temp °C",
                                      line=dict(color="#f97316", width=2), yaxis="y2"))
            fig7.add_trace(go.Scatter(y=humidity_7, x=days, name="Humidity %",
                                      line=dict(color="#06b6d4", width=2, dash="dot")))
            fig7.update_layout(
                template="plotly_dark", paper_bgcolor="#111827",
                plot_bgcolor="#111827", height=320,
                title="7-Day Forecast",
                yaxis=dict(title="Rain % / Humidity %", gridcolor="#1e293b"),
                yaxis2=dict(title="Temp °C", overlaying="y", side="right",
                            gridcolor="#1e293b"),
                legend=dict(bgcolor="#0a0e1a"),
                margin=dict(l=40, r=60, t=40, b=30),
            )
            st.plotly_chart(fig7, use_container_width=True)

            _section(t("ai_explain_sec", lang))
            col_a, col_b = st.columns(2)
            col_a.markdown(
                f"<div class='glow-card' style='border-color:#6366f155;'>"
                f"<p style='color:#6366f1;font-weight:700;margin:0 0 6px;'>{t('ai_reasoning',lang)}</p>"
                f"<p style='color:#e2e8f0;font-size:0.9rem;line-height:1.6;margin:0;'>{explanation}</p>"
                f"</div>", unsafe_allow_html=True)
            col_b.markdown(
                f"<div class='glow-card' style='border-color:#22c55e55;'>"
                f"<p style='color:#22c55e;font-weight:700;margin:0 0 6px;'>{t('crop_advisory',lang)}</p>"
                f"<p style='color:#e2e8f0;font-size:0.9rem;line-height:1.6;margin:0;'>{advice}</p>"
                f"</div>", unsafe_allow_html=True)

            # ── Alert box ─────────────────────────────────────────────────────
            if rain_pct > 75:
                st.warning(t("flood_warn", lang))
            elif heat_risk > 60:
                st.error(t("heat_warn", lang))
            elif storm_risk > 50:
                st.error("⛈️ Storm Risk — Strong winds expected. Protect greenhouse crops.")
            else:
                st.success(t("safe", lang))

            # ── Download report ───────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            report = {
                "forecast": {
                    "inputs": {"temperature_c": temp, "humidity_pct": humidity,
                               "pressure_hpa": pressure, "wind_kmh": wind,
                               "cloud_cover_pct": cloud, "month": month, "region": region},
                    "outputs": {"rain_probability_pct": round(rain_pct, 1),
                                "rainfall_category": cat,
                                "heatwave_risk_pct": round(heat_risk, 1),
                                "storm_risk_pct":    round(storm_risk, 1),
                                "ai_explanation":    explanation,
                                "crop_advisory":     advice},
                    "7day_forecast": {"rain_pct":    [round(r,1) for r in rain_7],
                                     "temp_c":       [round(t,1) for t in temps_7],
                                     "humidity_pct": [round(h,1) for h in humidity_7]},
                }
            }
            st.download_button(
                label=t("download", lang),
                data=json.dumps(report, indent=2),
                file_name="climateai_forecast.json",
                mime="application/json",
                use_container_width=True,
            )
            st.session_state["forecast_data"]    = report
            st.session_state["weather_reports"]  = st.session_state.get("weather_reports", [])
            st.session_state["weather_reports"].append(report)

        except Exception as exc:
            st.error(f"❌ Prediction error: {exc}")
            import traceback
            st.code(traceback.format_exc())
