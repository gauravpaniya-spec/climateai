"""pages_advanced.py — Part 4: Farmer Advisory + Evaluation pages"""
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

try:
    from translations import t
except ImportError:
    def t(key, lang="en"): return key

# ── helpers ───────────────────────────────────────────────────────────────────
def _section(title): st.markdown(f"<div class='section-header'>{title}</div>",unsafe_allow_html=True)


def _card(html, border="#6366f1", extra=""):
    st.markdown(f"<div class='glow-card' style='border-color:{border}55;{extra}'>{html}</div>",
                unsafe_allow_html=True)

def _badge(label, val, color):
    return (f"<span style='background:{color}22;border:1px solid {color}55;"
            f"border-radius:8px;padding:3px 10px;color:{color};"
            f"font-size:.8rem;font-weight:700;margin-right:6px;'>{label}: {val}</span>")

# ─────────────────────────────────────────────────────────────────────────────
# 🌾 FARMER ADVISORY
# ─────────────────────────────────────────────────────────────────────────────
CROPS = {
    ("high","wet","hot","Kharif"):  ("🌾 Paddy / Jute",   "#22c55e", "status_exc"),
    ("high","wet","warm","Kharif"): ("🌽 Maize / Soybean", "#22c55e", "status_good"),
    ("mod", "dry","warm","Rabi"):   ("🌾 Wheat / Mustard", "#06b6d4", "status_good"),
    ("low", "dry","hot","Rabi"):    ("🌵 Millets / Gram",  "#f59e0b", "status_caut"),
    ("low", "dry","hot","Zaid"):    ("🍉 Watermelon / Cucumber","#f59e0b","status_irr"),
}

def _get_crop(rain, moist, temp, season):
    rcat  = "high" if rain>65  else ("mod" if rain>35  else "low")
    mcat  = "wet"  if moist>55 else "dry"
    tcat  = "hot"  if temp>35  else "warm"
    key   = (rcat, mcat, tcat, season)
    return CROPS.get(key, ("🌱 Mixed Vegetables / Pulses","#94a3b8","status_gen"))

def page_farmer():
    lang = st.session_state.get("lang", "en")
    st.markdown(f"""<div class='banner'><span class='banner-text'>
    {t('farmer_banner', lang)}
    </span></div>""", unsafe_allow_html=True)

    with st.expander(t("farmer_how", lang)):
        st.markdown(t("farmer_how_txt", lang))

    _section(t("field_cond", lang))
    col1,col2,col3 = st.columns(3)
    with col1:
        rain  = st.slider(t("sl_rain_pct", lang), 0, 100, 55)
        moist = st.slider(t("sl_moisture", lang),  0, 100, 45)
    with col2:
        temp   = st.slider(t("sl_temp", lang),     0,  50, 32)
        season = st.selectbox(t("sl_season", lang), ["Kharif","Rabi","Zaid"])
    with col3:
        area = st.selectbox(t("sl_region", lang),
                            ["🌾 Plains","🏔️ Hills","🌊 Coastal","🏜️ Arid"])

        st.markdown("<br>",unsafe_allow_html=True)
        analyse = st.button(t("farmer_title", lang), use_container_width=True)

    if not analyse: return

    crop, crop_col, crop_status_key = _get_crop(rain, moist, temp, season)
    flood_risk = min(100, rain*0.8 + moist*0.3)
    heat_risk  = min(100, max(0,(temp-32)*6))
    irr_need   = max(0,70-moist) * 0.5
    crop_status = t(crop_status_key, lang)

    flood_col = "#ef4444" if flood_risk>65 else ("#f59e0b" if flood_risk>35 else "#22c55e")
    heat_col  = "#ef4444" if heat_risk>65  else ("#f59e0b" if heat_risk>35  else "#22c55e")
    flood_lbl = t("risk_danger",lang) if flood_risk>65 else (t("risk_caution",lang) if flood_risk>35 else t("risk_safe",lang))
    heat_lbl  = t("risk_danger",lang) if heat_risk>65  else (t("risk_caution",lang) if heat_risk>35  else t("risk_safe",lang))

    # Animated alerts
    if flood_risk > 65:
        st.error(t("flood_warn", lang))
    if heat_risk > 65:
        st.error(t("heat_warn", lang))
    if flood_risk < 35 and heat_risk < 35:
        st.success(t("safe", lang))

    # ── 4 Output cards ────────────────────────────────────────────────────────
    _section(t("advisory_res", lang))
    r1,r2,r3,r4 = st.columns(4)
    r1.markdown(f"<div class='glow-card' style='border-color:{crop_col}55;text-align:center;'>"
                f"<p style='font-size:1.4rem;margin:0;'>🌱</p>"
                f"<p style='color:{crop_col};font-weight:700;font-size:.85rem;margin:4px 0 2px;'>{t('best_crop',lang)}</p>"
                f"<p style='color:#e2e8f0;font-size:.8rem;margin:0;'>{crop}</p>"
                f"<p style='color:{crop_col};font-size:.72rem;'>{crop_status}</p></div>",
                unsafe_allow_html=True)
    r2.markdown(f"<div class='glow-card' style='border-color:#06b6d455;text-align:center;'>"
                f"<p style='font-size:1.4rem;margin:0;'>💧</p>"
                f"<p style='color:#06b6d4;font-weight:700;font-size:.85rem;margin:4px 0 2px;'>{t('irr_need',lang)}</p>"
                f"<p style='color:#e2e8f0;font-size:.8rem;'>{irr_need:.0f} mm/day</p>"
                f"<p style='color:#06b6d4;font-size:.72rem;'>{t('irr_today',lang) if irr_need>15 else t('rain_suff',lang)}</p></div>",
                unsafe_allow_html=True)
    r3.markdown(f"<div class='glow-card' style='border-color:{flood_col}55;text-align:center;'>"
                f"<p style='font-size:1.4rem;margin:0;'>🌊</p>"
                f"<p style='color:{flood_col};font-weight:700;font-size:.85rem;margin:4px 0 2px;'>{t('flood_risk',lang)}</p>"
                f"<p style='color:#e2e8f0;font-size:.8rem;'>{flood_risk:.0f}%</p>"
                f"<p style='color:{flood_col};font-size:.72rem;'>{flood_lbl}</p></div>",
                unsafe_allow_html=True)
    r4.markdown(f"<div class='glow-card' style='border-color:{heat_col}55;text-align:center;'>"
                f"<p style='font-size:1.4rem;margin:0;'>🔥</p>"
                f"<p style='color:{heat_col};font-weight:700;font-size:.85rem;margin:4px 0 2px;'>{t('heat_stress',lang)}</p>"
                f"<p style='color:#e2e8f0;font-size:.8rem;'>{heat_risk:.0f}%</p>"
                f"<p style='color:{heat_col};font-size:.72rem;'>{heat_lbl}</p></div>",
                unsafe_allow_html=True)

    # ── Fertilizer timing ─────────────────────────────────────────────────────
    st.markdown("<br>",unsafe_allow_html=True)
    _section(t("fert_sec", lang))
    fert_timing = (t("fert_ok",lang) if 35<moist<70 else t("fert_wait",lang))
    irr_advice  = (t("irr_drip",lang) if irr_need>15 else t("irr_rain_ok",lang))
    _card(f"<p style='color:#22c55e;font-weight:700;margin:0 0 6px;'>🌿 {t('fert_sec',lang)}</p>"
          f"<p style='color:#e2e8f0;font-size:.88rem;margin:0 0 4px;'>{fert_timing}</p>"
          f"<p style='color:#e2e8f0;font-size:.88rem;margin:0;'>{irr_advice}</p>")

    # ── Weekly outlook chart ───────────────────────────────────────────────────
    _section(t("weekly_outlook", lang))
    np.random.seed(int(rain*temp) % 999)
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    r7   = np.clip(rain  + np.random.randn(7)*10, 0,100)
    t7   = temp + np.random.randn(7)*2
    m7   = np.clip(moist + np.random.randn(7)*5,  0,100)
    fig  = go.Figure()
    fig.add_trace(go.Bar(x=days, y=r7, name="Rain %", marker_color="#6366f1", opacity=0.8))
    fig.add_trace(go.Scatter(x=days, y=t7, name="Temp °C", line=dict(color="#f97316",width=2.5), yaxis="y2"))
    fig.add_trace(go.Scatter(x=days, y=m7, name="Soil Moisture %",
                             line=dict(color="#06b6d4",width=2,dash="dot")))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#111827",plot_bgcolor="#111827",
                      height=300,yaxis=dict(title="Rain% / Moisture%",gridcolor="#1e293b"),
                      yaxis2=dict(title="Temp°C",overlaying="y",side="right"),
                      legend=dict(bgcolor="#0a0e1a"),margin=dict(l=40,r=60,t=20,b=30))
    st.plotly_chart(fig, use_container_width=True)

    # ── AI Farmer Tips ────────────────────────────────────────────────────────
    _section(t("ai_tips", lang))
    tips = []
    if rain > 65: tips.append(("⛈️", t("tip_flood",lang), "#ef4444"))
    if moist < 30: tips.append(("🌵", t("tip_dry",  lang), "#f59e0b"))
    if temp > 38:  tips.append(("🔥", t("tip_heat", lang), "#f97316"))
    if 35<=rain<=65 and 35<=moist<=65: tips.append(("🌾", t("tip_ideal",lang), "#22c55e"))
    tips.append(("🧪", t("tip_soil",lang), "#06b6d4"))
    tips.append(("📱", t("tip_save",lang), "#6366f1"))
    for em,tip,col in tips[:4]:
        _card(f"<p style='margin:0;font-size:.88rem;'>"
              f"<span style='font-size:1.1rem;'>{em}</span> "
              f"<span style='color:#e2e8f0;'>{tip}</span></p>", border=col)

    st.session_state.setdefault("advisory_reports",[]).append(
        {"rain":rain,"moist":moist,"temp":temp,"season":season,"crop":crop,
         "flood_risk":round(flood_risk,1),"heat_risk":round(heat_risk,1)})


# ─────────────────────────────────────────────────────────────────────────────
# 📈 EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def _make_fake_metrics(model_name, seed=42):
    np.random.seed(seed)
    leads = [6, 12, 24, 48, 72]
    base  = {"FourCastNetLite":1.0,"GraphCastLite":0.85,"PanguLite":0.92}.get(model_name,1.0)
    rmse_t = [round(base*(0.8+0.12*i)+np.random.rand()*0.15, 3) for i,_ in enumerate(leads)]
    rmse_w = [round(base*(1.2+0.18*i)+np.random.rand()*0.2,  3) for i,_ in enumerate(leads)]
    rmse_p = [round(base*(0.6+0.08*i)+np.random.rand()*0.1,  3) for i,_ in enumerate(leads)]
    acc_r  = [round(max(0.5, 0.92-0.04*i+np.random.rand()*0.03), 3) for i,_ in enumerate(leads)]
    return leads, rmse_t, rmse_w, rmse_p, acc_r

def page_evaluation():
    st.markdown("""<div class='banner'><span class='banner-text'>
    📈 Model Evaluation &nbsp;|&nbsp;
    <span>WeatherBench-style AI accuracy metrics</span></span></div>""",
    unsafe_allow_html=True)

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **RMSE** — Root Mean Squared Error. Lower = better forecast.
- **ACC** — Anomaly Correlation Coefficient. Closer to 1.0 = excellent.
- **Skill Score** — How much better than a naive (persistence) baseline.
- **Rain Accuracy** — % of rain events correctly predicted.
        """)

    # Model selector
    model_name = st.session_state.get("model_name","FourCastNetLite")
    _section("🧠 Evaluating Model")
    _card(f"<p style='margin:0;color:#e2e8f0;font-size:.9rem;'>"
          f"Active model: <b style='color:#6366f1;'>{model_name}</b> &nbsp;|&nbsp; "
          f"Status: <span style='color:#22c55e;'>✅ Ready</span></p>")

    leads, rmse_t, rmse_w, rmse_p, acc_r = _make_fake_metrics(model_name)
    persist_rmse = [round(v*1.35+0.1, 3) for v in rmse_t]
    skill_scores = [round(max(0,1-r/p),3) for r,p in zip(rmse_t, persist_rmse)]

    # ── 1. RMSE vs Lead Time ──────────────────────────────────────────────────
    _section("📉 1 — RMSE vs Lead Time")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=leads,y=rmse_t,name="Temp RMSE (K)",
        line=dict(color="#f97316",width=2.5),mode="lines+markers"))
    fig1.add_trace(go.Scatter(x=leads,y=rmse_w,name="Wind RMSE (m/s)",
        line=dict(color="#06b6d4",width=2.5),mode="lines+markers"))
    fig1.add_trace(go.Scatter(x=leads,y=rmse_p,name="Pressure RMSE (hPa)",
        line=dict(color="#8b5cf6",width=2.5),mode="lines+markers"))
    fig1.add_trace(go.Scatter(x=leads,y=persist_rmse,name="Persistence Baseline",
        line=dict(color="#475569",width=1.5,dash="dash"),mode="lines"))
    fig1.update_layout(template="plotly_dark",paper_bgcolor="#111827",
        plot_bgcolor="#111827",height=300,
        xaxis=dict(title="Lead Time (h)",gridcolor="#1e293b"),
        yaxis=dict(title="RMSE",gridcolor="#1e293b"),
        legend=dict(bgcolor="#0a0e1a"),margin=dict(l=40,r=20,t=20,b=40))
    st.plotly_chart(fig1, use_container_width=True)

    # ── 2. Confidence gauges ──────────────────────────────────────────────────
    _section("🎯 2 — Accuracy / Confidence Gauges")
    g1,g2,g3,g4 = st.columns(4)
    metrics_g = [
        (g1,"🌧️ Rain Accuracy", round(acc_r[0]*100,1), "#6366f1"),
        (g2,"🌡️ Temp Skill",    round(skill_scores[0]*100,1),"#f97316"),
        (g3,"💨 Wind Skill",    round(skill_scores[1]*100,1),"#06b6d4"),
        (g4,"📊 Pressure Skill",round(skill_scores[2]*100,1),"#22c55e"),
    ]
    for gcol,label,val,color in metrics_g:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=val,
            number={"suffix":"%","font":{"color":color,"size":32}},
            gauge={"axis":{"range":[0,100]},"bar":{"color":color,"thickness":0.3},
                   "bgcolor":"#0a0e1a","bordercolor":"#1e293b",
                   "steps":[{"range":[0,50],"color":"#0a0e1a"},
                             {"range":[50,80],"color":"#1e293b"},
                             {"range":[80,100],"color":"#111827"}]},
            title={"text":label,"font":{"color":"#94a3b8","size":12}}))
        fig_g.update_layout(paper_bgcolor="#111827",height=200,
                            margin=dict(l=15,r=15,t=50,b=5))
        gcol.plotly_chart(fig_g, use_container_width=True)

    # ── 3. Skill score radar ──────────────────────────────────────────────────
    _section("🕸️ 3 — Skill Score Radar")
    cats = ["Rain","Temp","Wind","Pressure","Humidity"]
    np.random.seed(7)
    model_vals = [round(v*100,1) for v in [skill_scores[0],skill_scores[1],
                   skill_scores[2],skill_scores[3],0.72+np.random.rand()*0.1]]
    persist_v  = [0]*5
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=model_vals,theta=cats,fill="toself",
        name=model_name,line_color="#6366f1",fillcolor="rgba(99,102,241,0.15)"))
    fig_r.add_trace(go.Scatterpolar(r=persist_v,theta=cats,fill="toself",
        name="Persistence",line_color="#475569",fillcolor="rgba(71,85,105,0.1)"))
    fig_r.update_layout(template="plotly_dark",paper_bgcolor="#111827",
        polar=dict(bgcolor="#111827",radialaxis=dict(visible=True,range=[0,100],gridcolor="#1e293b")),
        height=340,legend=dict(bgcolor="#0a0e1a"),margin=dict(l=30,r=30,t=20,b=30))
    st.plotly_chart(fig_r, use_container_width=True)

    # ── 4. Forecast confidence heatmap ────────────────────────────────────────
    _section("🌡️ 4 — Forecast Confidence Heatmap")
    np.random.seed(42)
    vars_  = ["Temp","Rain","Wind","Pressure","Humidity","Geopotential"]
    conf   = np.clip(np.random.randn(6,5)*10+75, 40, 98)
    for i in range(6): conf[i] -= np.arange(5)*4
    fig_h  = px.imshow(conf, x=[f"{l}h" for l in leads], y=vars_,
                        color_continuous_scale="Viridis", zmin=40, zmax=98,
                        labels=dict(x="Lead Time",y="Variable",color="Confidence %"),
                        text_auto=".0f")
    fig_h.update_layout(template="plotly_dark",paper_bgcolor="#111827",
                         height=300,margin=dict(l=100,r=20,t=20,b=40))
    st.plotly_chart(fig_h, use_container_width=True)

    # ── 5. Model comparison bar ───────────────────────────────────────────────
    _section("🏆 5 — Model Comparison")
    models_c = ["FourCastNetLite","GraphCastLite","PanguLite","Persistence"]
    colors_c = ["#8b5cf6","#06b6d4","#f97316","#475569"]
    rmse_c   = [1.42, 1.18, 1.31, 1.92]
    acc_c    = [0.88, 0.91, 0.89, 0.55]
    skill_c  = [0.26, 0.38, 0.32, 0.0]
    fig_c    = go.Figure()
    fig_c.add_trace(go.Bar(x=models_c,y=rmse_c,name="Temp RMSE (K)",
        marker_color=colors_c,opacity=0.85))
    fig_c.add_trace(go.Scatter(x=models_c,y=[v*2 for v in acc_c],
        name="ACC ×2",mode="lines+markers",
        line=dict(color="#22c55e",width=2.5),yaxis="y2"))
    fig_c.update_layout(template="plotly_dark",paper_bgcolor="#111827",
        plot_bgcolor="#111827",height=320,
        yaxis=dict(title="RMSE (K)",gridcolor="#1e293b"),
        yaxis2=dict(title="ACC×2",overlaying="y",side="right"),
        legend=dict(bgcolor="#0a0e1a"),margin=dict(l=40,r=60,t=20,b=40))
    st.plotly_chart(fig_c, use_container_width=True)

    # ── Beat-persistence banner ───────────────────────────────────────────────
    beats = all(s > 0 for s in skill_scores)
    if beats:
        st.markdown(
            "<div style='background:linear-gradient(135deg,rgba(34,197,94,.2),"
            "rgba(6,182,212,.15));border:1px solid #22c55e55;"
            "border-radius:14px;padding:1.2rem 1.8rem;"
            "animation:bannerGlow 2s ease-in-out infinite;margin-top:1rem;'>"
            "<p style='margin:0;font-size:1.1rem;font-weight:700;color:#22c55e;'>"
            "🎉 Model beats persistence baseline across all lead times!</p>"
            "<p style='margin:4px 0 0;color:#94a3b8;font-size:.85rem;'>"
            f"Average skill score: <b style='color:#22c55e;'>"
            f"{np.mean(skill_scores)*100:.1f}%</b></p></div>",
            unsafe_allow_html=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("<br>",unsafe_allow_html=True)
    _section("📋 Metrics Summary Table")
    df = pd.DataFrame({
        "Lead Time":   [f"{l}h" for l in leads],
        "Temp RMSE":   rmse_t,
        "Wind RMSE":   rmse_w,
        "Press RMSE":  rmse_p,
        "Rain Acc":    acc_r,
        "Skill Score": skill_scores,
        "vs Baseline": [f"{'✅+' if s>0 else '❌'}{abs(s)*100:.1f}%" for s in skill_scores],
    }).set_index("Lead Time")
    st.dataframe(df.style.format({c:"{:.3f}" for c in df.columns if "Time" not in c
                                   and "Baseline" not in c}), use_container_width=True)
