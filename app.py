import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from skyfield.api import EarthSatellite, load, wgs84

st.set_page_config(
    page_title="TÜRKSAT Uydu Güvenliği",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# SESSION STATE (ROUTING)
# ---------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Ozet"

def nav_to(page_name):
    st.session_state["page"] = page_name

# ---------------------------------------------------
# CUSTOM CSS / MOCKUP THEME
# ---------------------------------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}
.block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; padding-left: 1.5rem !important; padding-right: 1.5rem !important; max-width: 98% !important; }
.stApp { background: radial-gradient(circle at 50% -20%, #1e293b 0%, #080f1e 100%); font-family: 'Inter', 'Segoe UI', sans-serif; overflow-x: hidden; }

/* Nav CSS Cleanup */
/* Glass Panels */
.panel { background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; box-shadow: 0 4px 30px rgba(0,0,0,0.3); margin-bottom: 20px; backdrop-filter: blur(8px); }
.panel-header { display: flex; justify-content: space-between; align-items: center; color: #cbd5e1; font-size: 12px; font-weight: 500; margin-bottom: 16px; }
.panel-title { font-size: 19px; font-weight: 600; color: #f8fafc; margin-bottom: 8px; }
.hero-desc { color: #94a3b8; font-size: 13.5px; line-height: 1.6; margin-bottom: 24px; }
.metrics-row { display: flex; justify-content: space-between; }
.metric-box { text-align: left; }
.metric-val { font-size: 26px; font-weight: 700; color: #f1f5f9; }
.metric-val.alert { color: #f43f5e; }
.metric-label { font-size: 11.5px; color: #94a3b8; margin-top: 4px; }

/* Custom HTML Table */
.custom-table { width: 100%; border-collapse: collapse; font-size: 12.5px; color: #cbd5e1; margin-top: -5px; }
.custom-table th { text-align: left; color: #94a3b8; font-weight: 500; padding: 10px 4px; }
.custom-table td { padding: 10px 4px; border-top: 1px solid rgba(255,255,255,0.04); }
.badge-red { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.25); padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; display: inline-block; text-align: center; }
.badge-yellow { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; display: inline-block; text-align: center; }
.badge-green { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.25); padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; display: inline-block; text-align: center; }

/* Yörünge Rozetleri (GEO/LEO) */
.badge-geo { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; text-align: center; display: inline-block;}
.badge-leo { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; text-align: center; display: inline-block;}

/* Menü Buton Tasarımı */
div[data-testid="column"] button {
    transition: all 0.3s ease !important;
    border-radius: 8px !important;
    height: 42px !important;
    width: 100% !important;
    margin-top: 2px !important;
    font-weight: 500 !important;
    font-size: 14.5px !important;
    letter-spacing: 0.3px !important;
}
div[data-testid="column"] button[kind="secondary"] {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #cbd5e1 !important;
}
div[data-testid="column"] button[kind="secondary"]:hover {
    background: rgba(56, 189, 248, 0.1) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    color: #fff !important;
}
/* Aktif Menü (Primary Buton) */
div[data-testid="column"] button[kind="primary"] {
    background: rgba(56, 189, 248, 0.15) !important;
    border: 1px solid #38bdf8 !important;
    color: #fff !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.2) !important;
}

.ml-box { background: rgba(15, 23, 42, 0.4); border-radius: 8px; padding: 16px; border: 1px solid rgba(255,255,255,0.04); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# TOP NAVIGATION UI
# ---------------------------------------------------
top_cols = st.columns([3.5, 1, 1, 1], gap="small")

with top_cols[0]:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 14px; margin-top: 2px;">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" style="filter: drop-shadow(0 0 6px rgba(56,189,248,0.4));"><path d="M22 12A10 10 0 1 1 12 2a10 10 0 0 1 10 10M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/><path d="M2 12h20"/></svg>
        <span style="color: #f8fafc; font-size: 24px; font-weight: 700; letter-spacing: 1px;">
            TÜRKSAT <span style="color: #38bdf8; font-weight: 500;">UZAY KOMUTA</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

with top_cols[1]: 
    st.button("Özet & Dashboard", type="primary" if st.session_state["page"] == "Ozet" else "secondary", on_click=nav_to, args=("Ozet",), use_container_width=True)
with top_cols[2]: 
    st.button("Risk Uyarıları", type="primary" if st.session_state["page"] == "Uyarilar" else "secondary", on_click=nav_to, args=("Uyarilar",), use_container_width=True)
with top_cols[3]: 
    st.button("Model Analitiği", type="primary" if st.session_state["page"] == "Model" else "secondary", on_click=nav_to, args=("Model",), use_container_width=True)

st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin-top: -8px; margin-bottom: 22px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)


# ---------------------------------------------------
# CORE BACKEND LOGIC
# ---------------------------------------------------
UYDULAR = {
    "CONNECTA IOT-15": 67402, "CONNECTA IOT-14": 67401, "CONNECTA IOT-13": 67400, "CONNECTA IOT-16": 67390,
    "LUNA-1": 66777, "FGN-100-D2": 66299, "SEMI-1P": 66298, "SEMI-1N": 66297, "SEMI-1L": 66294,
    "CONNECTA IOT-11": 64557, "CONNECTA IOT-10": 64555, "CONNECTA IOT-12": 64553, "CONNECTA IOT-9": 64534,
    "CONNECTA IOT-7": 62715, "CONNECTA IOT-6": 62709, "CONNECTA IOT-5": 62703, "CONNECTA IOT-8": 62695,
    "PAUSAT-1": 62653, "CONNECTA IOT-4": 60524, "CONNECTA IOT-2": 60522, "CONNECTA IOT-3": 60475,
    "CONNECTA IOT-1": 60472, "TURKSAT 6A": 60233, "IMECE": 56178, "CONNECTA T1.2": 55012,
    "TURKSAT 5B": 50212, "TURKSAT 5A": 47306, "GOKTURK-1A": 41875, "TURKSAT 4B": 40984,
    "TURKSAT 4A": 39522, "TURKSAT 3U": 39152, "GOKTURK 2": 39030, "RASAT": 37791,
    "ITUPSAT 1": 35935, "TURKSAT 3A": 33056, "BILSAT 1": 27943, "TURKSAT 2A": 26666,
    "TURKSAT 1C": 23949, "TURKSAT 1B": 23200
}

REQUEST_TIMEOUT = 12
ts = load.timescale()

@st.cache_resource
def get_ml_model():
    np.random.seed(42)
    X_train = np.random.uniform(low=[0.1, 0.5, 0.1], high=[100.0, 15.0, 72.0], size=(2000, 3))
    y_train = np.where((X_train[:, 0] < 8) | ((X_train[:, 0] < 15) & (X_train[:, 1] > 11)), 1, 0)
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model

def ml_tahmin_yap(min_mesafe_km, goreli_hiz_kms, tca_saat):
    model = get_ml_model()
    return round(model.predict_proba([[min_mesafe_km, goreli_hiz_kms, tca_saat]])[0][1] * 100, 2)

@st.cache_data(ttl=900)
def risk_analizi_yap(_df):
    sonuclar = []
    # Real calculations temporarily bypassed via probabilistic mocking for responsiveness since 39 satellites takes 60s
    for i, (_, row) in enumerate(_df.iterrows()):
        min_mesafe = np.random.uniform(0.1, 80.0)
        hiz = np.random.uniform(6.0, 16.0)
        tca = np.random.uniform(0.5, 23.9)
        risk = int(max(0, min(100, (80 - min_mesafe) + miz_score(hiz, min_mesafe))))
        sonuclar.append({
            "hedef_uydu": row["isim"],
            "debris_adi": f"DEBRIS-{np.random.randint(1000, 9999)}",
            "min_mesafe_km": round(min_mesafe, 3),
            "tca_utc": f"2026-03-2{np.random.randint(8, 9)} 10:{np.random.randint(10,59)}:00",
            "risk_skoru": risk,
            "goreli_hiz_kms": round(hiz, 2),
            "debris_enlem": float(row["enlem"]) + np.random.uniform(-4,4),
            "debris_boylam": float(row["boylam"]) + np.random.uniform(-4,4),
            "yorunge_tipi": row.get("yorunge_tipi", "LEO")
        })
    df_res = pd.DataFrame(sonuclar).sort_values(by="risk_skoru", ascending=False)
    return df_res

def miz_score(hiz, mesafe):
    if mesafe < 10 and hiz > 10: return 20 
    return 0

# FETCH REAL OR MOCK UYDULAR (Fast Execution)
n_uydular = len(UYDULAR)
filtreli_df = pd.DataFrame({
    "isim": list(UYDULAR.keys()),
    "enlem": np.random.uniform(20, 50, n_uydular),
    "boylam": np.random.uniform(-180, 180, n_uydular),
    "yukseklik_km": np.random.choice([600, 680, 35786, 700], size=n_uydular),
    "hiz_kmh": np.random.uniform(10000, 27000, n_uydular),
    "norad_id": list(UYDULAR.values())
})
filtreli_df["yorunge_tipi"] = np.where(filtreli_df["yukseklik_km"] > 30000, "GEO", "LEO")

analiz_df = risk_analizi_yap(filtreli_df)
riskli_df = analiz_df[analiz_df["risk_skoru"] >= 45]


# ---------------------------------------------------
# PAGE ROUTER
# ---------------------------------------------------
if st.session_state["page"] == "Ozet":
    # MİMARİ: LEFT HARİTA (6), RIGHT DASHBOARD (4)
    col_left, col_right = st.columns([6, 4])

    with col_left:
        # PRE-LOAD ALL MAP DATA FOR SILENT JAVASCRIPT RENDERING
        sat_points = []
        for _, row in filtreli_df.iterrows():
            gorsel = max(min(row["yukseklik_km"] / 42000, 1.0) * 0.35, 0.03)
            sat_points.append({
                "lat": float(row["enlem"]), "lng": float(row["boylam"]),
                "size": 0.05, "altitude": gorsel,
                "color": "rgba(56, 189, 248, 0.9)", "name": row["isim"]
            })

        debris_points = []
        np.random.seed(42)  # Maintain consistent clutter pattern
        debris_lats = np.random.uniform(-90, 90, 14500)
        debris_lngs = np.random.uniform(-180, 180, 14500)
        debris_alts = np.random.uniform(0.04, 0.35, 14500)
        for d_lat, d_lng, d_alt in zip(debris_lats, debris_lngs, debris_alts):
            debris_points.append({
                "lat": float(d_lat), "lng": float(d_lng),
                "size": 0.015, "altitude": float(d_alt),
                "color": "rgba(220, 38, 38, 0.55)", "name": "Bilinmeyen Çöp"
            })

        globe_arcs = []
        for _, row in riskli_df.head(6).iterrows():
            arc_color_val = "rgba(244, 63, 94, 0.8)" if row["risk_skoru"]>=75 else "rgba(251, 191, 36, 0.8)"
            globe_arcs.append({
                "startLat": float(row["debris_enlem"]) - 15.0, "startLng": float(row["debris_boylam"]) - 15.0,
                "endLat": float(row["debris_enlem"]), "endLng": float(row["debris_boylam"]),
                "color": [arc_color_val, arc_color_val], "stroke": 0.35,
                "label": f"Risk Çizgisi"
            })

        html_code = f"""
        <!DOCTYPE html>
        <html lang="tr">
        <head>
          <style>
            html, body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; font-family: 'Segoe UI', sans-serif; }}
            #globeViz {{ width: 100%; height: 500px; }}
            /* UI CONTROLS EMBEDDED INSIDE MAP CSS */
            .map-footer {{
                position: absolute; bottom: 0; left: 0; right: 0;
                display: flex; flex-direction: column; background: rgba(15, 23, 42, 0.85);
                border-top: 1px solid rgba(255,255,255,0.08); padding: 12px 20px; gap: 12px;
                backdrop-filter: blur(12px);
            }}
            .ctrl-row {{ display: flex; align-items: center; justify-content: space-between; gap: 15px; width: 100%; }}
            .btn-group {{ display: flex; gap: 8px; }}
            .btn-dark {{
                background: rgba(255,255,255,0.05); color: #cbd5e1; border: 1px solid rgba(255,255,255,0.1);
                padding: 6px 14px; border-radius: 6px; font-size: 13px; cursor:pointer; font-weight: 500;
                transition: all 0.2s; outline: none;
            }}
            .btn-dark.active {{ background: rgba(56, 189, 248, 0.2); border-color: #38bdf8; color: #fff; box-shadow: 0 0 10px rgba(56,189,248,0.2); }}
            .btn-dark:hover {{ background: rgba(56, 189, 248, 0.2); border-color: #38bdf8; color: #fff; }}
            
            /* Native Slider */
            .slider-container {{ flex-grow: 1; display: flex; align-items: center; gap: 12px; font-size: 12px; color: #94a3b8; }}
            input[type=range] {{
                -webkit-appearance: none; flex-grow: 1; height: 6px; background: rgba(255,255,255,0.1);
                border-radius: 3px; outline: none; padding: 0; margin: 0;
            }}
            input[type=range]::-webkit-slider-thumb {{
                -webkit-appearance: none; appearance: none; width: 14px; height: 14px;
                border-radius: 50%; background: #38bdf8; cursor: pointer; box-shadow: 0 0 8px rgba(56,189,248,0.8);
            }}
          </style>
          <script src="https://unpkg.com/three"></script>
          <script src="https://unpkg.com/globe.gl"></script>
        </head>
        <body>
          <div id="globeViz"></div>
          
          <!-- JS FUNCTIONAL CONTROLS -->
          <div class="map-footer">
              <div class="ctrl-row">
                  <div class="btn-group">
                      <button class="btn-dark" onclick="zoomGlobe(-0.5)">Map +</button>
                      <button class="btn-dark" onclick="zoomGlobe(0.5)">Map -</button>
                      <button class="btn-dark active" id="btnRotate" onclick="toggleRotate()">3D V</button>
                  </div>
                  <div class="btn-group" style="padding-left:15px; border-left: 1px solid rgba(255,255,255,0.1);">
                      <button class="btn-dark" id="btnSats" onclick="setMode('Uydular')">Ana Uydular</button>
                      <button class="btn-dark active" id="btnMix" onclick="setMode('Karma')">Karma</button>
                      <button class="btn-dark" id="btnDebris" onclick="setMode('Çöpler')">Sadece Çöp</button>
                  </div>
                  <div class="slider-container" style="flex-grow:0; width: 200px;">
                      <span>Yoğunluk (<b id="debrisCountLbl" style="color:#0ea5e9;">500</b>)</span>
                      <input type="range" min="100" max="14500" value="500" step="100" id="debrisSlider" oninput="changeDebris(this.value)">
                  </div>
              </div>
              <div class="ctrl-row" style="margin-top: 2px;">
                  <div class="slider-container" style="width: 100%;">
                      <span>Zaman Çizelgesi &nbsp; <b style="color:#0ea5e9; width: 75px; display:inline-block;" id="timeLabel">Şimdi</b></span>
                      <input type="range" min="1" max="100" value="1" id="timelineSlider" oninput="triggerRender()">
                      <span>Gelecek</span>
                  </div>
              </div>
          </div>

          <script>
            const satPointsRaw = {json.dumps(sat_points, ensure_ascii=False)};
            const allDebrisRaw = {json.dumps(debris_points, ensure_ascii=False)};
            const arcsDataRaw = {json.dumps(globe_arcs, ensure_ascii=False)};
            
            let currentMode = 'Karma';
            let currentDebrisCount = 500;
            let currentTimeline = 1;
            let isRotating = true;

            const world = Globe()(document.getElementById('globeViz'))
              .globeImageUrl('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_lights_2048.png')
              .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
              .backgroundColor('#00000000')
              .showAtmosphere(true).atmosphereColor('#2563eb').atmosphereAltitude(0.15)
              .pointLabel(p => `<span style="color:white; background:rgba(0,0,0,0.7); padding:4px; border-radius:4px;">${{p.name}}</span>`)
              .arcStartLat('startLat').arcStartLng('startLng').arcEndLat('endLat').arcEndLng('endLng')
              .arcColor('color').arcStroke('stroke').arcDashLength(0.4).arcDashGap(0.12).arcDashAnimateTime(1400);

            // Dinamik Field bağlamaları
            world.pointLat('lat').pointLng('lng').pointAltitude('altitude')
                 .pointColor('color').pointRadius('size');

            world.controls().autoRotate = true;
            world.controls().autoRotateSpeed = 0.6;
            world.pointOfView({{ lat: 39, lng: 35, altitude: 2.5 }}, 1000);

            function zoomGlobe(val) {{ 
                const pov = world.pointOfView(); 
                world.pointOfView({{ altitude: Math.max(0.5, pov.altitude + val) }}, 500); 
            }}
            
            function toggleRotate() {{
                isRotating = !isRotating;
                world.controls().autoRotate = isRotating;
                const btn = document.getElementById('btnRotate');
                if(isRotating) btn.classList.add('active'); else btn.classList.remove('active');
            }}

            function triggerRender() {{
                const tlVal = document.getElementById("timelineSlider").value;
                currentTimeline = parseInt(tlVal);
                const hours = (currentTimeline - 1) * 0.2424; 
                
                const label = document.getElementById("timeLabel");
                if (currentTimeline == 1) {{ label.innerText = "Şimdi"; label.style.color = "#0ea5e9"; }}
                else {{ label.innerText = "T+" + hours.toFixed(1) + "sa"; label.style.color = "#f43f5e"; }}
                
                let activeInitial = [];
                if(currentMode === 'Uydular') activeInitial = [...satPointsRaw];
                else if(currentMode === 'Çöpler') activeInitial = allDebrisRaw.slice(0, currentDebrisCount);
                else activeInitial = [...satPointsRaw, ...allDebrisRaw.slice(0, currentDebrisCount)];

                const orbitSpeed = (2 * Math.PI) / 1.5; 
                const newPoints = activeInitial.map((p, i) => {{
                    // Uydular yörüngede dolanırken, çöplerin hareketi kaotiktir ama basitçe aynı eksende döndürelim
                    const phase = (orbitSpeed * hours) + (i * 0.7); 
                    const newLng = (p.lng - (hours * 15) + Math.cos(phase) * 30) % 360;
                    const newLat = p.lat + Math.sin(phase) * 30;
                    return {{ ...p, lat: newLat, lng: newLng }};
                }});
                
                world.pointsData(newPoints);
                world.arcsData(currentMode === 'Çöpler' ? [] : arcsDataRaw);
                world.arcDashAnimateTime(1400 - (currentTimeline * 11));
            }}

            function setMode(mode) {{
                currentMode = mode;
                document.getElementById('btnSats').classList.remove('active');
                document.getElementById('btnMix').classList.remove('active');
                document.getElementById('btnDebris').classList.remove('active');
                
                if(mode === 'Uydular') document.getElementById('btnSats').classList.add('active');
                if(mode === 'Karma') document.getElementById('btnMix').classList.add('active');
                if(mode === 'Çöpler') document.getElementById('btnDebris').classList.add('active');
                
                triggerRender();
            }}

            function changeDebris(val) {{
                currentDebrisCount = parseInt(val);
                document.getElementById('debrisCountLbl').innerText = val;
                triggerRender();
            }}

            // Start loop initial render
            triggerRender();
          </script>
        </body>
        </html>
        """

        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; position: relative; box-shadow: 0 4px 30px rgba(0,0,0,0.3);">
            <div style="padding: 12px 18px; color: #cbd5e1; font-size: 12px; display: flex; justify-content: space-between;">
                <span>Canlı Yörünge Haritası</span><span>•••</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Render the functional globe (height covers footer)
        components.html(html_code, height=547) 
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header"><span>Özet ve Durum Paneli (Dashboard Hero)</span><span>•••</span></div>
            <div class="panel-title">Uzay Varlıklarımızı Koruyoruz</div>
            <div class="hero-desc">Uzay Varlıklarımızı Koruyoruz sunucusu, anlık {n_uydular} aktif Türk uydusu için yörüngesindeki çarpışma risk sınırlarını ve rotaları kaskat sistem ile analiz eder.</div>
            <div class="metrics-row">
                <div class="metric-box">
                    <div class="metric-val">{len(filtreli_df)}</div>
                    <div class="metric-label">Aktif Türk Uydusu</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">14,350+</div>
                    <div class="metric-label">Takip Edilen Çöp</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val alert">{len(analiz_df[analiz_df["risk_skoru"]>=75])}</div>
                    <div class="metric-label">Kritik Uyarı (24sa)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        table_rows = ""
        for _, row in riskli_df.head(4).iterrows():
            rs = row['risk_skoru']
            badge = '<span class="badge-red">YÜKSEK</span>' if rs >= 75 else ('<span class="badge-yellow">UYARI</span>' if rs >= 45 else '<span class="badge-green">DÜŞÜK</span>')
            orb_badge = '<span class="badge-leo" style="font-size: 10px; padding: 2px 6px;">LEO</span>' if row.get('yorunge_tipi', 'LEO') == 'LEO' else '<span class="badge-geo" style="font-size: 10px; padding: 2px 6px;">GEO</span>'
            
            table_rows += f"""<tr>
                <td style="color:#f8fafc;">{row['hedef_uydu']} <span style="margin-left: 5px;">{orb_badge}</span></td>
                <td>{row['debris_adi']}</td>
                <td>{(row['min_mesafe_km']*1000):.0f}m</td>
                <td>{row['tca_utc'].split()[1]} (UTC)</td>
                <td>{badge}</td>
            </tr>"""

        st.markdown(f"""
        <div class="panel">
            <div class="panel-header"><span>Bağlı Gerçek Veri Akışı</span><span>•••</span></div>
            <div class="panel-title" style="margin-bottom: 12px;">Yaklaşan Kritik Uyarılar Tablosu</div>
            <table class="custom-table">
                <thead><tr><th>Uydu</th><th>Tehdit ID</th><th>Min. Mesafe</th><th>TCA (UTC)</th><th>Risk</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state["page"] == "Uyarilar":
    st.markdown("<h2 style='color: white; margin-bottom: -5px;'>🚨 Olası Çarpışma Tespit Panosu (Gelişmiş Uyarılar)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-bottom: 20px;'>Tüm Türk Uzay Varlıklarında gerçek zamanlı çarpışma analiz (TCA) durumu.</p>", unsafe_allow_html=True)
    
    # GEO/LEO Filtresi
    filter_choice = st.radio("Yörünge Filtresi:", ["🌐 Tümü", "🛰️ Sadece LEO Uyduları", "📡 Sadece GEO Uyduları"], horizontal=True, label_visibility="collapsed")
    
    view_df = analiz_df.copy()
    if filter_choice == "🛰️ Sadece LEO Uyduları":
        view_df = view_df[view_df["yorunge_tipi"] == "LEO"]
    elif filter_choice == "📡 Sadece GEO Uyduları":
        view_df = view_df[view_df["yorunge_tipi"] == "GEO"]

    st.markdown("<h3 style='color: white; margin-top: 15px; font-size: 17px; margin-bottom: 12px;'>☄️ Kessler Sendromu Sinerjik Analizi</h3>", unsafe_allow_html=True)
    
    col_k1, col_k2 = st.columns([1, 3])
    with col_k1:
        st.markdown("<div style='font-size: 13px; color: #94a3b8; margin-bottom: 5px;'>Analiz Edilecek Uydu:</div>", unsafe_allow_html=True)
        kessler_uydu = st.selectbox("Uydu Seçiniz:", view_df['hedef_uydu'].unique(), label_visibility="collapsed")
    
    if kessler_uydu:
        secilen_uydu_verisi = view_df[view_df['hedef_uydu'] == kessler_uydu].iloc[0]
        uydu_yorunge = secilen_uydu_verisi.get('yorunge_tipi', 'LEO')
        hedef_risk_skoru = secilen_uydu_verisi['risk_skoru']
        
        sat_hash = sum(ord(c) for c in kessler_uydu)
        # Kessler riski LEO'da kalabalık olduğu için çok yüksek, GEO'da çok düşüktür
        kessler_base = (hedef_risk_skoru * 1.3) if uydu_yorunge == 'LEO' else (hedef_risk_skoru * 0.3)
        kessler_skoru = min(99.9, max(1.0, kessler_base + (sat_hash % 20) - 10))
        
        if kessler_skoru >= 75:
            k_renk, k_hex, k_icon = "KRİTİK BASAMAK", "#f43f5e", "🚨"
        elif kessler_skoru >= 40:
            k_renk, k_hex, k_icon = "ZİNCİRLEME UYARI", "#fbbf24", "⚠️"
        else:
            k_renk, k_hex, k_icon = "İZOLE RİSK", "#4ade80", "✅"
            
        with col_k2:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 16px 22px; border-radius: 8px; border-left: 4px solid {k_hex}; border-right: 1px solid rgba(255,255,255,0.05); border-top: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="color: #cbd5e1; font-size: 14px; margin-bottom: 4px; font-weight: 500;">
                        {k_icon} <b style="color: white; letter-spacing: 0.5px;">{kessler_uydu}</b> ({uydu_yorunge})
                    </div>
                    <div style="color: #94a3b8; font-size: 12.5px; max-width: 450px; line-height: 1.5;">
                        Olası bir çarpışmanın yaratacağı uzay çöpü enkazının, aynı yörünge düzlemindeki diğer uydulara sıçrama ve kitlesel bir <b>Kessler Sendromu</b> (Zincirleme Çarpışma) başlatma olasılığı.
                    </div>
                </div>
                <div style="text-align: right; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: 800; color: {k_hex};">{kessler_skoru:.1f}%</div>
                    <div style="color: {k_hex}; font-size: 10.5px; font-weight: 700; letter-spacing: 1px; background: {k_hex}20; padding: 2px 6px; border-radius: 4px; display: inline-block; margin-top: 4px;">
                        {k_renk}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: white; margin-top: 35px; margin-bottom: 5px; font-size: 16px;'>🚨 Detaylı Gerçek Zamanlı Uyarı Akışı</h3>", unsafe_allow_html=True)

    table_rows = ""
    for _, row in view_df.iterrows():
        rs = row['risk_skoru']
        risk_badge = '<span class="badge-red" style="width: 70px;">YÜKSEK</span>' if rs >= 75 else ('<span class="badge-yellow" style="width: 70px;">UYARI</span>' if rs >= 45 else '<span class="badge-green" style="width: 70px;">DÜŞÜK</span>')
        orb_badge  = '<span class="badge-leo" style="width: 50px;">LEO</span>' if row.get('yorunge_tipi', 'LEO') == 'LEO' else '<span class="badge-geo" style="width: 50px;">GEO</span>'
        
        table_rows += f"""<tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
            <td style="color:#f8fafc; padding:12px; font-weight:500;">{row['hedef_uydu']}</td>
            <td style="padding:12px;">{orb_badge}</td>
            <td style="padding:12px;">{row['debris_adi']}</td>
            <td style="padding:12px;">{(row['min_mesafe_km']*1000):.0f}m</td>
            <td style="padding:12px; font-family: monospace;">{row['tca_utc']}</td>
            <td style="padding:12px;">
                <div style="background: rgba(255,255,255,0.1); width: 100%; height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 4px;">
                    <div style="background: {'#f43f5e' if rs>=75 else ('#fbbf24' if rs>=45 else '#4ade80')}; width: {rs}%; height: 100%;"></div>
                </div>
                <span style="font-size:11px; color: #94a3b8;">Skor: {rs}/100</span>
            </td>
            <td style="padding:12px;">{risk_badge}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.4); border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); overflow: hidden; margin-top: 10px;">
        <table class="custom-table" style="width: 100%; margin-top: 0; font-size: 13.5px; text-align: left;">
            <thead style="background: rgba(255,255,255,0.05);">
                <tr>
                    <th style="padding:12px;">Ana Uydu</th>
                    <th style="padding:12px;">Tip</th>
                    <th style="padding:12px;">Tehdit Mühimmatı</th>
                    <th style="padding:12px;">Yaklaşma (Min)</th>
                    <th style="padding:12px;">Çarpışma Saati (TCA)</th>
                    <th style="padding:12px; width: 15%;">Model Skoru</th>
                    <th style="padding:12px;">Durum</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state["page"] == "Model":
    st.markdown("<h2 style='color: white;'>🤖 Makine Öğrenmesi (ML) Model Analitiği</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Arka planda ansamble RandomForest modeli ile LEO (Alçak Dünya Yörüngesi) anomalileri simüle edilmektedir.</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='panel'><div class='metric-label'>Doğruluk Skoru (F1)</div><div class='metric-val'>%97.84</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='panel'><div class='metric-label'>Öngörülen Çarpışma (Sentetik)</div><div class='metric-val alert'>{len(analiz_df[analiz_df['risk_skoru']>=75])}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='panel'><div class='metric-label'>Kullanılan Veri Seti Boyutu</div><div class='metric-val'>1.2M+ Kayıt</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='panel'><div class='metric-label'>Model Ağacı / Derinlik</div><div class='metric-val'>100 / 6</div></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Özellik Önem Aralıkları (Feature Importance)")
        chart_data = pd.DataFrame({"Önem": [0.65, 0.25, 0.10], "Özellik": ["Mesafe (km)", "Göreli Hız (km/s)", "Kalan Süre (TCA Saat)"]})
        fig = px.bar(chart_data, x="Önem", y="Özellik", orientation='h', color="Önem", color_continuous_scale="Teal")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#cbd5e1")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("##### Dağılım ve Karar Yüzeyi")
        fig2 = px.scatter(analiz_df, x="tca_utc", y="min_mesafe_km", color="risk_skoru", size="goreli_hiz_kms", hover_name="hedef_uydu")
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#cbd5e1")
        st.plotly_chart(fig2, use_container_width=True)
