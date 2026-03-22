import streamlit as st
import numpy as np
import pandas as pd
from skyfield.api import load

st.set_page_config(page_title="Gerçek Uydu Risk Analizi", page_icon="🛰️", layout="centered")

# --------------------------------------------------
# 1) GERÇEK UYDU VERİSİ ÇEK
# --------------------------------------------------
@st.cache_resource
def load_satellites():
    ts = load.timescale()

    # CelesTrak grup verileri
    urls = {
        "active": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
        "stations": "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
        "visual": "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle",
    }

    sats = []
    for _, url in urls.items():
        try:
            group_sats = load.tle_file(url)
            sats.extend(group_sats)
        except Exception:
            pass

    # İsim tekrarlarını temizle
    unique = {}
    for sat in sats:
        name = sat.name.strip() if sat.name else "UNKNOWN"
        if name not in unique:
            unique[name] = sat

    satellite_names = sorted(unique.keys())
    return ts, unique, satellite_names

ts, satellites, satellite_names = load_satellites()

# --------------------------------------------------
# 2) YARDIMCI FONKSİYONLAR
# --------------------------------------------------
def satellite_position_km(sat, t):
    geocentric = sat.at(t)
    return geocentric.position.km, geocentric.velocity.km_per_s

def compute_risk(distance_km, relative_speed_km_s):
    # Hackathon için basit ama anlaşılır skor
    # Mesafe az ve göreli hız yüksekse risk yükselir
    score = 0.0

    if distance_km < 5:
        score += 0.55
    elif distance_km < 20:
        score += 0.35
    elif distance_km < 50:
        score += 0.18

    if relative_speed_km_s > 10:
        score += 0.35
    elif relative_speed_km_s > 5:
        score += 0.22
    elif relative_speed_km_s > 2:
        score += 0.10

    return min(score, 0.95)

def risk_label(score):
    if score >= 0.6:
        return "Yüksek Risk 🚨"
    elif score >= 0.3:
        return "Orta Risk ⚠️"
    return "Düşük Risk ✅"

# --------------------------------------------------
# 3) UI
# --------------------------------------------------
st.title("🛰️ Gerçek Uydu Çarpışma Risk Analizi")
st.caption("Veri kaynağı: CelesTrak GP/TLE verileri, hesaplama: Skyfield")

if not satellite_names:
    st.error("Uydu verisi yüklenemedi. İnternet bağlantısını kontrol et.")
    st.stop()

default_a = satellite_names.index("ISS (ZARYA)") if "ISS (ZARYA)" in satellite_names else 0
default_b = 1 if len(satellite_names) > 1 else 0

sat1_name = st.selectbox("1. uydu", satellite_names, index=default_a)
sat2_name = st.selectbox("2. uydu", satellite_names, index=default_b)

minutes_ahead = st.slider("Kaç dakika sonrasını analiz edelim?", 0, 180, 10)

if sat1_name == sat2_name:
    st.warning("Lütfen farklı iki uydu seç.")
    st.stop()

if st.button("Analizi Başlat"):
    sat1 = satellites[sat1_name]
    sat2 = satellites[sat2_name]

    t_now = ts.now()
    # Skyfield zaman nesnesi üstünden dakika eklemek yerine UTC bileşenleriyle ilerlemek daha sade:
    dt = t_now.utc_datetime()
    from datetime import timedelta, timezone
    target_dt = dt + timedelta(minutes=minutes_ahead)
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)

    t = ts.from_datetime(target_dt)

    pos1, vel1 = satellite_position_km(sat1, t)
    pos2, vel2 = satellite_position_km(sat2, t)

    distance_km = float(np.linalg.norm(pos1 - pos2))
    relative_speed_km_s = float(np.linalg.norm(vel1 - vel2))

    score = compute_risk(distance_km, relative_speed_km_s)
    label = risk_label(score)

    st.subheader("Sonuç")
    st.write(f"**Analiz zamanı (UTC):** {target_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Uydu 1:** {sat1_name}")
    st.write(f"**Uydu 2:** {sat2_name}")
    st.write(f"**Tahmini mesafe:** {distance_km:,.2f} km")
    st.write(f"**Göreli hız:** {relative_speed_km_s:.2f} km/s")
    st.write(f"**Risk skoru:** %{score * 100:.1f}")

    st.progress(int(score * 100))

    if score >= 0.6:
        st.error(label)
    elif score >= 0.3:
        st.warning(label)
    else:
        st.success(label)

    # Basit tablo
    result_df = pd.DataFrame({
        "Metrik": ["Mesafe (km)", "Göreli Hız (km/s)", "Risk Skoru (%)"],
        "Değer": [round(distance_km, 2), round(relative_speed_km_s, 2), round(score * 100, 2)]
    })
    st.dataframe(result_df, use_container_width=True)

st.markdown("---")
st.markdown("### Notlar")
st.markdown("""
- Bu sürüm gerçek uydu yörünge verilerini kullanır.
- Risk skoru şu an hackathon amaçlı bir sezgisel modeldir.
- Bir sonraki adımda bunu zaman serisi taraması yapıp “en yakın yaklaşım anı” bulan sisteme çevirebiliriz.
""")
