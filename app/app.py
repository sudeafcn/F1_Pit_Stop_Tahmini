import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import plotly.graph_objects as go

# --- 1. SAYFA VE PRESTİJLİ F1 TEMA AYARLARI ---
st.set_page_config(page_title="Pit Strategy", page_icon="🏎️", layout="wide")

# CSS ile "Pit Wall" Karbon ve Canlı Kırmızı Tasarım Düzeni
st.markdown("""
    <style>
    /* Ana Arka Plan Grafikleri */
    .main {
        background-color: #0B0C10;
        color: #F3F3F3;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Merkezi Kart Tasarımı */
    .dashboard-card {
        background-color: #14161D;
        padding: 35px;
        border-radius: 15px;
        border: 1px solid #222531;
        border-top: 6px solid #e10600;
        box-shadow: 0 10px 30px rgba(0,0,0,0.7);
        margin-bottom: 25px;
    }
    /* Buton Tasarımı */
    .stButton>button {
        background: linear-gradient(90deg, #e10600 0%, #b30000 100%);
        color: white !important;
        font-weight: bold;
        font-size: 18px !important;
        letter-spacing: 1px;
        padding: 12px 0px !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(225, 6, 0, 0.6);
        background: #e10600;
    }
    /* Metrik Alanları */
    .telemetry-title {
        color: #00D2BE;
        font-weight: bold;
        border-bottom: 1px solid #222531;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. MODEL VE PIPELINE YÜKLEME ---
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('../notebooks/deployment/f1_preprocessor.joblib')
        model = joblib.load('../notebooks/deployment/f1_xgb_model.joblib')
        return preprocessor, model
    except:
        return None, None

preprocessor, model = load_models()

# --- 3. MERKEZİ BAŞLIK ALANI ---
st.markdown("<div style='text-align: center; margin-top: 20px; margin-bottom: 40px;'>", unsafe_allow_html=True)
st.title("🏎️ PIT-STOP KARAR DESTEK SİSTEMİ")
st.markdown("<h4 style='color: #94a3b8; font-weight: 400; margin-top: -10px;'>XGBoost Tarafından Güçlendirilmiştir</h4>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- 4. ANA TELEMETRİ PANELİ ---
left_pad, main_col, right_pad = st.columns([1, 10, 1])

with main_col:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("<h3 class='telemetry-title'>📡 Canlı Yarış Telemetrisi ve Koşul Girişi</h3>", unsafe_allow_html=True)
    
    # Grid Düzeni
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    
    with row1_col1:
        lap_number = st.number_input("Şu Anki Tur (Lap Number)", min_value=1, max_value=80, value=25)
    with row1_col2:
        tyre_life = st.number_input("Lastik Yaşı (Atılan Tur)", min_value=1, max_value=50, value=15)
    with row1_col3:
        stint = st.number_input("Mevcut Stint (Bölüm)", min_value=1, max_value=5, value=2)
        
    with row2_col1:
        position = st.number_input("Pist Üstü Pozisyonu (P)", min_value=1, max_value=20, value=3)
    with row2_col2:
        compound = st.selectbox("Lastik Hamuru (Compound)", ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'])
    with row2_col3:
        driver = st.selectbox("Pilot Kataloğu (Driver)", ['VER', 'HAM', 'LEC', 'SAI', 'NOR', 'PIA', 'RUS', 'ALO'])

    st.markdown("<br>", unsafe_allow_html=True)
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        lap_time_delta = st.slider("Tur Süresi Sapması / Tempo Kaybı (Saniye)", min_value=-3.0, max_value=8.0, value=0.5, step=0.1)
    
    with row3_col2:
        # OTOMATİK KÜMÜLATİF AŞINMA MOTORU
        compound_weights = {'SOFT': 2.8, 'MEDIUM': 2.0, 'HARD': 1.3, 'INTERMEDIATE': 2.2, 'WET': 1.8}
        base_weight = compound_weights.get(compound, 2.0)
        
        calculated_degradation = (tyre_life * base_weight) + (max(0.0, lap_time_delta) * 8.5)
        cum_degradation = min(100.0, max(0.0, calculated_degradation))
        
        st.markdown(f"""
            <div style='background-color: #1A1A24; padding: 12px 20px; border-radius: 8px; border-left: 4px solid #00D2BE; margin-top: 15px;'>
                <span style='color: #94a3b8; font-size: 14px;'>🤖 Otomatik Hesaplanan Kümülatif Aşınma Skoru</span><br>
                <span style='color: #fff; font-size: 22px; font-weight: bold; font-family: "Orbitron";'>% {cum_degradation:.1f}</span>
            </div>
        """, unsafe_allow_html=True)

    race_progress = lap_number / 70.0  
    
    st.markdown("<br>", unsafe_allow_html=True)
    execute_button = st.button("🚨 Taktiksel Analizi ve Stratejiyi Çalıştır")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 5. TAHMİN VE RADAR PANELİ ---
    if execute_button:
        if preprocessor is not None and model is not None:
            input_data = pd.DataFrame({
                'LapNumber': [lap_number], 
                'TyreLife': [tyre_life], 
                'Stint': [stint],
                'Position': [position], 
                'LapTime_Delta': [lap_time_delta], 
                'Cumulative_Degradation': [cum_degradation], 
                'RaceProgress': [race_progress],
                'Driver': [driver], 
                'Compound': [compound], 
                'Race': ['Italian Grand Prix'], 
                'Year': [2026], 
                'LapTime (s)': [80.0], 
                'Position_Change': [0], 
                'PitStop': [0]
            })
            
            # Değişken Mühendisliği Adımları
            input_data['Degradation_Rate'] = input_data['Cumulative_Degradation'] / (input_data['TyreLife'] + 0.01)
            input_data['Pace_Struggle_Index'] = input_data['TyreLife'] * input_data['LapTime_Delta']
            input_data['Race_Phase'] = pd.cut(input_data['RaceProgress'], bins=[-0.1, 0.33, 0.66, 1.1], labels=['Erken', 'Orta', 'Gec'])
            input_data['Tyre_Stress_Ratio'] = input_data['TyreLife'] / (input_data['LapNumber'] + 1)
            input_data['High_Degradation_Flag'] = (input_data['Cumulative_Degradation'] > 55).astype(int) 
            input_data['TyreLife_x_RaceProgress'] = input_data['TyreLife'] * input_data['RaceProgress']
            input_data['TyreLife_per_Stint'] = input_data['TyreLife'] / (input_data['Stint'] + 0.01)
            input_data['Position_Group'] = pd.cut(input_data['Position'], bins=[0, 4, 12, 100], labels=['Lider_Grup', 'Orta_Sira', 'Arka_Sira'])
            input_data['Clipped_LapTime_Delta'] = input_data['LapTime_Delta'].clip(lower=-2.0, upper=3.0)
            input_data['Is_Late_Race'] = (input_data['RaceProgress'] > 0.85).astype(int)
            
            input_data = input_data.drop(columns=['LapNumber'])

            # Tahmin
            processed_data = preprocessor.transform(input_data)
            pit_probability = model.predict_proba(processed_data)[0][1] * 100
            
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([12, 10])
            
            with col1:
                st.markdown("<h3 style='color: #fff !important; margin-bottom: 20px;'>📊 Pit Duvarı Operasyon Raporu</h3>", unsafe_allow_html=True)
                if pit_probability > 75:
                    st.error(f"🔴 KRİTİK SEVİYE: % {pit_probability:.1f} | BOX, BOX, BOX!")
                    st.markdown("""
                        <div style='background-color: rgba(225,6,0,0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #e10600; color: #E0E0E0;'>
                            <b>Strateji Alarmı:</b> Lastiklerin mekanik performansı tamamen çökmüş durumda ve araç tempo uçurumuna (Tyre Cliff) girdi. 
                            Gecikilen her tur podyum kaybı riskidir. <b>Pit ekibini sert hamur (HARD) lastiklerle garaj önüne çıkartın!</b>
                        </div>
                    """, unsafe_allow_html=True)
                elif pit_probability > 40:
                    st.warning(f"🟡 TAKİP MODU: % {pit_probability:.1f} | STRATEJİ HAZIRLIĞI")
                    st.markdown("""
                        <div style='background-color: rgba(241,196,15,0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #f1c40f; color: #E0E0E0;'>
                            <b>Strateji Alarmı:</b> Lastik aşınma eğrisi kritik bölge sınırında. Pilot telsizden kayma bildirebilir. 
                            B planı (Undercut önleme) için pit ekibini hazırda bekletin ve bir sonraki sektör sürelerini anlık izleyin.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"🟢 STABİL DURUM: % {pit_probability:.1f} | PİSTE DEVAM")
                    st.markdown("""
                        <div style='background-color: rgba(46,204,113,0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #2ecc71; color: #E0E0E0;'>
                            <b>Strateji Alarmı:</b> Telemetri verileri yeşil bölgede. Lastik aşınma hızı ve araç temposu planlanan yarış simülasyonuyla kusursuz uyum gösteriyor. 
                            Mevcut sürüş bölümünü (Stint) uzatıp pist üstü avantajını koruyun.
                        </div>
                    """, unsafe_allow_html=True)
                    
            with col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pit_probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#ffffff", 'thickness': 0.25},
                        'bgcolor': "#14161D",
                        'borderwidth': 1,
                        'bordercolor': "#334155",
                        'steps': [
                            {'range': [0, 40], 'color': "#2ecc71"},
                            {'range': [40, 75], 'color': "#f1c40f"},
                            {'range': [75, 100], 'color': "#e10600"}],
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white", 'family': "Orbitron"}, 
                    height=260,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Tahmin yürütülemedi. Lütfen model dosyalarının deployment klasöründe (.joblib) mevcut olduğundan emin olun.")
    else:
        st.markdown("<div style='text-align: center; color: #64748b; padding: 20px;'>👈 Telemetri girdilerini kontrol ettikten sonra 'Stratejiyi Çalıştır' butonuna basarak yapay zeka analiz raporunu üretebilirsiniz.</div>", unsafe_allow_html=True)