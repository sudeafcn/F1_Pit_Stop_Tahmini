# 🏎️ F1 Pit-Stop Prediction: A Decision Support System

Bu proje, Formula 1 yarışlarındaki telemetri ve yarış verilerini kullanarak bir pilotun **bir sonraki turda pite girip girmeyeceğini (`PitNextLap`)** tahmin eden makine öğrenmesi tabanlı bir Karar Destek Sistemidir (Decision Support System). Proje, uçtan uca **CRISP-DM** (Cross-Industry Standard Process for Data Mining) metodolojisi takip edilerek geliştirilmiştir.

## 📌 İş Problemi (Business Understanding)
F1'de erken pit-stop trafik riskini artırırken, geç pit-stop lastik patlamasına veya kritik zaman kayıplarına yol açar. Bu model, otonom bir karar verici olmaktan ziyade, pit duvarındaki strateji mühendisleri için tasarlanmış bir **"Erken Uyarı Radarı"** olarak konumlandırılmıştır. Modelin hatalı kararlarının operasyonel maliyeti (Cost of Error) analiz edilerek, yanlış negatifleri (kaçırılan pit-stopları) minimize edecek şekilde **Recall (Duyarlılık)** ve genel ayırma gücünü ölçen **ROC-AUC** metrikleri hedeflenmiştir.

## 🧠 Makine Öğrenmesi Yaklaşımı
Proje kapsamında XGBoost, LightGBM, Random Forest gibi 10 farklı algoritma test edilmiştir. Dengesiz veri setindeki (Imbalanced Data) üstün performansı ve karmaşık stratejik kararları (Race Phase, Stint Load, Tyre Degradation) modelleme yeteneği nedeniyle **XGBoost (Extreme Gradient Boosting)** şampiyon model olarak seçilmiştir.

**Öne Çıkan Özellik Mühendisliği (Feature Engineering) Adımları:**
* `TyreLife_x_RaceProgress`: Yarışın evresine göre lastik aşınmasının anlamsal değişimi.
* `TyreLife_per_Stint`: Mevcut stint içerisindeki lastik yükü stres indeksi.
* `Position_Group`: Pilotların liderlik savaşı veya arka sıra stratejisi psikolojisi.
* `Clipped_LapTime_Delta`: Trafik veya sarı bayrak kaynaklı aykırı değerlerin (outliers) filtrelenmesi.

## 🚀 Kurulum ve Çalıştırma

1. Repoyu bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/sudeafcn/F1_Pit_Stop_Tahmini]
   Gerekli kütüphaneleri yükleyin:

Bash
pip install -r requirements.txt
Jupyter Notebook'u başlatarak analizleri inceleyin:

Bash
jupyter notebook F1_PitStop_CRISP_DM.ipynb
Karar Destek Sistemi (Streamlit) arayüzünü başlatın:

Bash
python -m streamlit run app/app.py
📊 Kullanılan Teknolojiler
Dil: Python

Veri İşleme: Pandas, NumPy

Makine Öğrenmesi: Scikit-Learn, XGBoost, LightGBM

Görselleştirme: Plotly, FastF1 (Telemetri Analizi)

Canlı Arayüz (Deployment): Streamlit
