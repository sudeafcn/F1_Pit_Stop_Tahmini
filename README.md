# 🏎️ F1 Pit-Stop Stratejisi Tahmin Motoru  🏁

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-172434?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
</div>

## 📌 Proje Hakkında

Formula 1 yarışlarında pit-stop zamanlaması, bir yarışın kazanılmasını veya kaybedilmesini belirleyen en kritik stratejik hamledir. Bu proje, yarış içindeki pilotların **bir sonraki turda pite girip girmeyeceğini** tahmin etmek amacıyla geliştirilmiş bir makine öğrenmesi modelidir. 

Yalnızca bir algoritma denemesi olmaktan ziyade, Yönetim Bilişim Sistemleri (YBS) vizyonuyla ele alınmış; yarış stratejistleri ve pit duvarı mühendisleri için anlık yarış verilerini işleyerek operasyonel tetikleyiciler sunan analitik bir **Karar Destek Sistemi (Decision Support System)** olarak tasarlanmıştır. Geliştirme sürecinde uçtan uca veri madenciliği standardı olan **CRISP-DM** metodolojisi uygulanmıştır.

## 📊 Veri Seti ve Özellikler

Araç içi telemetri sistemlerinden elde edilen anlık durum verileri kullanılmıştır. Sınıf dengesizliği (class imbalance) barındıran bu veri setinde hedef değişkenimiz (`PitNextLap`) pilotun pite girme kararıdır (1 = Girecek, 0 = Girmeyecek).

* **Eğitim Seti (Train):** 439.140 satır
* **Test Seti (Test):** 188.165 satır

**Kullanılan Temel Değişkenler:**
* `TyreLife`: Mevcut lastiğin tur bazında yaşı
* `Compound`: Kullanılan lastik hamuru (Soft, Medium, Hard, Wet, Intermediate)
* `LapTime (s)` & `LapTime_Delta`: Tur süresi ve bir önceki tura göre zaman farkı
* `Cumulative_Degradation`: Kümülatif lastik aşınma göstergesi
* `RaceProgress`: Yarışın tamamlanma yüzdesi
* `Position_Change`: Pilotun anlık sıra kazanım/kaybı

## ⚙️ Modelleme ve Başarı Kriterleri

Modelin yanlış kararlarının operasyonel maliyeti (Cost of Error) çok yüksektir:
* **Yanlış Pozitif (False Positive):** Ekibin gereksiz yere hazırlanması. (Kısmen tolere edilebilir)
* **Yanlış Negatif (False Negative):** Hazırlıksız pit-stop, saniye kayıpları veya lastik patlaması. (Kritik risk)

Model, kesin 0-1 kararları vermek yerine **XGBoost** algoritması ile pite girme ihtimallerini olasılıksal olarak hesaplar (`predict_proba`). Bu sayede takımın risk iştahına göre dinamik eşik değerleri (threshold) belirlenebilir. Değerlendirme metriği olarak **ROC-AUC** skoru odak noktasına alınmıştır.

## 📈 Veri Görselleştirme (EDA)
Proje içerisinde **Plotly** kullanılarak yarış verileri üzerinde interaktif analizler yapılmıştır:
- Lastik ömrüne göre pit ihtimali trendleri.
- Yarış ilerlemesine (Race Progress) göre karar dağılımları.
- Lastik hamuru bazında aşınma ve pite girme oranları.

## 🚀 Kurulum ve Kullanım

Projeyi yerel bilgisayarınızda incelemek ve çalıştırmak için:

1. Bu depoyu klonlayın:
```bash
git clone [https://github.com/sudeafcn/F1_Pit_Stop_Tahmini](https://github.com/sudeafcn/F1_Pit_Stop_Tahmini)
 cd f1-pitstop-prediction
pip install pandas numpy scikit-learn xgboost plotly jupyter
jupyter notebook F1_PitStop_CRISP_DM.ipynb

