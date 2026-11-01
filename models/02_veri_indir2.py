import pandas as pd
import os

# ---------------- 1. BÖLÜM: KATALOĞU OKUMA VE FİLTRELEME ----------------

print("Katalog indiriliyor ve okunuyor, lütfen bekleyin...")
catalog_url = "https://sevir.s3.amazonaws.com/CATALOG.csv"
df = pd.read_csv(catalog_url, low_memory=False)

# Sadece radar (vil) verisini filtrele — eğitimde sadece vil kullanılıyor
final_veri_listesi = df[df['img_type'] == 'vil']
print(f"Toplam VIL olay sayısı: {len(final_veri_listesi)}")
year_counts = final_veri_listesi['file_name'].str.extract(r'(\d{4})_\d{4}_\d{4}')[0].value_counts().sort_index()
print(f"Yıl dağılımı:\n{year_counts}")

 
# ---------------- 2. BÖLÜM: BİLGİSAYARA KAYDETME VE İNDİRME DOSYASI HAZIRLAMA ----------------

# 1. Proje Dizinimiz (Spyder'ın o an çalıştığı mevcut klasör)
proje_dizini = os.getcwd()

# ---------------- 2. BÖLÜM: BİLGİSAYARA KAYDETME VE İNDİRME DOSYASI HAZIRLAMA ----------------

import os
proje_dizini = os.getcwd()

# Veri dizini
veri_dizini = "/path/to/data_root/"
os.makedirs(veri_dizini, exist_ok=True) 

katalog_yolu = os.path.join(proje_dizini, "benim_tez_katalogum.csv")
final_veri_listesi.to_csv(katalog_yolu, index=False)

indirilecek_dosyalar = final_veri_listesi['file_name'].unique()

bat_dosyasi_yolu = os.path.join(proje_dizini, "verileri_indir.bat")

with open(bat_dosyasi_yolu, "w") as f:
    f.write("@echo off\n")
    f.write("echo SEVIR Verileri Indiriliyor (Boyut Kontrollu SYNC ile)...\n")
    
    for dosya_yolu in indirilecek_dosyalar:
        # S3 ve yerel dizin yollarını ayarla
        s3_dir = f"s3://sevir/data/{os.path.dirname(dosya_yolu)}"
        yerel_klasor = os.path.join(veri_dizini, "data", os.path.dirname(dosya_yolu))
        yerel_klasor = yerel_klasor.replace("\\", "/")
        dosya_adi = os.path.basename(dosya_yolu)
        
        f.write(f"if not exist \"{yerel_klasor}\" mkdir \"{yerel_klasor}\"\n")
        
        # SYNC KOMUTU: Sadece belirttiğimiz dosyayı senkronize eder. 
        # Dosya yarimsa tamamlar, tamamsa hizlica atlar.
        f.write(f"aws s3 sync {s3_dir} \"{yerel_klasor}\" --exclude \"*\" --include \"{dosya_adi}\" --no-sign-request\n")
        
    f.write("echo Tum indirmeler tamamlandi veya dogrulandi!\n")
    f.write("pause\n")

print(f"\nAkıllı indirme scripti (SYNC) başarıyla oluşturuldu!")