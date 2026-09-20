# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

- **Nama:** Hafiz Nazwa Nugraha
- **Email:** hafiznugraha04@gmail.com
- **ID Dicoding:** FIshiwak

---

## Business Understanding

**Jaya Jaya Institut** merupakan institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun
2000 dan telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak
juga siswa yang tidak menyelesaikan pendidikannya alias **dropout**.

Berdasarkan data yang tersedia, **32,1% mahasiswa (1.421 dari 4.424) berakhir dropout** — hampir
sepertiga populasi, dan dua kali lipat di atas batas yang umumnya dianggap wajar untuk pendidikan
tinggi (10–15%). Angka setinggi ini merugikan institusi dari tiga sisi sekaligus:

1. **Reputasi & akreditasi** — rasio kelulusan adalah komponen penilaian akreditasi.
2. **Pendapatan** — setiap mahasiswa yang keluar di tengah jalan berarti kehilangan uang kuliah untuk
   seluruh semester sisanya.
3. **Dampak sosial** — mahasiswa kehilangan investasi waktu dan biaya yang sudah dikeluarkan, sering
   kali dengan sisa tunggakan.

Oleh karena itu, Jaya Jaya Institut ingin **mendeteksi secepat mungkin siswa yang mungkin akan
melakukan dropout** sehingga dapat diberi bimbingan khusus, serta meminta dibuatkan **business
dashboard** agar tim akademik mudah memahami data dan memonitor performa siswa secara mandiri.

### Permasalahan Bisnis

1. **Seberapa besar angka dropout Jaya Jaya Institut saat ini dan berapa mahasiswa yang terdampak?**
2. **Faktor apa saja yang paling memengaruhi keputusan mahasiswa untuk dropout** — faktor akademik,
   finansial, demografis, atau jalur masuk?
3. **Bagaimana mendeteksi mahasiswa berisiko dropout sedini mungkin**, idealnya pada akhir semester
   pertama, agar intervensi masih sempat dilakukan?
4. **Bagaimana tim akademik dapat memonitor performa mahasiswa secara mandiri**, tanpa harus meminta
   analisis ulang setiap kali dibutuhkan?

### Cakupan Proyek

1. **Data Understanding** — memahami struktur, kualitas, dan distribusi dataset performa mahasiswa
   (4.424 baris, 37 kolom).
2. **Exploratory Data Analysis** — menggali faktor-faktor yang berhubungan dengan dropout melalui
   visualisasi data.
3. **Data Preparation** — membentuk target biner, menyeleksi fitur, membagi data latih/uji, dan
   membangun pipeline preprocessing (`ColumnTransformer` + `Pipeline`).
4. **Modeling** — membandingkan tiga algoritma klasifikasi, melakukan *hyperparameter tuning*, dan
   menetapkan *decision threshold* berdasarkan pertimbangan biaya bisnis.
5. **Evaluation** — mengukur performa dengan metrik yang relevan untuk deteksi dini (recall, F1,
   ROC-AUC) dan menafsirkan fitur yang paling berpengaruh.
6. **Deployment** — membangun **business dashboard Metabase** dan **prototype Streamlit** yang siap
   dipakai staf akademik.

### Persiapan

**Sumber data:** [Students' Performance Dataset — Dicoding Academy](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)
(dataset asli: *Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository).
Berkas sudah disertakan pada folder ini sebagai `data.csv` (pemisah kolom: titik koma `;`).

**Setup environment:**

```bash
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Pasang seluruh dependensi
pip install -r requirements.txt

# 3. Jalankan notebook analisis (opsional — model sudah disertakan)
jupyter notebook notebook.ipynb
```

Versi pustaka yang digunakan tercantum pada `requirements.txt` (Python 3.13, pandas 2.2.3,
scikit-learn 1.6.1, Streamlit 1.64.0).

---

## Business Dashboard

Business dashboard dibangun menggunakan **Metabase** di atas basis data **SQLite** (`students.db`)
yang dihasilkan notebook. Seluruh kode kategorikal pada dataset asli sudah diterjemahkan menjadi
label yang dapat dibaca (`Nursing`, `Menunggak`, `Lajang`, …), sehingga tim akademik dapat membuat
pertanyaan sendiri di Metabase tanpa perlu kamus kode.

**Berkas:** `FIshiwak-dashboard.png` (tangkapan layar) dan `metabase.db.mv.db` (basis data Metabase).

### Isi dashboard

| Bagian | Kartu | Yang dijawab |
|---|---|---|
| **Ringkasan (KPI)** | Total Mahasiswa · Jumlah Dropout · Dropout Rate (%) · Mahasiswa Berisiko | Seberapa besar masalahnya hari ini |
| **Sebaran** | Status Akhir Mahasiswa · Dropout Rate per Program Studi | Di mana dropout paling terkonsentrasi |
| **Faktor finansial & demografis** | Status SPP · Beasiswa · Kelompok Usia · Gender & Waktu Kuliah | Kelompok mana yang paling rentan |
| **Faktor akademik** | Rata-rata Mata Kuliah Lulus per Status | Sejak kapan sinyal risiko mulai terlihat |
| **Hasil model** | Dropout Rate Aktual per Kategori Risiko · Daftar Prioritas Penanganan | Apakah skor model layak dipercaya, dan siapa yang harus ditemui lebih dulu |

Prinsip desain yang diterapkan agar dashboard mudah dibaca sekaligus jujur: satu warna aksen
(`#2a78d6`) untuk seluruh seri tunggal sehingga panjang batang — bukan warna — yang menyampaikan
besaran; label nilai ditampilkan langsung pada setiap batang; judul kartu menyatakan isi, bukan nama
kolom; dan setiap kartu memiliki deskripsi (ikon **i**) yang menjelaskan cara membacanya.

### Menjalankan dashboard di komputer lokal

Berkas basis data Metabase (`metabase.db.mv.db`) sudah disertakan lengkap dengan seluruh kartu dan
dashboard. Berkas `metabase.jar` (± 390 MB) **tidak disertakan** karena ukurannya; unduh dari
[metabase.com/start/oss/jar](https://www.metabase.com/start/oss/jar).

```bash
# Jalankan dari dalam folder submission ini (butuh Java 21)
java -jar metabase.jar
```

Metabase akan membaca `metabase.db.mv.db` yang ada di folder yang sama, lalu buka
<http://localhost:3000>.

| Kredensial | Nilai |
|---|---|
| Email | `hafiznugraha04@gmail.com` |
| Password | `JayaJaya2024!` |

> **Catatan:** koneksi basis data di dalam Metabase menunjuk ke `students.db` dengan path absolut
> `C:/Users/ASUS/Downloads/submission/submission_akhir/students.db`. Bila folder ini dipindahkan,
> perbarui path tersebut lewat **Settings → Admin → Databases → Jaya Jaya Institut**.

---

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning dibangun dengan **Streamlit** (`app.py`). Aplikasi memuat pipeline
terlatih dari `model/dropout_model.joblib` beserta metadata `model/model_metadata.json`, sehingga
seluruh logika preprocessing, daftar fitur, label kategori, dan ambang keputusan mengikuti apa yang
dihasilkan notebook — tidak ada yang disalin ulang secara manual.

```bash
# Dari dalam folder submission ini
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka di <http://localhost:8501> dengan tiga halaman:

1. **Prediksi Individu** — formulir data mahasiswa yang dikelompokkan dalam tiga tab (Profil &
   Administrasi, Semester 1, Semester 2). Keluarannya berupa probabilitas dropout, kategori risiko,
   tindakan yang disarankan, serta daftar indikator peringatan yang terdeteksi pada mahasiswa
   tersebut.
2. **Prediksi Massal (CSV)** — unggah data satu angkatan/prodi sekaligus, lengkap dengan template CSV
   yang dapat diunduh, ringkasan jumlah mahasiswa berisiko, dan hasil prediksi terurut dari yang
   paling genting untuk diunduh kembali.
3. **Tentang Model** — performa model, faktor paling berpengaruh, definisi kategori risiko, dan
   **batasan penggunaan** yang perlu diketahui pengguna.

### Cara membaca keluaran model

| Kategori risiko | Rentang skor | Tindakan yang disarankan |
|---|---|---|
| Rendah | 0 – 25% | Monitoring rutin |
| Sedang | 25 – 42% | Pantau nilai semester berjalan |
| Tinggi | 42 – 75% | Jadwalkan konseling akademik |
| Sangat Tinggi | 75 – 100% | Intervensi segera: konseling + tinjau keringanan biaya |

Mahasiswa ditandai **berisiko** bila skornya ≥ **0,42**. Ambang ini bukan nilai bawaan 0,50,
melainkan ditetapkan sebagai kebijakan yang mengejar **recall ≥ 85%**, karena biaya melewatkan satu
calon dropout jauh lebih besar daripada biaya satu sesi konseling tambahan.

---

## Conclusion

**1. Seberapa besar masalahnya?**
**32,1% mahasiswa (1.421 dari 4.424) berakhir dropout** dan hanya 49,9% yang lulus. Masalah ini nyata,
berskala besar, dan terkonsentrasi pada kelompok tertentu — bukan tersebar merata.

**2. Faktor apa yang paling berpengaruh?**

- **Performa akademik semester berjalan (paling dominan).** Mahasiswa yang kelak dropout hanya
  melulus **2,6 dari 5,8 mata kuliah** di semester 1 dan memburuk menjadi 1,9 di semester 2,
  dibanding 6,2 pada kelompok yang lulus. Beban studi mereka sama — yang berbeda adalah tingkat
  keberhasilan menuntaskannya.
- **Kondisi finansial.** Tunggakan SPP adalah sinyal paling tajam di seluruh dataset: **86,6%
  dropout** versus 24,7% pada yang lancar membayar. Sebaliknya, penerima beasiswa hanya dropout
  12,2% dibanding 38,7% pada yang bukan penerima.
- **Profil saat mendaftar.** Risiko naik seiring usia pendaftaran (21,2% pada ≤20 tahun menjadi
  51–58% pada kelompok di atas 24 tahun), lebih tinggi pada mahasiswa laki-laki (45,1% vs 25,1%),
  kelas malam, dan mahasiswa yang sudah berkeluarga. Antarprogram studi selisihnya mencapai hampir
  40 poin persen (Nursing 15,4% vs Equinculture 55,3%).
- Sebaliknya, variabel makroekonomi (`Unemployment_rate`, `GDP`, `Inflation_rate`) nyaris tidak
  berkorelasi. **Masalah dropout di sini bersifat internal dan dapat ditindaklanjuti institusi.**

**3. Bagaimana mendeteksi sedini mungkin?**
Model **Logistic Regression** yang terpilih — setelah dibandingkan dengan Random Forest dan Gradient
Boosting lewat 5-fold cross-validation — mencapai performa berikut pada 885 mahasiswa data uji yang
belum pernah dilihat:

| Metrik | Nilai | Artinya |
|---|---|---|
| **Recall** | **87,7%** | Sekitar 9 dari 10 calon dropout berhasil ditandai |
| **Precision** | 73,9% | Sekitar 3 dari 4 mahasiswa yang dipanggil memang benar-benar berisiko |
| **F1-score** | 0,802 | Keseimbangan keduanya |
| **ROC-AUC** | **0,930** | Kemampuan memeringkat risiko jauh di atas tebakan acak |
| Accuracy | 86,1% | — |

Yang lebih berguna daripada label ya/tidak adalah **peringkat risikonya**: kelompok "Sangat Tinggi"
terbukti dropout di angka **90,3%**, sedangkan kelompok "Rendah" hanya **5,0%**. Dengan pemeringkatan
seperti ini, tim kemahasiswaan dapat mendatangi lebih dulu sekitar seperempat mahasiswa yang paling
genting alih-alih menyisir seluruh 4.424 nama. Karena seluruh fitur yang dipakai sudah tersedia sejak
akhir semester pertama, peringatan dapat terbit **pada tahun pertama** — saat intervensi masih sempat
mengubah hasil.

**4. Bagaimana memonitornya secara mandiri?**
Dua produk siap pakai diserahkan: **dashboard Metabase** untuk memantau tren dropout per program
studi, status pembayaran, dan kelompok risiko; serta **prototype Streamlit** untuk memeriksa
mahasiswa satu per satu maupun secara massal lewat unggahan CSV.

### Rekomendasi Action Items

1. **Jadikan tunggakan SPP sebagai pemicu otomatis konseling, bukan sekadar urusan administrasi
   keuangan.** Dengan 86,6% mahasiswa penunggak berakhir dropout, setiap tunggakan yang lewat 30 hari
   seharusnya langsung memunculkan tiket ke bagian kemahasiswaan — bukan hanya surat tagihan.
   Tawarkan cicilan atau keringanan **sebelum** mahasiswa memutuskan berhenti.

2. **Perluas skema beasiswa/bantuan biaya ke mahasiswa berisiko tinggi.** Penerima beasiswa dropout
   tiga kali lebih jarang (12,2% vs 38,7%). Prioritaskan penerima baru dari irisan mahasiswa yang
   menunggak **dan** berprestasi akademik memadai — kelompok yang kehilangannya paling mungkin
   dicegah hanya dengan uang.

3. **Pasang peringatan dini akademik di akhir semester 1.** Mahasiswa yang melulus **kurang dari
   setengah** mata kuliah yang diambilnya wajib masuk program bimbingan: kelas remedial, penasihat
   akademik, atau pengurangan beban SKS. Indikator ini sudah tersedia di sistem akademik dan tidak
   menuntut data baru apa pun.

4. **Audit program studi yang berada di atas rata-rata institusi.** Equinculture (55,3%), Informatics
   Engineering (54,1%), dan Management kelas malam (50,7%) butuh peninjauan beban kurikulum, kualitas
   pengajaran, dan kesesuaian ekspektasi calon mahasiswa saat penerimaan.

5. **Rancang dukungan khusus mahasiswa non-tradisional.** Mahasiswa berusia di atas 23 tahun, kelas
   malam, dan yang sudah berkeluarga menanggung beban di luar kampus. Jadwal kuliah yang lebih lentur,
   kelas daring, serta layanan konseling di luar jam kerja menyasar tepat kelompok ini.

6. **Operasikan model sebagai proses rutin, bukan laporan sekali jadi.** Jalankan penilaian risiko
   setiap akhir semester, salurkan daftar mahasiswa "Risiko Tinggi" dan "Sangat Tinggi" ke wali
   akademik masing-masing, dan **catat hasil setiap intervensi**. Catatan itu menjadi bahan pelatihan
   ulang model di tahun berikutnya sekaligus bukti apakah programnya benar-benar berhasil.

7. **Pantau keadilan model sebelum dan sesudah dipakai.** Karena jenis kelamin dan usia ikut menjadi
   fitur, keluaran model harus diperiksa berkala agar bimbingan tidak berubah menjadi stigma.
   Perlakukan skor sebagai **daftar siapa yang perlu ditanyai kabarnya lebih dulu**, bukan sebagai
   vonis atas masa depan mahasiswa.

---

## Struktur Berkas

```
submission_akhir/
├── model/
│   ├── dropout_model.joblib        # pipeline terlatih (preprocessing + Logistic Regression)
│   └── model_metadata.json         # fitur, threshold, label kategori, metrik, kategori risiko
├── .streamlit/
│   └── config.toml                 # tema tampilan prototype
├── notebook.ipynb                  # analisis lengkap (sudah dijalankan, beserta seluruh output)
├── app.py                          # prototype sistem machine learning (Streamlit)
├── data.csv                        # dataset students' performance (4.424 baris, 37 kolom)
├── students.db                     # basis data SQLite sumber dashboard Metabase
├── students_dashboard.csv          # versi CSV dari data dashboard (label + skor risiko)
├── metabase.db.mv.db               # basis data Metabase berisi dashboard dan seluruh kartunya
├── FIshiwak-dashboard.png          # tangkapan layar business dashboard
├── requirements.txt                # daftar dependensi
└── README.md                       # dokumen ini
```
