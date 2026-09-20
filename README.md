# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

- **Nama:** Hafiz Nazwa Nugraha
- **ID Dicoding:** FIshiwak

---

## Streamlit link
[https://submissionakhir-bhxr7zowrzhvl6ef5ycqkz.streamlit.app/](https://submissionakhir-bhxr7zowrzhvl6ef5ycqkz.streamlit.app/)

## Business Understanding

**Jaya Jaya Institut** merupakan institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun
2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi,
terdapat banyak juga siswa yang **tidak menyelesaikan pendidikannya alias dropout**.

Jumlah dropout yang tinggi tentunya menjadi salah satu masalah yang besar untuk sebuah institusi
pendidikan, karena merugikan dari tiga sisi sekaligus:

1. **Reputasi & akreditasi** — rasio kelulusan adalah salah satu komponen penilaian akreditasi.
2. **Pendapatan** — setiap siswa yang keluar di tengah jalan berarti kehilangan uang kuliah untuk
   seluruh semester sisanya.
3. **Dampak sosial** — siswa kehilangan investasi waktu dan biaya yang sudah dikeluarkan, sering kali
   dengan sisa tunggakan.

Oleh karena itu, Jaya Jaya Institut ingin **mendeteksi secepat mungkin siswa yang mungkin akan
melakukan dropout** sehingga dapat diberi bimbingan khusus. Selain itu, pihak institusi juga meminta
dibuatkan **business dashboard** agar tim akademik mudah memahami data dan memonitor performa siswa
secara mandiri.

### Permasalahan Bisnis

1. **Seberapa besar angka dropout Jaya Jaya Institut saat ini dan berapa siswa yang terdampak?**
2. **Faktor apa saja yang paling memengaruhi keputusan siswa untuk dropout** — faktor akademik,
   finansial, demografis, atau jalur masuk?
3. **Bagaimana mendeteksi siswa berisiko dropout sedini mungkin**, idealnya pada akhir semester
   pertama, agar bimbingan khusus masih sempat diberikan?
4. **Bagaimana tim akademik dapat memonitor performa siswa secara mandiri**, tanpa harus meminta
   analisis ulang setiap kali dibutuhkan?

### Cakupan Proyek

1. **Data Understanding** — memahami struktur, kualitas, dan distribusi dataset performa siswa
   (4.424 baris, 37 kolom).
2. **Exploratory Data Analysis** — menggali faktor-faktor yang berhubungan dengan dropout melalui
   visualisasi data.
3. **Data Preparation** — memisahkan data berlabel dari data prediksi, membentuk target biner,
   menyeleksi fitur, membagi data latih/uji, dan membangun pipeline preprocessing
   (`ColumnTransformer` + `Pipeline`).
4. **Modeling** — membandingkan tiga algoritma klasifikasi, melakukan *hyperparameter tuning*, dan
   menetapkan *decision threshold* berdasarkan pertimbangan biaya bisnis.
5. **Evaluation** — mengukur performa dengan metrik yang relevan untuk deteksi dini (recall, F1,
   ROC-AUC) dan menafsirkan fitur yang paling berpengaruh.
6. **Deployment** — membangun **business dashboard Metabase** dan **prototype Streamlit** yang siap
   dipakai staf akademik serta dapat diakses secara *online*.

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

## Data Preparation: Memisahkan Data Berlabel dan Data Prediksi

Kolom `Status` pada dataset memiliki tiga nilai, tetapi hanya dua di antaranya yang merupakan
**hasil akhir studi**. Perbedaan ini menentukan seluruh alur pemodelan:

| Status | Jumlah | Peran dalam proyek | Target `is_dropout` |
|---|---|---|---|
| `Dropout` | 1.421 | data latih & uji | **1** |
| `Graduate` | 2.209 | data latih & uji | **0** |
| `Enrolled` | 794 | **data prediksi** — tidak ikut dilatih | *tidak ada label* |

Siswa berstatus `Enrolled` **tidak dilibatkan dalam proses training**. Statusnya bukan berarti
"tidak dropout", melainkan *"belum diketahui hasilnya"* — bisa saja tahun depan mereka lulus, bisa
juga keluar. Menyamakan encoding `Enrolled` dengan `Graduate` menjadi 0 akan membuat target menjadi
ambigu dan menurunkan validitas model.

Karena itu, model dilatih hanya pada **3.630 siswa berlabel** dengan target biner
**1 = Dropout, 0 = Graduate**, sementara **794 siswa `Enrolled` dipisahkan** untuk diberi skor risiko
sebagai data prediksi di masa depan — dan justru merekalah sasaran utama sistem peringatan dini ini,
karena hasil akhir mereka masih bisa diubah.

Seluruh analisis faktor pada EDA juga dihitung hanya pada 3.630 siswa berlabel tersebut, agar
angkanya konsisten dengan definisi target model. Di antara mereka, **39,2% berakhir dropout**.

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
| **Ringkasan (KPI)** | Total Mahasiswa · Jumlah Dropout · Dropout Rate (%) · **Mahasiswa Aktif Berisiko** | Seberapa besar masalahnya, dan berapa siswa aktif yang perlu ditangani sekarang |
| **Sebaran** | Status Akhir Mahasiswa · Dropout Rate per Program Studi | Di mana dropout paling terkonsentrasi |
| **Faktor finansial & demografis** | Status SPP · Beasiswa · Kelompok Usia · Gender & Waktu Kuliah | Kelompok mana yang paling rentan |
| **Faktor akademik** | Rata-rata Mata Kuliah Lulus per Status | Sejak kapan sinyal risiko mulai terlihat |
| **Hasil model** | Validasi Model: Dropout Rate Aktual per Kategori Risiko · Sebaran Kategori Risiko Mahasiswa Aktif · Daftar Prioritas Penanganan | Apakah skor model layak dipercaya, dan siapa yang harus ditemui lebih dulu |

Dua hal penting dalam pembacaan dashboard:

- Seluruh kartu **dropout rate** dihitung hanya pada siswa berlabel (`Dropout` + `Graduate`) — siswa
  `Enrolled` dikecualikan karena hasil akhirnya belum diketahui. Catatan ini juga tertulis pada
  deskripsi setiap kartu (ikon **i**).
- Kartu **Validasi Model** menguji kategori risiko pada siswa yang hasilnya sudah diketahui,
  sedangkan kartu **Sebaran Kategori Risiko Mahasiswa Aktif** dan **Daftar Prioritas Penanganan**
  menerapkan kategori yang sama pada 794 siswa yang masih kuliah.

Prinsip desain yang diterapkan agar dashboard mudah dibaca sekaligus jujur: satu warna aksen
(`#2a78d6`) untuk seluruh seri tunggal sehingga panjang batang — bukan warna — yang menyampaikan
besaran; label nilai ditampilkan langsung pada setiap batang; judul kartu menyatakan isi, bukan nama
kolom; dan setiap kartu memiliki deskripsi yang menjelaskan cara membacanya.

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

Prototype sistem machine learning dibangun dengan **Streamlit** (`app.py`) dan sudah di-*deploy* ke
**Streamlit Community Cloud** sehingga dapat diakses tanpa instalasi apa pun:

**🔗 <https://submissionakhir-bhxr7zowrzhvl6ef5ycqkz.streamlit.app/>**

Aplikasi memuat pipeline terlatih dari `model/dropout_model.joblib` beserta metadata
`model/model_metadata.json`, sehingga seluruh logika preprocessing, daftar fitur, label kategori, dan
ambang keputusan mengikuti apa yang dihasilkan notebook — tidak ada yang disalin ulang secara manual.

### Menjalankan secara lokal

```bash
# Dari dalam folder submission ini
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka di <http://localhost:8501> dengan tiga halaman:

1. **Prediksi Individu** — formulir data siswa yang dikelompokkan dalam tiga tab (Profil &
   Administrasi, Semester 1, Semester 2). Keluarannya berupa probabilitas dropout, kategori risiko,
   tindakan yang disarankan, serta daftar indikator peringatan yang terdeteksi pada siswa tersebut.
2. **Prediksi Massal (CSV)** — unggah data satu angkatan/prodi sekaligus, lengkap dengan template CSV
   yang dapat diunduh, ringkasan jumlah siswa berisiko, dan hasil prediksi terurut dari yang paling
   genting untuk diunduh kembali.
3. **Tentang Model** — performa model, faktor paling berpengaruh, definisi kategori risiko, dan
   **batasan penggunaan** yang perlu diketahui pengguna.

### Langkah deploy ke Streamlit Community Cloud

Bila perlu men-*deploy* ulang (misalnya setelah model dilatih ulang):

1. **Unggah proyek ke GitHub** — minimal berisi `app.py`, folder `model/`, `requirements.txt`, dan
   `.streamlit/config.toml`. Berkas `metabase.db.mv.db` tidak perlu ikut karena tidak dipakai aplikasi.
2. **Login ke [share.streamlit.io](https://share.streamlit.io)** menggunakan akun GitHub.
3. **Integrasikan repositori** — pilih **New app**, tentukan repo, *branch*, dan *main file path*
   (`app.py`), lalu klik **Deploy**.
4. Setiap `git push` berikutnya akan otomatis memperbarui aplikasi yang sudah tayang.

### Cara membaca keluaran model

| Kategori risiko | Rentang skor | Tindakan yang disarankan |
|---|---|---|
| Rendah | 0 – 30% | Monitoring rutin |
| Sedang | 30 – 51% | Pantau nilai semester berjalan |
| Tinggi | 51 – 75% | Jadwalkan konseling akademik |
| Sangat Tinggi | 75 – 100% | Intervensi segera: konseling + tinjau keringanan biaya |

Siswa ditandai **berisiko** bila skornya ≥ **0,51**. Ambang ini bukan nilai bawaan 0,50, melainkan
ditetapkan sebagai kebijakan yang menjamin **recall ≥ 85%**, karena biaya melewatkan satu calon
dropout jauh lebih besar daripada biaya satu sesi konseling tambahan.

---

## Conclusion

**1. Seberapa besar masalahnya?**
Dari 4.424 siswa, **1.421 (32,1%) berakhir dropout**, 2.209 lulus, dan 794 sisanya masih aktif
kuliah. Bila dihitung hanya di antara siswa yang studinya sudah tuntas, **39,2% berakhir dropout** —
jauh di atas batas yang umumnya dianggap wajar untuk pendidikan tinggi (10–15%).

**2. Faktor apa yang paling berpengaruh?**

- **Performa akademik semester berjalan (paling dominan).** Siswa yang berakhir dropout rata-rata
  hanya melulus **2,55 dari 5,82 mata kuliah** di semester 1 dan memburuk menjadi 1,94 di semester 2,
  dibanding 6,23 pada siswa yang lulus. Beban studi keduanya hampir sama — yang berbeda adalah
  tingkat keberhasilan menuntaskannya.
- **Kondisi finansial.** Tunggakan SPP adalah sinyal paling tajam di seluruh dataset: **94,0%
  dropout** versus 30,7% pada siswa yang lancar membayar. Sebaliknya, penerima beasiswa hanya dropout
  **13,8%** dibanding 48,4% pada yang bukan penerima.
- **Profil saat mendaftar.** Risiko naik seiring usia pendaftaran (26,1% pada ≤20 tahun menjadi
  60–66% pada kelompok di atas 23 tahun), lebih tinggi pada siswa laki-laki (56,1% vs 30,2%), kelas
  malam (50,7% vs 37,7%), dan siswa yang sudah berkeluarga (55–56% vs 37,0%). Antarprogram studi,
  selisihnya mencapai hampir 70 poin persen (Nursing 17,7% vs Informatics Engineering 86,8%).
- Sebaliknya, variabel makroekonomi (`Unemployment_rate`, `GDP`, `Inflation_rate`) nyaris tidak
  berkorelasi. **Masalah dropout di sini bersifat internal dan dapat ditindaklanjuti institusi.**

**3. Bagaimana mendeteksi sedini mungkin?**
Model **Logistic Regression** — terpilih setelah dibandingkan dengan Random Forest dan Gradient
Boosting lewat 5-fold cross-validation — dilatih hanya pada siswa berlabel (1 = Dropout,
0 = Graduate), lalu diuji pada **726 siswa** yang belum pernah dilihatnya:

| Metrik | Nilai | Artinya |
|---|---|---|
| **Recall** | **91,9%** | Lebih dari 9 dari 10 calon dropout berhasil ditandai |
| **Precision** | **89,1%** | Hampir 9 dari 10 siswa yang dipanggil memang benar-benar berisiko |
| **F1-score** | **0,905** | Keseimbangan keduanya |
| **ROC-AUC** | **0,972** | Kemampuan memeringkat risiko sangat baik |
| Accuracy | 92,4% | — |

Dengan target yang tidak lagi ambigu, seluruh metrik naik tajam dibanding pendekatan yang
mencampurkan `Enrolled` ke dalam kelas negatif.

Yang lebih berguna daripada label ya/tidak adalah **peringkat risikonya**. Pada siswa berlabel,
kategori "Sangat Tinggi" terbukti dropout **97,7%** sedangkan kategori "Rendah" hanya **6,0%**.
Peringkat yang sama kemudian diterapkan pada **794 siswa `Enrolled`**, dan hasilnya **433 siswa aktif
masuk kategori Tinggi/Sangat Tinggi** — inilah daftar kerja konkret tim bimbingan akademik tahun ini.
Karena seluruh fitur yang dipakai sudah tersedia sejak akhir semester pertama, peringatan dapat
terbit **pada tahun pertama**, saat bimbingan masih sempat mengubah hasil.

**4. Bagaimana memonitornya secara mandiri?**
Dua produk siap pakai diserahkan: **dashboard Metabase** untuk memantau tren dropout per program
studi, status pembayaran, dan kelompok risiko; serta **prototype Streamlit** yang sudah tayang di
Streamlit Community Cloud untuk memeriksa siswa satu per satu maupun secara massal lewat unggahan CSV.

### Rekomendasi Action Items

1. **Jadikan tunggakan SPP sebagai pemicu otomatis konseling, bukan sekadar urusan administrasi
   keuangan.** Dengan 94,0% siswa penunggak berakhir dropout, setiap tunggakan yang lewat 30 hari
   seharusnya langsung memunculkan tiket ke bagian kemahasiswaan — bukan hanya surat tagihan.
   Tawarkan cicilan atau keringanan **sebelum** siswa memutuskan berhenti.

2. **Perluas skema beasiswa/bantuan biaya ke siswa berisiko tinggi.** Penerima beasiswa dropout tiga
   kali lebih jarang (13,8% vs 48,4%). Prioritaskan penerima baru dari irisan siswa yang menunggak
   **dan** berprestasi akademik memadai — kelompok yang kehilangannya paling mungkin dicegah hanya
   dengan uang.

3. **Pasang peringatan dini akademik di akhir semester 1.** Siswa yang melulus **kurang dari
   setengah** mata kuliah yang diambilnya wajib masuk program bimbingan: kelas remedial, penasihat
   akademik, atau pengurangan beban SKS. Indikator ini sudah tersedia di sistem akademik dan tidak
   menuntut data baru apa pun.

4. **Tangani lebih dulu 433 siswa aktif berkategori Tinggi/Sangat Tinggi.** Daftarnya sudah tersedia
   pada dashboard (kartu *Daftar Prioritas Penanganan*) dan dapat diunduh dari prototype Streamlit.
   Mendatangi 433 nama jauh lebih realistis daripada menyisir seluruh 4.424 siswa.

5. **Audit program studi yang berada di atas rata-rata institusi.** Informatics Engineering,
   Equinculture, dan Management kelas malam butuh peninjauan beban kurikulum, kualitas pengajaran,
   dan kesesuaian ekspektasi calon siswa saat penerimaan.

6. **Rancang dukungan khusus siswa non-tradisional.** Siswa berusia di atas 23 tahun, kelas malam,
   dan yang sudah berkeluarga menanggung beban di luar kampus. Jadwal kuliah yang lebih lentur, kelas
   daring, serta layanan konseling di luar jam kerja menyasar tepat kelompok ini.

7. **Operasikan model sebagai proses rutin dan pantau keadilannya.** Jalankan penilaian risiko setiap
   akhir semester atas seluruh siswa aktif, salurkan daftarnya ke wali akademik, dan **catat hasil
   setiap intervensi** sebagai bahan pelatihan ulang model tahun berikutnya. Karena jenis kelamin dan
   usia ikut menjadi fitur, keluaran model perlu diperiksa berkala agar bimbingan tidak berubah
   menjadi stigma — perlakukan skor sebagai **daftar siapa yang perlu ditanyai kabarnya lebih dulu**,
   bukan vonis atas masa depan siswa.

---

## Struktur Berkas

```
submission/
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
