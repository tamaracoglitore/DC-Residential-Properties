# DC-Residential-Properties
Prediction on DC Residential Properties Pricing with regression models and properties segmentation using clustering models.
# OmegaTeam_JC_DS_LS_02_FinalProject
# DC Residential Properties

Team members:   
1. Faykel Nicandro Hattu
2. Muhammad Rafi Amiruddin  
3. Tamara Coglitore

### Source
[(Dataset)](https://www.kaggle.com/datasets/christophercorrea/dc-residential-properties?select=DC_Properties.csv)

### Outline

* Business Problem
* Data Understanding
* Exploratory Data Analysis & Data Preprocessing
* Data Analysis
* Modeling
* Conclusion and Recommendation

## Business Problem

**Context**<br>  
   
Washington, D.C. adalah Ibu Kota dari Amerika Serikat dengan jumlah penduduk 689.545 dan memiliki jumlah tempat tinggal sebanyak 350.364 unit berdasarkan sensus tahun 2020 [(opendataDC)](https://opendata.dc.gov/). Sebagai kota pusat pemerintahan di Amerika Serikat, tidak dapat dipungkiri bila biaya hidup disana cukup tinggi dimana salah satunya adalah harga properti. Pada tahun 2017, rata-rata harga rumah pribadi (*single-family home*) di kota tersebut berada di kisaran $647.000, sedangkan harga rata-rata rumah di Amerika Serikat pada tahun yang sama berada di kisaran $399.700 [(FRED)](https://fred.stlouisfed.org/series/ASPUS). Meskipun hal ini menunjukkan tingginya harga rumah di Washington, D.C, wilayah tersebut secara konsisten menjadi salah satu pasar properti yang paling banyak diminati di Amerika Serikat [(Atlaslane)](https://www.atlaslane.com/post/investment-property-washington-dc-best-areas#:~:text=Washington%2C%20DC%20is%20consistently%20ranked,month%20when%20rent%20is%20collected.). Oleh karena itu, banyak perusahaan *real estate* seperti [The One Street](https://www.onestreet.one/), [GreenLine](https://www.greenlinere.com/), [Fulcrum Property](https://www.fulcrumproperty.com/), dan perusahaan lainnya yang ingin memanfaatkan peluang ini sehingga membuat persaingan bisnis di bidang *residential real estate* menjadi sangat kompetitif. Untuk dapat bersaing, Residential Real Estate Company perlu memberikan layanan yang optimal antara lain dengan memberikan penawaran harga terbaik dan sesuai dengan segmentasi pasar. 

Harga properti di Amerika cukup berdampak pada stabilitas ekonomi negara [(TheBalanceMoney)](https://www.thebalancemoney.com/how-does-real-estate-affect-the-u-s-economy-3306018), yang menyebabkan pemerintah juga turut menentukan dan mengatur kebijakan yang berdampak pada harga properti [(Fannie and Freddie)](https://www.chicagobooth.edu/review/should-government-intervene-housing-market). Sehingga untuk memberikan penawaran harga, kita perlu mengikuti kebijakan dan kualifikasi dari pemerintah. 
   
Selain mengikuti kebijakan pemerintah, survey terhadap properti residential juga perlu dilakukan untuk memperkirakan harga yang tepat. Namun, seperti yang dikutip dilaman Department of Consumer and Regulatory Affairs Washington, D.C., proses survey secara konvensional memakan biaya yang cukup besar [(Angi)](https://www.angi.com/articles/how-much-does-land-survey-cost.htm) serta memakan waktu yang cukup lama. Selain itu, metode konvensional seperti survey berpotensi terjadinya *overpriced* dan *underpriced*.
   
Perusahaan juga perlu mengetahui segmentasi properti residential untuk diberikan ke target market tertentu sebagai penunjang Department Marketing perusahaan. Strategi marketing dengan cara menawarkan rumah secara acak tanpa memberikan batasan tertentu dapat memberikan hasil yang kurang efektif. 
<br>

**Problem Statement**<br>
  
Survey yang dilakukan untuk memperkirakan harga properti residential cukup memakan waktu dan biaya. Selain itu, subjektifitas dalam penentuan harga jual oleh surveyor dapat menyebabkan terjadinya *overpriced* dan *underpriced*. *Overpriced* maupun *underpriced* dapat merugikan perusahaan. Kasus *overpriced* dapat menyebabkan properti tersebut sulit terjual karena harga yang kurang bersaing, sedangkan pada kasus *underpriced* dapat menyebabkan *profit loss* [(CPG)](https://www.durangohomesforsale.com/blog/what-to-know-about-overpricing-or-underpricing-your-home/). Disamping itu, budget marketing akan menjadi kurang efisien jika perusahaan tidak mampu menentukan segmentasi properti dengan tepat. 
<br>

**Goals**<br>  
  
Dengan adanya permasalahan di atas, maka tujuan analisa yang dilakukan adalah sebagai berikut :
1. Memberikan prediksi harga properti residential dengan meminimalisir error sehingga tidak *overpriced* dan *underpriced*, serta mengurangi biaya dan waktu pengerjaan survey properti
2. Melakukan segmentasi properti untuk meningkatkan produktivitas Department Marketing

Dengan menggunakan pemodelan machine learning, permasalahan di atas bisa diselesaikan dengan cara yang lebih efisien dibandingkan dengan cara konvensional sehingga kinerja perusahaan meningkat.
<br>

**Analytic Approach**<br>
   
Pendekatan analitik yang dilakukan adalah dengan menganalisa data untuk dapat menemukan pola dari fitur-fitur yang ada. Akan dilakukan pembuatan, evaluasi, dan implementasi model machine learning regresi dan clustering sebagai *tool* yang dapat digunakan untuk memprediksi harga dan segmentasi properti residential.
<br>

**Metric**<br>
    
Evaluation metrics digunakan untuk mengukur performa atau kualitas model machine learning. Dengan menggunakan 2 model yang berbeda, maka metrik yang akan digunakan juga berbeda. 

Pada model machine learning regresi terdapat berbagai macam metrik yang digunakan, seperti MAE, MAPE, dan R-squared. MAE merupakan rata-rata nilai absolut dari error dan MAPE merupakan rata-rata persentase error. MAE dan MAPE merupakan metrik yang tidak sensitif terhadap outlier. Semakin rendah nilai MAE dan MAPE maka semakin akurat model dalam prediksi harga properti residential. Selain itu R-squared merupakan koefisien determinasi yang mengindikasi besarnya kombinasi variabel independen mempengaruhi variabel dependen. Nilai R-squared berkisar diantara 0 dan 1, dimana semakin tinggi nilai R-square maka semakin baik model regresi. R-squared hanya bisa digunakan pada model linear.

Pada model machine learning clustering, metrik yang dapat digunakan adalah Silhouette Score. Metrik digunakan untuk menentukan jumlah clustering yang optimal. Jumlah cluster yang ideal dengan metrik silhouette dilihat dari nilai koefisien dengan rentang -1 sampai 1. Nilai 1 menunjukkan nilai terbaik dimana data sangat compact pada clusternya, nilai 0 menunjukkan cluster saling tumpang tindih, dan nilai terburuk adalah -1. Silhouette score digunakan karena memiliki pendekatan yang lebih akurat dan *reliable* pada penentuan jumlah cluster.
<br>

### Conclusion

#### Price Prediction

* **Meminimalisir Error**<br>  
  
Dari hasil di atas disimpulkan bahwa prediksi dengan model XGBoost yang sudah dibuat akan menghasilkan nilai error dengan rata-rata error +- USD 65784.21 atau setara dengan 11.36% harga. 
  
Dengan hal tersebut, perusahaan diharapkan bisa menentukan harga sesuai dengan fitur masing-masing properti dengan error minimal dan mengurangi potensi overpriced atau underpriced. Dengan contoh kasus **underpriced** yang ditemukan sebelumnya pada tahap data understanding (Harga underpriced pada kolom unqualified jika dibandingkan dengan kolom qualified yang memiliki fitur yang hampir identik) dan dibandingkan dengan hasil pemodelan, maka perhitungan jumlah kerugian yang bisa dihindari sebagai berikut. 

* Sebelum pemodelan => Harga Qualified - Harga Unqualified = USD 846000 - USD 339500 = USD 506500  
* Setelah pemodelan => MAE = USD 65784.21

Maka kerugian yang bisa dihindari adalah => USD 506500 - USD 65784.21 = USD 440715.79
<br>

* **Memotong Budget Survey**<br>
  
Selain itu, perusahaan juga mampu memotong pengeluaran untuk pemakaian jasa surveyor properti. Harga survey properti untuk [cicilan](https://www.angi.com/articles/how-much-does-land-survey-cost.htm) berkisar antara US$500 per surveynya. Maka dari itu, harga tersebut akan digunakan sebagai perhitungan.

Pengeluaran biaya untuk jasa surveyor properti dengan asumsi memiliki 1 orang surveyor dengan ketentuan 40jam kerja per minggu:  

Harga rata-rata 1x survey = US$500  
Waktu rata-rata 1x survey = 3 jam + 1 jam untuk laporan, perjalanan, dan keperluan lain   
Survey/minggu = 40jam/minggu : 4 jam = 10x survey  
Survey/tahun = 10 x 52 minggu = 520 survey  
Biaya jasa surveyor per tahun = 520 x 500 = US$260.000

Dengan bantuan Machine Learning, bila diasumsikan jasa surveyor tidak digunakan sama sekali, maka perusahaan bisa menghemat sebesar US$260.000/surveyor setiap tahunnya.
<br>

* **Fitur-fitur penting dan limitasi**<br>
  
Features dari dataset yang paling berpengaruh terhadap harga properti adalah GRADE, diikuti dengan kolom GBA dan SALEYEAR. Kolom GRADE memiliki pengaruh yang paling signifikan terhadap PRICE yaitu semakin tinggi GRADE makan harga cenderung semakin tinggi juga, ditunjukkan dengan median tiap GRADE yang meningkat sesuai order. Hal yang sama terdapat pada GBA dan SALEYEAR. Semakin luas nilai GBA maka harga cenderung naik, dan semakin terkini tahun SALEYEAR maka harga cenderung naik.

Model hanya bisa digunakan pada data properti Residential. Selain itu diketahui, model memiliki error yang lebih kecil dengan harga di bawah USD 3Jt-4Jt. Di atas harga tersebut model masih mengalami error yang cukup signifikan yang disebabkan karena sedikitnya data dengan harga tersebut.
<br>

#### Property Segmentation

* **Hasil Clustering**<br>  
  
Dari hasil pemodelan clustering didapatkan model terbaik dengan menggunakan Agglomerative Ward dengan silhouette score 0.601 pada jumlah cluster 3. Pada dataset diketahui bahwa terdapat 3 segmentasi properti yaitu, **Lower**, **Middle**, dan **Upper** dengan `PRICE` sebagai pembeda segmentasi. Dimana terdapat kecenderungan beberapa fitur pada masing-masing segmentasi sebagai berikut:
1. **LOWER** : tidak memiliki FIREPLACE, Luas Bangunan berada di rentang **1100-1650** sqft, kondisi properti **average**.
1. **MIDDLE** : telah melakukan proses remodeling, memiliki Half Bathroom, Luas Bangunan berada di rentang **1400-1950** sqft, kondisi properti **good**.
1. **UPPER** : terdapat FIREPLACE, telah melakukan proses remodeling, memiliki Half Bathroom, Luas Bangunan lebih dari **1800** sqft, kondisi properti **good** & **very good**, berlokasi di kuadran ***Northwest*** (Ward 2 & 3).
<br>

* **Business Approach**<br>
  
Dengan adanya clustering di atas, maka perusahaan mampu menurunkan leadtime hingga 67%. Angka itu didapatkan dari mengurangi 2 kemungkinan dari total 3 kemungkinan yang ada dalam menawarkan rumah kepada customer. Sehingga dengan spare waktu 67% tersebut dapat meningkatkan efisiensi kerja Departemen Marketing, seperti efisiensi workload dan efisiensi manpower.
<br>

* **Limitasi**<br>  
  
Model sangat sensitif terhadap data outlier, sehingga data dengan outlier akan dihapus. Selain itu karena model memiliki sifat kompleks, model membutuhkan waktu dan *space device* yang mumpuni untuk menangani **jumlah data yang terlalu besar**.
<br>

### Recommendation

Model:
- Membuat pemodelan untuk properti Condominium untuk menjangkau prediksi harga seluruh properti Washington DC.
- Memastikan bahwa tidak terjadi kesalahan-kesalahan input data seperti tahun, nomor ataupun fitur lainnya pada saat pengumpulan dataset.
- Menambahkan fitur lain seperti jarak ke sekolah atau pusat perbelanjaan untuk menjelaskan apakah properti tersebut berada di lokasi strategis atau tidak.
- Menambah jumlah data, khususnya pada range harga di atas US$2 juta untuk membuat akurasi model prediksi yang lebih baik.
- Melakukan improvement pada parameter tuning model price prediction dengan tuning di atas sebagai benchmark.
- Melakukan pemodelan price prediction ulang setelah dilakukan feature importance sehingga mendapatkan model yang lebih baik dengan fitur-fitur lebih penting yang berpengaruh pada prediksi harga properti untuk mengurangi kompleksitas model dan efisiensi waktu pemodelan.
- Mencoba menggunakan pemodelan DBScan pada Properti Segmentation dimana model lebih robust terhadap data outlier dan bisa digunakan pada dataset yang lebih besar.

Business:
- Menggunakan model machine learning price prediction yang telah dibuat sebagai solusi untuk menentukan harga properti dengan error minimum serta menghindari kasus overprice dan underprice.
- Memberikan rekomendasi berdasarkan tren bulan yang dapat membantu Departemen Marketing untuk menyarankan waktu beli terbaik kepada customer.
- Menyediakan informasi segmentasi properti untuk Departemen Marketing yang disesuaikan dengan target market, berdasarkan karakteristik properti yang telah dijabarkan pada cluster *Lower*, *Middle*, dan *Upper*.
- Menyarankan Department Marketing untuk membuat iklan berdasarkan kelompok rumah, **Lower** -> Type 1, **Middle** -> Type 2, **Upper** -> Type 3 pada space/platform iklan dengan deskripsi fitur dan harga sesuai segmentasi untuk lebih menarik perhatian customer.
- Membuat sebuah aplikasi dengan model yang telah dibuat untuk mempermudah prediksi dan segmentasi dan menunjang perusahaan Residential Real Estate Company.

### Dashboard

[Dashboard Tableau](https://public.tableau.com/app/profile/tamara.coglitore/viz/pricetrend/Dashboard2?publish=yes)
![alt text](https://github.com/tamaracoglitore/DC-Residential-Properties/blob/main/Dashboard%202%20-%201.png)
