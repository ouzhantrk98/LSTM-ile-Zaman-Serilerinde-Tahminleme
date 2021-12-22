## Problemin Tanımı
Borsada işlem gören hisse senetlerinin gelecekteki değerlerini tahminleyen bir makine öğrenmesi modeli oluşturulması amaçlanmaktadır. Bu işlemi gerçekleştirmek için LSTM algoritması kullanılmıştır. İnceleme yapılan borsa BIST100’ dür. Ele alınan problem zaman serisi içerdiği için LSTM kullanmayı uygun gördük.

### Araştırma
Makine öğrenmesi, diğer yapay zeka uygulamalarından farklı olarak, bir yandan insan zekasını taklit ederken, diğer yandan bizim yorumlayıp elle gireceğimiz kurallara ihtiyaç duymayan algoritmalar bütünüdür. Nasıl ki bir insan gördükleri ve duydukları ile kavramları kendi kendine öğreniyor ve birilerinin bu kuralları beynine işlemesine ihtiyaç duymuyorsa, makine öğrenmesi uygulamaları da benzer bir şekilde, kendisine sunulan veri kümelerini özümseyerek yapılması istenen görevi öğrenir.
Derin öğrenme , yapay sinir ağlarını temel alan makine öğrenmesinin bir alt kümesidir. Yapay sinir ağların yapısı birden çok giriş, çıkış ve gizli katmanlardan oluşuyorsa öğrenme süreci çok önemlidir. Her katman, giriş verilerini bir sonraki katmanın belirli bir tahmin görevi için kullanabileceği bilgilere dönüştüren birimler içerir. Bu yapı sayesinde bir makine, kendi veri işleme aracılığıyla bilgi alabilir.

![Aciklayici 1](https://i.hizliresim.com/4nxsxx0.jpg)

### Kullanılan Ortam, Yöntem ve Kütüphaneler
Ortam olarak Spyder idesi kullanıldı. Yöntem olarak LSTM algoritması kullanıldı. Kullanılan kütüphaneler şu şekildedir;

![Aciklayici 2](https://i.hizliresim.com/6ytc37q.jpg)

### Kullanılan Yöntemin Açıklanması
LSTM gözetimli(supervised) makine öğrenmesi tekniğidir. Yani eğitim verisi için girdi ve çıktıları vermek zorundayız. Hisse fiyatlarını içeren zaman serisi tek değişkenlidir. Yani elimizdeki tek çıktı hisse senedinin kapanış fiyatıdır. Bu tipte olan verinin eğitim veri setinde kullanımı iki farklı veri yapısıyla gerçekleştirilir. Birincisi many-to-many diğeri ise many-to-one’ dır.

#### Many-to-Many 
Geçmiş zamandaki X adet gündeki fiyatları kullanarak gelecekteki Y adet günü tahmin etmeyle ilgileniyoruz. Basit olması için geçmişteki 5 günün fiyatını gelecekteki 2 günü tahmin etmek için kullanalım. Yani birden fazla olan input’ a birden fazla sayıda output durumu var. Bu veri yapısı many-to-many olarak ifade edilir. Kırmızı pencere seri boyunca hareket ettikçe tek değişkenli zaman serisinden örnekler yaratılır. Her örnek 5 input ve 2 output’ a sahiptir. Her örnekteki input’ a time step denir ve her time step feature olarak adlandırılan bir numaraya sahiptir. Feature’ ların sayısı birden fazla olabilir. Örneğin «Adj. Close» ve «Open» fiyatlarını birlikte modellersek iki adet feature olmuş olur. Burada sadece «Adj. Close»’ u modelledik. Yani feature sayısı birdir. 

![Aciklayici 3](https://i.hizliresim.com/65jix2i.jpg)

#### Many-to-one
Burada yalnızca bir outputlu durum ele alınıyor. Buna çoktan bire (many-to-one) denir. 

![Aciklayici 4](https://i.hizliresim.com/rrorzue.jpg)

#### RNN Kullanamaz mıydık ?
Zaman serilerinde işlem yapmak için RNN modelini de kullanabiliriz. Peki biz niçin LSTM kullanıyoruz. RNN’ in optimize edicisi en iyi değerleri aramak için kayıp fonksiyonunun birinci dereceden türevini alır. Çünkü RNN özyinelemelidir. Birinci derecen türev alma işlemi bir sayıyı küçükken daha küçük yapacaktır, sonunda kaybolasıya dek. Buna gradient vanishing denir. Bu matematiksel işlem RNN geçmiş verileri tutması için iyi bir seçenek yapmaz. Biz bilgiyi çabucak yok etmeyen özyinelemeli bir yapıya ihtiyaç duyuyoruz. LSTM’ ye ihtiyaç duyulmasının nedeni budur. 
Optimizasyon sürecine ait detaylı bilgi için: Kayıp fonksiyonu, gerçek ve tahmin edilen değerler arasındaki hataları ölçen ölçüdür. Optimize edici, minimum hatayı takip etmek için nöronların ağırlıklarını değiştiren algoritmadır. Popüler bir optimize edici, Stokastik Gradyan İnişidir (SGD). “https://medium.com/analytics-vidhya/a-lecture-note-on-random-forest-gradient-boosting-and-regularization-834fc9a7fa52” linki SDG için detaylı açıklama sağlar.

### SONUÇ
Bu çalışmada modeli eğittikten sonra gelecekteki veriler için tahmin veren bir fonksiyon oluşturduk. Fonksiyon tahmin verileceği haftanın öncesindeki hafta için verileri alıyor tüm hafta boyu olası değerleri tahmin ediyor. Burada 31 mayısa dek olan verileri aldık. 31Mayıs-7Haziran fonksiyon tahminleme işlemi yapacak. 

![Aciklayici 5](https://i.hizliresim.com/7eipvla.jpg)
