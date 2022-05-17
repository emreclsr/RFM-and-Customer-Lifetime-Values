import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###############################################################
# Görev 1: Veriyi Anlama ve Hazırlama
###############################################################


# Adım 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("week3/Hw3/FLO_RFM_Analizi/flo_data_20K.csv")
df = df_.copy()


# Adım 2: Veri setinde a. İlk 10 gözlem, b. Değişken isimleri, c. Betimsel istatistik, d. Boş değer,
# e. Değişken tipleri, incelemesi yapınız.
df.head(10)  # a.
df.columns  # b.
df.describe().T  # c.
df.isnull().sum()  # d.
df.dtypes  # e.


# Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["order_num_total_ever_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever_omnichannel"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes  # Değişken tiplerini incelemeye yarar.
date_cols = [col for col in df.columns if "date" in col]  # Tarih ifade eden değişkenlerin listesini oluştur.
df[date_cols] = df[date_cols].apply(pd.to_datetime)  # Tarih ifade eden değişkenlerin tipini date'e çevir.

# Adım 5: Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların
# dağılımına bakınız.

channel_result = df.groupby("order_channel").agg({"master_id": lambda x: x.nunique(),
                                                  "order_num_total_ever_omnichannel": lambda x: x.mean(),
                                                  "customer_value_total_ever_omnichannel": lambda x: x.mean()})

# Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.


df.sort_values(by="customer_value_total_ever_omnichannel", ascending=False)[["master_id"]].head(10)

# Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız

df.sort_values(by="order_num_total_ever_omnichannel", ascending=False)[["master_id"]].head(10)

# Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız


def data_prepare(dataframe):
    """
    Veri ön hazırlık sürecini fonksiyonlaştırır.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        İşlenecek dataframe.

    Returns
    -------
    dataframe: pandas.DataFrame
        İşlenmiş dataframe.
    channel_result: pandas.DataFrame
        Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamalarının dağılımı
    top10_customer_by_value: pandas.DataFrame
        En fazla kazancı getiren ilk 10 müşteri
    top10_customer_by_ordernum: pandas.DataFrame
        En fazla siparişi veren ilk 10 müşteri

    """

    # Veri seti gözlemleri
    print("\n" + " Ham veri seti gözlemleri ".center(92, "_") + "\n")
    print("\n" + " İlk 10 satır ".center(92, "_") + "\n")
    print(dataframe.head(10))
    print("\n" + " Kolon isimleri ".center(92, "_") + "\n")
    print(dataframe.columns)
    print("\n" + " Betimsel istatistik ".center(92, "_") + "\n")
    print(dataframe.describe().T)
    print("\n" + " Veriseti null verilerin her bir kolondaki toplamı ".center(92, "_") + "\n")
    print(dataframe.isnull().sum())
    print("\n" + " Her bir kolona ait veri tipi ".center(92, "_") + "\n")
    print(dataframe.dtypes)
    print("\n" + "_".center(92, "_"))

    # Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
    # Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturma.
    dataframe["order_num_total_ever_omnichannel"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total_ever_omnichannel"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    # Tarih ifade eden değişkenlerin tipini date'e çevirme.
    date_cols = [col for col in dataframe.columns if "date" in col]
    dataframe[date_cols] = dataframe[date_cols].apply(pd.to_datetime)

    # Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımı

    channel_result = dataframe.groupby("order_channel").agg({"master_id": lambda x: x.nunique(),
                                                             "order_num_total_ever_omnichannel": lambda x: x.mean(),
                                                             "customer_value_total_ever_omnichannel": lambda x: x.mean()})

    # En fazla kazancı getiren ilk 10 müşteri

    top10_customer_by_value = dataframe.sort_values(by="customer_value_total_ever_omnichannel", ascending=False)[["master_id"]].head(10)


    # En fazla siparişi veren ilk 10 müşteri

    top10_customer_by_ordernum = dataframe.sort_values(by="order_num_total_ever_omnichannel", ascending=False)[["master_id"]].head(10)

    return dataframe, channel_result, top10_customer_by_value, top10_customer_by_ordernum


df, channel_result, top10_customer_by_value, top10_customer_by_ordernum = data_prepare(df)


###############################################################
# Görev 2: RFM Metriklerinin Hesaplanması
###############################################################

"""
Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.

Recency: Müşteri son alışverişini kaç gün önce yaptı?
Frequency: Müşteri kaç kere alışveriş yaptı?
Monetary: Müşteri ne kadar para harcadı?
"""

# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.

date_max = df["last_order_date"].max()
calculation_date = date_max + pd.DateOffset(days=2)

# Bu veri özelinde Frequency ve Monetary değerlerinin toplamını almaya gerek yok ama genelleştirmek için eklendi.
# Bu dataframe özelinde master_id her bir satırda eşsiz olduğundan groupby'a almaya gerek yoktur. Fakat daha genel
# bir kullanım için bu özellikten yararlanılmıştır.

rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (calculation_date - x.max()).days,  # Recency
                                   "order_num_total_ever_omnichannel": lambda x: x.sum(),           # Frequency
                                   "customer_value_total_ever_omnichannel": lambda x: x.sum()})     # Monetary

rfm.describe().T

# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.

rfm.columns = ["recency", "frequency", "monetary"]


###############################################################
# Görev 3: RF Skorunun Hesaplanması
###############################################################

# RFM skorların pd.qcut ile kategorize edilmesi
recency_score = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
frequency_score = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
monetary_score = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# RF skorların dataframe yeni kolon olarak eklenmesi
rfm["RF_SCORE"] = (recency_score.astype(str) + frequency_score.astype(str))


###############################################################
# Görev 4: RF Skorunun Segment Olarak Tanımlanması
###############################################################

# Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.

"""
can't loose them   : Geçmiş dönemde sık işlem yapmış fakat uzun süredir işlem yapmayan müşterilerdir. Geri kazanılması
                     önemli bir gruptur.
at risk            : Geçmiş dönemde yine sık işlem yapan bir sınıftır. Uzun süredir işlem yapmayan müşterilerdir.
hibernating        : Geçmişte nadiren işlem yapmış ve uzun süredir yeni bir işlem yapmamış müşterileri tanımlamaktadır.
about_to_sleep     : Son işlemini orta vadede yapmış fakat sık işlem yapmayan müşterilerdir.
need_attention     : Yenilik ve frekans bakımından ortada kalan bir sınıftır. İyi bir şekilde yönlendirilerek
                     loyal customer, champions segmentine geçirilebilir.
loyal customers    : Sık ve yakın tarihlerde işlem gerçekleştiren sadık müşterilerdir.
champions          : En yakın tarihlerde en çok işlemi yapan segmenttir.
potential loyalist : Yakın tarihte işlem yapmış fakat çok sık işlem yapmayan müşterilerdir. İlgilenilmesi durumunda
                     şirket için daha yararlı olabilecekleri loyal customers ve champions segmetine kolaylıkla 
                     geçiş yapabilirler.
promising          : Yakın tarihte işlem yapmış fakat çok sık işlem yapmayan müşterilerdir.
new customers      : Yeni müşterilerdir. Yakın bir tarihte yeni işlemler yapmışlardır.
"""

# Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

# SEGMENTLERIN ISIMLENDIRILMESI
seg_map = {
    r'[12]{2}': 'hibernating',
    r'[12][34]': 'at_risk',
    r'[12]5': 'cant_loose',
    r'3[12]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[34][45]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[45][23]': 'potential_loyalists',
    r'5[45]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

###############################################################
# Görev 5: Aksiyon Zamanı!
###############################################################

# Adım 1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm.groupby("segment").agg({"recency": lambda x: x.mean(),
                            "frequency": lambda x: x.mean(),
                            "monetary": lambda x: x.mean()})

# Adım 2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun
# ve müşteri id'lerini csv olarak kaydediniz.

# a. segment: champions, loyal_customers, monetary >= 250, kadın kategorisinden alışveriş yapan müşteriler

# index üzerinden işlemlere daha kolay devam edileceğinden df'in index verisi master_id olarak ayarlanmıştır.
df.set_index("master_id", inplace=True)

index_from_rfm = rfm[(rfm["segment"] == "champions") |
                     (rfm["segment"] == "loyal_customers") &
                     (rfm["monetary"] >= 250)].index

selected_customers = [id for id in index_from_rfm if "KADIN" in df["interested_in_categories_12"][id]]
selected_customers_df = pd.DataFrame(selected_customers, columns=["master_id"])
selected_customers_df.to_csv("week3/Hw3/FLO_RFM_Analizi/selected_customers_caseA.csv", index=False)

# b. segment: cant_loose, about_to_sleep, new_customers, erkek ve çocuk kategorisinden alışveriş yapan müşteriler

index_from_rfm = rfm[(rfm["segment"] == "cant_loose") |
                     (rfm["segment"] =="about_to_sleep") |
                     (rfm["segment"] =="new_customers")].index

# ERKEK ve COCUK kategorileri dahil edilmiş, AKTIFCOCUK kategorisi hariç tutulmuştur.

selected_customers = [id for id in index_from_rfm if "ERKEK" in df["interested_in_categories_12"][id] or "COCUK" in df["interested_in_categories_12"][id]]
selected_customers = [id for id in selected_customers if "AKTIFCOCUK" not in df["interested_in_categories_12"][id]]
selected_customers_df = pd.DataFrame(selected_customers, columns=["master_id"])
selected_customers_df.to_csv("week3/Hw3/FLO_RFM_Analizi/selected_customers_caseB.csv", index=False)

# Test edilen case'lerin sonuçlarının kontrol edilmesi
test = df.loc[selected_customers]

