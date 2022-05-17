import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###############################################################
# Görev 1: Veriyi Anlama ve Hazırlama
###############################################################

# Adım 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("week3/Hw3/FLO_RFM_Analizi/flo_data_20K.csv")
df = df_.copy()


# Adım 2: Aykırı değerleri baskılamak için gerekli olan
# outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.


def outlier_thresholds(dataframe, variable):
    """
    Returns the lower and upper limits for outliers in a variable.

    Parameters
    ----------
    dataframe: pd.DataFrame
    variable: str

    Returns
    -------
    low_limit: int
    up_limit: int

    """
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    """
    Replaces outliers in a variable with the threshold values.

    Parameters
    ----------
    dataframe: pd.DataFrame
    variable: str

    Returns
    -------
    None

    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.

df.describe().T

if_has_outlier = ["order_num_total_ever_online", "order_num_total_ever_offline","customer_value_total_ever_offline",
                  "customer_value_total_ever_online"]

for i in if_has_outlier:
    replace_with_thresholds(df, i)

# Adım 4: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını
# ifade etmektedir. Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["order_num_total_ever_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever_omnichannel"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Adım 5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes # Veri tiplerini incelemek için
date_cols = [col for col in df.columns if "date" in col]
df[date_cols] = df[date_cols].apply(pd.to_datetime)


###############################################################
# Görev 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

date_max = df["last_order_date"].max()
calculation_date = date_max + pd.DateOffset(days=2)

# Adım 2: CLTV dataframe'in oluşturulması

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days/7
cltv_df["T_weekly"] = (calculation_date - df["first_order_date"]).dt.days/7
cltv_df["frequency"] = df["order_num_total_ever_omnichannel"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total_ever_omnichannel"] / df["order_num_total_ever_omnichannel"]

cltv_df.set_index("customer_id", inplace=True)

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]  # En az 2 alışveriş olmalı

###################################################################################
# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
###################################################################################

# Adım 1: BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

cltv_df["exp_sales_3_month"] = bgf.predict(3*4,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

cltv_df["exp_sales_6_month"] = bgf.predict(6*4,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

cltv_df[["exp_sales_3_month"]].sort_values("exp_sales_3_month", ascending=False).head(10)
cltv_df[["exp_sales_6_month"]].sort_values("exp_sales_6_month", ascending=False).head(10)

"""
İki beklentide de en çok satın alım yapması beklenen müşteriler ve sıralamaları aynıdır.
"""

# Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip
# exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])

# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_cltv_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6,  # 6 aylık hesaplama
                                   freq="W",  # T'nin frekans bilgisi
                                   discount_rate=0.01)

cltv = cltv.reset_index()  # customer_id'nin indexten veri kolonu haline dönüştürmek için kullanıldı.
cltv_df = cltv_df.reset_index() # customer_id'nin indexten veri kolonu haline dönüştürmek için kullanıldı.

# iki veri seti customer_id veri kolonu baz alınarak cltv_df üzerinde birleştirilmiştir.
cltv_final = cltv_df.merge(cltv, on="customer_id", how="left")

# ggf'ten clv olarak gelen isim cltv olarak değiştirildi.
cltv_final.rename({"clv": "cltv"}, axis=1, inplace=True)

# cltv verisine standartlaştırma uygunlanmıştır. (z=(x-mean)/std)
cltv_final["scaled_cltv"] = (cltv_final["cltv"] - cltv_final["cltv"].mean())/cltv_final["cltv"].std()

cltv_final[["customer_id", "cltv"]].sort_values("cltv", ascending=False).head(20)
"""
CLTV değerine göre sıralanan ilk 20 kişinin 3327 ile 1263 birim arasında değişen bir CLTV değeri vardır.
"""

###############################################################
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
###############################################################

# Adım 1: 6 aylık standartlaştırılmış CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve
# grup isimlerini veri setine ekleyiniz.

cltv_final["segment"] = pd.qcut(cltv_final["scaled_cltv"], 4, labels=["D", "C", "B", "A"])

# Adım 2: CLTV skorlarına göre müşterileri 4 gruba ayırmak mantıklı mıdır?
# Daha az mı ya da daha çok mu olmalıdır? Yorumlayınız

cltv_final.groupby("segment").agg({"count", "mean", "sum"})

# 4 segment olması durumu
#         recency_cltv_weekly                    T_weekly                    frequency                 monetary_cltv_avg                     exp_sales_3_month                exp_sales_6_month                exp_average_value                      cltv                     scaled_cltv
#                       count    mean        sum    count    mean        sum     count  mean       sum             count    mean         sum             count  mean      sum             count  mean      sum             count    mean         sum count    mean         sum       count   mean       sum
# segment
# D                      4987 139.000 693193.857     4987 162.183 808807.714      4987 3.769 18795.000              4987  93.152  464547.046              4987 0.409 2039.164              4987 0.818 4078.328              4987  98.691  492172.441  4987  80.340  400657.955        4987 -0.864 -4307.718
# C                      4986  92.630 461850.857     4986 112.818 562512.143      4986 4.405 21962.000              4986 125.789  627181.647              4986 0.525 2619.885              4986 1.051 5239.769              4986 132.251  659401.453  4986 138.312  689621.178        4986 -0.428 -2131.604
# B                      4986  81.988 408794.000     4986 100.327 500228.000      4986 5.093 25392.000              4986 160.637  800933.959              4986 0.601 2997.110              4986 1.202 5994.219              4986 168.001  837650.882  4986 199.533  994870.784        4986  0.033   165.609
# A                      4986  67.427 336191.714     4986  82.550 411592.857      4986 6.647 33140.000              4986 228.831 1140952.075              4986 0.773 3854.313              4986 1.546 7708.626              4986 238.024 1186787.639  4986 362.316 1806505.089        4986  1.258  6273.713

# 3 segment olması durumu
#         recency_cltv_weekly                    T_weekly                     frequency                 monetary_cltv_avg                     exp_sales_3_month                exp_sales_6_month                exp_average_value                      cltv                     scaled_cltv
#                       count    mean        sum    count    mean         sum     count  mean       sum             count    mean         sum             count  mean      sum             count  mean      sum             count    mean         sum count    mean         sum       count   mean       sum
# segment
# C                      6649 129.012 857797.571     6649 151.521 1007460.000      6649 3.876 25774.000              6649  98.907  657632.704              6649 0.431 2867.116              6649 0.862 5734.232              6649 104.627  695663.858  6649  90.444  601362.536        6649 -0.788 -5237.770
# B                      6648  86.415 574485.143     6648 105.808  703414.857      6648 4.729 31439.000              6648 142.356  946380.239              6648 0.563 3742.840              6648 1.126 7485.680              6648 149.254  992238.701  6648 167.056 1110591.425        6648 -0.211 -1404.006
# A                      6648  70.359 467747.714     6648  86.081  572265.857      6648 6.329 42076.000              6648 215.042 1429601.784              6648 0.737 4900.515              6648 1.474 9801.030              6648 223.843 1488109.856  6648 327.873 2179701.044        6648  0.999  6641.776

"""
4 segment olarak yapılan incelemede değerlendirildiğinde, her bir segmentte 4986 kişi bulunmaktadır.
cltv değerlerindeki segmentler arası artışlar 1.45 - 1.85 kat arası değişmektedir.
3 segement olarak yapılan değerlendirmede her bir segmette 6648 kişi ver almıştır. Kişi sayısı 4 segmentli duruma 
kıyasla 1.33 kat artarken, segmentler arasındaki cltv artışları 1.85 - 1.95 arasında daha stabil bir duruma gelmiştir.
Eğer müşteri sayısının artışı sorun teşkil etmiyorsa 3 segmentli bir sınıflandırma daha uygun olacaktır.
"""

# Adım 3: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

"""
D grubu müşteri yaşı en yüksek gruptur. Buna karşın bu müşterilerden 6 ay içinde yaklaşık 4080 işlem 
gelmesi beklenmektedir. Bu müşteriler için özel kampanyalar düzenlenerek B ve C segmenti müşterilerin özelliklerine
yakınlaştırılabilirler. Bu sayede harcamaları beklenen miktar neredeyse 2 katına ulaşmış olacaktır.

A grubu müşteriler müşteri yaşı bakımından en genç müşterilerdir.Diğer gruplara kıyasla çok sık alışveriş 
yapmaktadırlar. Bu müşterileri hedefleyen bir sadakat programı ile müşterilerin ilerleyen süreçlerde alt segmentlere 
düşmeleri engellenebilecektir. Şirkete genel olarak bakıldığında müşteri yaşı arttıkça yapılan işlem sayısı 
azalmaktadır. A segmentindeki müşterileri aktif tutmak ilerleyen süreçlerde bu gruptan kazanılan karın daha da 
artmasını sağlayacaktır.
"""



