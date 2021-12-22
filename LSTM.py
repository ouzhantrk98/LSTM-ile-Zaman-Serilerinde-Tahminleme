#Kütüphaneleri yükleyelim...

import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
from keras.layers import Dense, SimpleRNN, GRU
from keras.optimizers import SGD
from keras.layers import Dropout

plt.style.use('fivethirtyeight')



def hisse_senedi_fiyat_tahminle(hisse_senedi_adi):
    
    #Hisse senedini alalım..
    df = yf.download(hisse_senedi_adi,start = '2013-01-01',end='2021-07-05',progess=False)
    
    #satır ve sütunların sayısını aldığımız fonksiyon
    df.shape

    #fiyat geçmişini görsel olarak yazdıralım...
    plt.figure(figsize=(16,8))
    plt.title('Kapanis Fiyat Gecmisi')
    plt.plot(df['Close'])
    plt.xlabel('Tarih',fontsize=18)
    plt.ylabel('Kapanis Fiyati TL',fontsize=18)
    plt.show()

    #close sütunu için yeni bir dataframe oluşturalım...
    data = df.filter(['Close'])
    
    #Dataframe i numpy array'  çevirelim...
    dataset= data.values
    
    #Train veri seti için kullanılacak satırları alalım..
    training_data_len = math.ceil(len(dataset) * .8)
    
    #Veriyi ölçekleyelim...
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data= scaler.fit_transform(dataset)

    #training data set' i yaratalım..
    train_data = scaled_data[0:training_data_len, : ]
    
    #veriyi x_train ve y_train olarsak ikiye ayıralımm..
    #Bunu lstm nin çalışma yapısını sağlamak için yapıyoruz..
    x_train = []
    y_train = []
    x_train_size = 20
    for i in range(x_train_size,len(train_data)):
        x_train.append(train_data[i-x_train_size:i,0])
        y_train.append(train_data[ i , 0])
        
    #x_train ve y_train dizilerini numpy array' e çeviriyoruz bu sayede LSTM için kullanılabilecek
    x_train,y_train = np.array(x_train),np.array(y_train)
        
    #x_train' i 3D yapalım...
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    #LSTM modelini inşa edelim...
    model = Sequential()
    model.add(LSTM(70,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(70,return_sequences=False))

    model.add(Dense(40))
    model.add(Dense(1))

    #Modeli derleyelim..
    model.compile(optimizer='adam',loss='mean_squared_error')
    
    #Modeli eğitelim..
    model.fit(x_train,y_train,batch_size=16,epochs=1)

    #Test veri setini yaratalım..
    #training_data_len den tüm veri sayısına dek olan ölçeklenmiş verileri içeren 
    #test veri setini oluşturalım..
    test_data = scaled_data[training_data_len-x_train_size: , :]
    
    #x_test ve y_test test veri setleirni yaratalım..
    x_test= []
    y_test = dataset[training_data_len:,:]
    for i in range(x_train_size,len(test_data)):
        x_test.append(test_data[i-x_train_size:i,0])

    #x_test' i numpy array' e çevirelim...
    x_test = np.array(x_test)
    
    #x_test verisini 3D yapalım..
    x_test= np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    #Şimdi modele test verisini vererek tahminleme işlemini yapalım...
    predictions = model.predict(x_test)
    
    #Ölçeklenmiş verileri normale çevirelim...
    predictions = scaler.inverse_transform(predictions)
    
    #rmse' yi hesaplayalım ne kadar 0' a yakın olursa o kadar iyi...
    rmse = np.sqrt(np.mean(((predictions-y_test) ** 2)))
    
    print("****RMSE Degeri:**** ",rmse)
    
    train= data[:training_data_len]
    valid= data[training_data_len:]
    valid['Tahmin'] = predictions
    
    #Veriyi görselleştirelim...
    plt.figure(figsize=(16,8))
    plt.title(hisse_senedi_adi)
    plt.xlabel('Tarih')
    plt.ylabel('Kapanis Fiyati Tl',fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Tahmin']])
    plt.legend(['Egitim','Dogru','Tahmin'],loc='lower right')
    plt.show()
    
    gru_model(x_train,y_train,x_test,scaler,y_test,train,valid)
    bes_gunluk_tahmin(hisse_senedi_adi, scaler, model, valid,'FROTO.IS-LSTM5GUNLUK')

def gru_model(X_train,y_train,X_test,sc,y_test,train,valid):
    # The GRU architecture
    my_GRU_model = Sequential()
    # First GRU layer with Dropout regularisation
    my_GRU_model.add(GRU(units=130,return_sequences = True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    # Second GRU layer
    my_GRU_model.add(GRU(units=130, return_sequences=True, activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    # Third GRU layer
    my_GRU_model.add(GRU(units=130, return_sequences=True, activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    # Fourth GRU layer
    my_GRU_model.add(GRU(units=130, activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    
    # The output layer
    my_GRU_model.add(Dense(units=1))
    # Compiling the RNN
    my_GRU_model.compile(optimizer='adam',loss='mean_squared_error')
    # Fitting to the training set
    my_GRU_model.fit(X_train,y_train,epochs=16,batch_size=16, verbose=0)
    
    GRU_predictions = my_GRU_model.predict(X_test)
    GRU_predictions = sc.inverse_transform(GRU_predictions)
    
    rmse = np.sqrt(np.mean(((GRU_predictions-y_test) ** 2)))
    
    print('***GRUICIN-RMSE_DEGERI: ', rmse)
    
    valid['Tahmin'] = GRU_predictions
    
    plt.figure(figsize=(16,8))
    plt.title('GRU-FROTO.IS')
    plt.xlabel('Tarih')
    plt.ylabel('Kapanis Fiyati Tl',fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Tahmin']])
    plt.legend(['Egitim','Dogru','Tahmin'],loc='lower right')
    plt.show()
    
    bes_gunluk_tahmin('FROTO.IS', sc, my_GRU_model, valid,'FROTO.IS-GRU5GUNLUK')

#Her iki modelin 5 günlük tahmin için çalıştırılması ve karşılaştırılması..

def bes_gunluk_tahmin(hisse_senedi_adi,scaler,model,valid,etiket):
    
    
    gun_degeri = 5
    #Öncelikle veriyi alalım..
    #Şimdi ayın 18' inde olan hisse senedinin fiyatını tahminlemek için kullanalım eğitilen modeli..
    ford_quote= yf.download(hisse_senedi_adi,start = '2021-01-01',end='2021-05-31',progress=False)
    new_df=ford_quote.filter(['Close'])
    last_5_days=new_df[-gun_degeri:].values
    last_5_days_scaled = scaler.transform(last_5_days)
    
    for i in range(0,gun_degeri):
        X_test = []
        X_test.append(last_5_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
        pred_price= model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        last_5_days_scaled = scaler.inverse_transform(last_5_days_scaled)
        
        for i in range(0,gun_degeri-1):
            last_5_days_scaled[i] = last_5_days_scaled[i+1]
            
        last_5_days_scaled[gun_degeri-1] = pred_price[0]
        
        last_5_days_scaled = scaler.transform(last_5_days_scaled)
    
    last_5_days_scaled = scaler.inverse_transform(last_5_days_scaled)
    

    ford_quote= yf.download(hisse_senedi_adi,start = '2021-01-01',end='2021-06-07',progress=False)
    new_df=ford_quote.filter(['Close'])
    last_10_days=new_df[-10:].values
    predict_values = last_10_days[-10:-5]
    predict_values = np.append(predict_values,last_5_days_scaled.ravel())
    plt.figure(figsize=(16,8))
    plt.title(etiket)
    plt.xlabel('Tarih')
    plt.ylabel('Kapanis Fiyati Tl',fontsize=18)
    plt.plot(last_10_days)
    plt.plot(predict_values)
    plt.legend(['Dogru','Tahmin'],loc='lower right')
    plt.show()

#Ford otosan şirketinin 2020 yılı için hisse senedinin tahmini değerlerini oluşturan fonksiyonu çağırıyoruz.
hisse_senedi_fiyat_tahminle('CCOLA.IS')