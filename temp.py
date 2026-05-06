# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
# Kendi bilgisayarındaki tam yolu tırnak içine yaz (r harfini unutma)
veriler = pd.read_csv(r"C:\Users\kader\OneDrive\Desktop\makina ögrenmesi\maaslar.csv")
print(veriler)

#data frame dilimleme
x= veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]
X=x.values
Y=y.values
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#plt.scatter(X,Y,color='pink')
#plt.plot(x,lin_reg.predict(X),color='blue')

#polinomal regression

from sklearn.preprocessing import PolynomialFeatures
# 1. Önce nesneyi oluşturuyoruz
poly_reg = PolynomialFeatures(degree = 2)

# 2. ŞİMDİ DİKKAT: fit_transform bir fonksiyondur ve poly_reg nesnesine aittir
x_poly = poly_reg.fit_transform(X)

# 3. Sonucu yazdır
print(x_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
# Önceki scatter ve plotları siliyoruz (sadece polinomal kalsın diye)
plt.scatter(X, Y, color='red') # Gerçek noktalar (Hocada kırmızı)

# Polinomal çizgiyi çizdiriyoruz
# Predict içine direkt dönüştürülmüş X'i (x_poly) veriyoruz
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')

plt.show()