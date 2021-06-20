import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# 2 adet farklı dataset kullanıldı.
# Değerler sırasıyla;
# Distance: Yüz tanıma oranı, 1'den küçükse yüzü tanıyor.
# Duration: Yüz tanıma süresi.
# Actual: Gerçek sonuç.
# Prediction: Modelden gelen sonuç.
df = pd.DataFrame([
        [0.96, 3.47, True, True],
        [0.22, 6.12, False, True],
        [0.76, 2.72, True, True],
        [0.53, 3.55, True, True],
        [1.02, 5.14, False, False],
        [0.93, 3.97, True, True],
        [0.57, 2.85, True, True],
        [0.61, 3.51, True, True],
        [0.99, 3.93, True, False],
        [0.66, 2.25, True, True],
        [0.98, 3.94, True, False],
        [0.91, 3.12, True, True],
        [1.15, 5.84, False, False],
        [0.59, 4.17, True, True],
        [0.60, 2.95, True, True],
        [0.53, 3.22, True, True],
        [0.81, 4.13, True, True],
        [0.88, 3.11, True, True],
        [0.99, 3.45, True, True],
        [0.80, 2.18, True, True],
        [0.86, 3.07, True, True],
        [0.62, 3.84, True, True],
        [1.03, 4.98, False, False],
        [0.50, 3.51, True, True],
        [0.55, 5.84, False, False],
        [0.79, 4.17, True, True],
        [0.65, 2.95, True, True],
        [0.53, 3.22, True, True],
        [0.81, 4.13, True, True],
        [0.61, 6.11, True, True],
        [0.72, 3.45, True, True],
        [0.80, 2.18, True, True],
        [0.86, 3.07, True, True],
        [0.62, 3.84, True, True],
        [1.03, 4.98, True, False],
        [0.50, 3.51, True, True],
    ], columns=["Distance", "Duration", "Actual", "Predicted"])
train, test = df.iloc[:18], df.iloc[18:]

train

test

from sklearn.linear_model import LogisticRegression
vc1 = pd.crosstab(train.Predicted, train.Actual)
vc2 = pd.crosstab(test.Predicted, test.Actual)


from sklearn.metrics import accuracy_score
fig = plt.figure(figsize=(12,5))
ax1 = plt.subplot(121)
sn.heatmap(vc1, annot=True, cmap='Blues')
ax1.set_title("Model I")
ax2 = plt.subplot(122)
sn.heatmap(vc2, annot=True, cmap='Blues')
ax2.set_title("Model II")


