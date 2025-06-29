import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os

# === データ読み込み ===
# ※ ファイル名は適宜変更してください
df = pd.read_csv("ロト6(検証)5月17日　当選番号予測.csv")  # ファイル名が違う場合は変更！

# 1列目：回数、2〜7列目：当選番号（6個）
numbers = df.iloc[:, 1:7].values

# 特徴量とラベルの作成（1つ前の当選番号から次の回を予測）
X, y = [], []
for i in range(1, len(numbers)):
    X.append(numbers[i - 1])
    y.append(numbers[i])

X = np.array(X)
y = np.array(y)

# One-hot エンコーディング
mlb = MultiLabelBinarizer(classes=range(1, 44))
mlb.fit(X)

X_bin = mlb.transform(X)
y_bin = mlb.transform(y)

# 学習用データとテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

# モデルの構築と学習
model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# === モデル保存 ===
output_path = os.path.join("api", "models", "loto6_model.pkl")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump((model, mlb), f)

print("✅ モデルを学習して保存しました！ ->", output_path)