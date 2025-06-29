from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

# === モデルの読み込み ===
model_path = os.path.join("api", "models", "loto6_model.pkl")
with open(model_path, "rb") as f:
    model, mlb = pickle.load(f)

# === リクエスト用のデータ型 ===
class PredictionRequest(BaseModel):
    last_numbers: list[int]  # 例: [1, 5, 10, 20, 30, 40]

# === ルート確認用 ===
@app.get("/")
def read_root():
    return {"message": "FastAPIは正常に動作中です！"}

# === 予測API ===
@app.post("/predict")
def predict_numbers(req: PredictionRequest):
    try:
        input_data = np.array(req.last_numbers).reshape(1, -1)
        input_bin = mlb.transform(input_data)
        predicted_proba = model.predict_proba(input_bin)

        # 予測確率から上位6個を選ぶ
        proba = np.array([p[0][1] for p in predicted_proba])
        top6 = np.argsort(proba)[-6:] + 1
        top6.sort()

        return {"predicted_numbers": top6.tolist()}
    except Exception as e:
        return {"error": str(e)}
