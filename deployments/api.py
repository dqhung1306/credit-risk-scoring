"""
api.py — FastAPI server cho Credit Risk Model
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
import warnings
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import class DataPreprocessor từ utils
from utils import DataPreprocessor

warnings.filterwarnings("ignore")

import sys
sys.modules.setdefault('__main__', sys.modules[__name__])
sys.modules['__main__'].DataPreprocessor = DataPreprocessor

# ==================== KHỞI TẠO APP ====================
app = FastAPI(
    title="Credit Risk Scoring API",
    description="API dự đoán rủi ro tín dụng sử dụng LightGBM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== LOAD MODEL KHI KHỞI ĐỘNG ====================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgbm_best_model.pkl"
PROCESSOR_PATH = BASE_DIR / "models" / "data_preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    processor = joblib.load(PROCESSOR_PATH)
    processor.is_training = False
    explainer = shap.TreeExplainer(model)
    print("✅ Model và processor đã được load thành công")
except Exception as e:
    raise RuntimeError(f"❌ Không thể load model: {e}")


try:
    if processor.train_features is not None:
        FEATURES = [c for c in processor.train_features if c not in ('TARGET', 'SK_ID_CURR')]
        print(f"✅ Dùng train_features từ processor: {len(FEATURES)} features")
    else:
        raise AttributeError("train_features is None")
except AttributeError:
    try:
        FEATURES = model.feature_name_
    except AttributeError:
        FEATURES = list(model.booster_.feature_name())
    print(f"⚠️  Dùng feature_name từ model: {len(FEATURES)} features")


# ==================== SCHEMA ====================

class PredictBatchRequest(BaseModel):
    records: list[dict[str, Any]]
    features: list[str] | None = None


class PredictSingleRequest(BaseModel):
    record: dict[str, Any]
    features: list[str] | None = None


class SHAPRequest(BaseModel):
    record: dict[str, Any]
    features: list[str] | None = None
    top_n: int = 10


class PredictResponse(BaseModel):
    probabilities: list[float]
    classifications: list[str]
    total: int


class SHAPResponse(BaseModel):
    feature_names: list[str]
    shap_values: list[float]
    feature_values: list[float]


# ==================== HELPER ====================

def classify(prob: float) -> str:
    if prob < 0.35:
        return "LOW"
    elif prob < 0.65:
        return "MEDIUM"
    return "HIGH"


def records_to_df(records: list[dict], request_features: list[str]) -> pd.DataFrame:
    """
    Chuyển list of records → DataFrame khớp với FEATURES của model.

    [FIX BUG 1 + 2]
    Logic cũ: df = pd.DataFrame(records) → for col in features: df[col] = 0 nếu thiếu
              → df[features].fillna(0)
    Vấn đề:  - Thứ tự cột không đảm bảo khớp model
             - Nếu request gửi features khác FEATURES thì bị lệch

    Logic mới:
    1. Build DataFrame từ records (dùng request_features để biết cột nào cần)
    2. Reindex theo FEATURES của model (thứ tự chuẩn khi train)
    3. fillna(0) sau reindex → cột thiếu = 0, cột thừa bị loại
    """
    df = pd.DataFrame(records)

    # Reindex theo đúng thứ tự FEATURES của model
    # - Cột có trong model nhưng không có trong request → 0.0
    # - Cột có trong request nhưng không có trong model → bị loại (không reindex vào)
    df = df.reindex(columns=FEATURES, fill_value=0.0)

    return df.fillna(0.0).astype(float)


# ==================== ENDPOINTS ====================

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "model": type(model).__name__,
        "n_features": len(FEATURES),
        "features_source": "processor.train_features" if processor.train_features is not None else "model.feature_name_",
    }


@app.get("/features", tags=["System"])
def get_features():
    return {"features": FEATURES, "n_features": len(FEATURES)}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_batch(request: PredictBatchRequest):
    """
    Dự đoán xác suất rủi ro cho nhiều khách hàng.

    main.py gửi lên:
        records = processed_df[features].fillna(0).to_dict(orient="records")
        features = [c for c in processed_df.columns if c not in ['TARGET', 'SK_ID_CURR']]

    API nhận records → reindex về FEATURES model → predict.
    """
    if not request.records:
        raise HTTPException(status_code=400, detail="records không được rỗng")

    # [FIX] request.features chỉ dùng để log/debug; thực tế reindex theo FEATURES model
    try:
        X = records_to_df(request.records, request.features or FEATURES)
        probs = model.predict_proba(X)[:, 1].tolist()
        classifications = [classify(p) for p in probs]
        return PredictResponse(
            probabilities=probs,
            classifications=classifications,
            total=len(probs),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {str(e)}")


@app.post("/predict/single", tags=["Prediction"])
def predict_single(request: PredictSingleRequest):
    """Dự đoán cho 1 khách hàng."""
    try:
        X = records_to_df([request.record], request.features or FEATURES)
        prob = float(model.predict_proba(X)[:, 1][0])
        return {
            "probability": prob,
            "probability_pct": f"{prob:.1%}",
            "classification": classify(prob),
            "risk_level": (
                "Thấp" if prob < 0.35 else
                "Trung bình" if prob < 0.65 else
                "Cao"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {str(e)}")


@app.post("/shap", response_model=SHAPResponse, tags=["Explainability"])
def get_shap(request: SHAPRequest):
    """SHAP values cho 1 khách hàng (top N features quan trọng nhất)."""
    try:
        X = records_to_df([request.record], request.features or FEATURES)
        shap_vals = explainer.shap_values(X)

        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[1][0])
        else:
            sv = np.array(shap_vals[0]) if shap_vals.ndim == 2 else np.array(shap_vals)

        top_n = min(request.top_n, len(FEATURES))
        top_idx = np.argsort(np.abs(sv))[-top_n:][::-1]

        return SHAPResponse(
            feature_names=[FEATURES[i] for i in top_idx],
            shap_values=[float(sv[i]) for i in top_idx],
            feature_values=[float(X.iloc[0, i]) for i in top_idx],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi SHAP: {str(e)}")