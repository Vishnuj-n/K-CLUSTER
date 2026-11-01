import os
# Fix Windows joblib loky wmic warning and locale issue
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"

from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Mall Customers K-Means API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Mall_Customers.csv")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

def load_and_preprocess_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates().dropna()
    
    display_df = df.copy()
    
    df_encoded = df.copy()
    if "Gender" in df_encoded.columns:
        df_encoded["Gender_Numeric"] = df_encoded["Gender"].map({"Female": 0, "Male": 1})
        
    return display_df, df_encoded

@app.get("/api/meta")
def get_meta():
    display_df, _ = load_and_preprocess_dataset()
    features = [
        {"id": "Annual Income (k$)", "label": "Annual Income (k$)", "type": "numeric"},
        {"id": "Spending Score (1-100)", "label": "Spending Score (1-100)", "type": "numeric"},
        {"id": "Age", "label": "Age", "type": "numeric"},
        {"id": "Gender_Numeric", "label": "Gender (0: Female, 1: Male)", "type": "categorical"}
    ]
    return {
        "total_records": len(display_df),
        "available_features": features,
        "default_x": "Annual Income (k$)",
        "default_y": "Spending Score (1-100)"
    }

class ClusterRequest(BaseModel):
    k: int = 5
    feature_x: str = "Annual Income (k$)"
    feature_y: str = "Spending Score (1-100)"
    additional_features: Optional[List[str]] = []
    scale_features: bool = True
    random_state: int = 42

@app.post("/api/cluster")
def run_cluster(req: ClusterRequest):
    if req.k < 2 or req.k > 10:
        raise HTTPException(status_code=400, detail="k must be between 2 and 10")
    
    display_df, df_encoded = load_and_preprocess_dataset()
    
    features = [req.feature_x, req.feature_y] + [f for f in req.additional_features if f not in [req.feature_x, req.feature_y]]
    
    for f in features:
        if f not in df_encoded.columns:
            raise HTTPException(status_code=400, detail=f"Feature '{f}' not found in dataset")
    
    X_raw = df_encoded[features].values
    
    if req.scale_features:
        scaler = StandardScaler()
        X_used = scaler.fit_transform(X_raw)
    else:
        scaler = None
        X_used = X_raw

    kmeans = KMeans(n_clusters=req.k, random_state=req.random_state, n_init=10)
    labels = kmeans.fit_predict(X_used)
    
    try:
        sil_score = float(silhouette_score(X_used, labels))
    except Exception:
        sil_score = 0.0

    inertia = float(kmeans.inertia_)

    # Calculate centroids in original unscaled space for clear visual alignment
    if req.scale_features and scaler is not None:
        centroids_raw = scaler.inverse_transform(kmeans.cluster_centers_)
    else:
        centroids_raw = kmeans.cluster_centers_

    # Format points for response
    points = []
    for i, row in display_df.iterrows():
        pt = {
            "id": int(row["CustomerID"]) if "CustomerID" in row else i,
            "Gender": row.get("Gender", ""),
            "Age": int(row["Age"]),
            "Annual Income (k$)": float(row["Annual Income (k$)"]),
            "Spending Score (1-100)": float(row["Spending Score (1-100)"]),
            "x": float(df_encoded.iloc[i][req.feature_x]),
            "y": float(df_encoded.iloc[i][req.feature_y]),
            "cluster": int(labels[i])
        }
        points.append(pt)

    centroids = []
    for c_idx, c_vals in enumerate(centroids_raw):
        centroids.append({
            "cluster": c_idx,
            "x": float(c_vals[0]),
            "y": float(c_vals[1]),
            "label": f"Cluster {c_idx} Center"
        })

    # Cluster summary statistics
    cluster_summaries = []
    # Compute overall medians for dynamic segment naming
    income_median = float(display_df["Annual Income (k$)"].median())
    spending_median = float(display_df["Spending Score (1-100)"].median())
    age_median = float(display_df["Age"].median())

    def derive_segment_name(avg_income, avg_spending, avg_age):
        high_income = avg_income >= income_median
        high_spending = avg_spending >= spending_median
        young = avg_age < age_median
        if high_income and high_spending:
            return "High-Value Targets" if not young else "Young VIP Spenders"
        elif high_income and not high_spending:
            return "Cautious High Earners" if not young else "Young Conservative Earners"
        elif not high_income and high_spending:
            return "Impulse Buyers" if young else "Lifestyle Spenders"
        else:
            return "Budget-Conscious" if not young else "Thrifty Young Shoppers"

    for c in range(req.k):
        sub = display_df.iloc[labels == c]
        avg_income = round(float(sub["Annual Income (k$)"].mean()), 1)
        avg_spending = round(float(sub["Spending Score (1-100)"].mean()), 1)
        avg_age = round(float(sub["Age"].mean()), 1)
        name = derive_segment_name(avg_income, avg_spending, avg_age)

        # Business interpretation
        if avg_income >= income_median and avg_spending >= spending_median:
            interpretation = "High-revenue customers. Frequent, high-value purchases."
            strategy = "Loyalty rewards, premium offers, early access programs."
        elif avg_income >= income_median and avg_spending < spending_median:
            interpretation = "High earners who spend conservatively. Untapped potential."
            strategy = "Personalized outreach, value-based promotions, trust building."
        elif avg_income < income_median and avg_spending >= spending_median:
            interpretation = "Lower income but high engagement. Impulse-prone segment."
            strategy = "Installment plans, limited-time deals, lifestyle branding."
        else:
            interpretation = "Price-sensitive shoppers. Respond well to discounts."
            strategy = "Discount campaigns, essential bundles, value messaging."

        cluster_summaries.append({
            "cluster": c,
            "name": name,
            "count": int(len(sub)),
            "percentage": round(float(len(sub) / len(display_df) * 100), 1),
            "avg_age": avg_age,
            "avg_income": avg_income,
            "avg_spending": avg_spending,
            "gender_distribution": {
                "Female": int((sub["Gender"] == "Female").sum()),
                "Male": int((sub["Gender"] == "Male").sum())
            } if "Gender" in sub.columns else {},
            "interpretation": interpretation,
            "strategy": strategy
        })

    return {
        "k": req.k,
        "feature_x": req.feature_x,
        "feature_y": req.feature_y,
        "features_used": features,
        "inertia": inertia,
        "silhouette_score": round(sil_score, 4),
        "centroids": centroids,
        "points": points,
        "cluster_summaries": cluster_summaries
    }

class ElbowRequest(BaseModel):
    feature_x: str = "Annual Income (k$)"
    feature_y: str = "Spending Score (1-100)"
    additional_features: Optional[List[str]] = []
    scale_features: bool = True
    max_k: int = 10

@app.post("/api/elbow")
def calculate_elbow(req: ElbowRequest):
    _, df_encoded = load_and_preprocess_dataset()
    features = [req.feature_x, req.feature_y] + [f for f in req.additional_features if f not in [req.feature_x, req.feature_y]]
    
    X_raw = df_encoded[features].values
    if req.scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        X = X_raw

    results = []
    for k in range(1, req.max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        sil = None
        if k > 1:
            try:
                sil = round(float(silhouette_score(X, km.labels_)), 4)
            except Exception:
                sil = None
        results.append({
            "k": k,
            "inertia": round(float(km.inertia_), 2),
            "silhouette": sil
        })
    
    return {"elbow_data": results}

if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
