import os
# Fix Windows joblib loky wmic warning and locale issue
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"

from typing import List, Optional
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

    # Clean aliases without the ugly k$ notation
    if "Annual Income (k$)" in df_encoded.columns:
        df_encoded["Annual Income ($)"] = df_encoded["Annual Income (k$)"]
        df_encoded["Annual Income"] = df_encoded["Annual Income (k$)"]
        display_df["Annual Income ($)"] = display_df["Annual Income (k$)"]
        display_df["Annual Income"] = display_df["Annual Income (k$)"]
        
    return display_df, df_encoded

@app.get("/api/meta")
def get_meta():
    display_df, _ = load_and_preprocess_dataset()
    features = [
        {"id": "Annual Income ($)", "label": "Annual Income ($)", "type": "numeric"},
        {"id": "Spending Score (1-100)", "label": "Spending Score (1-100)", "type": "numeric"},
        {"id": "Age", "label": "Age", "type": "numeric"},
        {"id": "Gender_Numeric", "label": "Gender (0: Female, 1: Male)", "type": "categorical"}
    ]
    return {
        "status": "online",
        "total_records": len(display_df),
        "available_features": features,
        "default_x": "Annual Income ($)",
        "default_y": "Spending Score (1-100)"
    }

class ClusterRequest(BaseModel):
    k: int = 5
    feature_x: str = "Annual Income ($)"
    feature_y: str = "Spending Score (1-100)"
    additional_features: Optional[List[str]] = []
    scale_features: bool = True
    random_state: int = 42

@app.post("/api/cluster")
def run_cluster(req: ClusterRequest):
    if req.k < 2 or req.k > 10:
        raise HTTPException(status_code=400, detail="k must be between 2 and 10")
    
    display_df, df_encoded = load_and_preprocess_dataset()
    
    # Resolve aliases if legacy column name was sent
    fx = req.feature_x
    if fx not in df_encoded.columns:
        if fx == "Annual Income ($)" and "Annual Income (k$)" in df_encoded.columns:
            fx = "Annual Income (k$)"
        elif fx == "Annual Income (k$)" and "Annual Income ($)" in df_encoded.columns:
            fx = "Annual Income ($)"

    fy = req.feature_y
    if fy not in df_encoded.columns:
        if fy == "Annual Income ($)" and "Annual Income (k$)" in df_encoded.columns:
            fy = "Annual Income (k$)"
        elif fy == "Annual Income (k$)" and "Annual Income ($)" in df_encoded.columns:
            fy = "Annual Income ($)"

    features = [fx, fy] + [f for f in req.additional_features if f not in [fx, fy]]
    
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

    # Precompute cluster summaries to annotate points and centroids
    cluster_summaries = []
    income_col = "Annual Income ($)" if "Annual Income ($)" in display_df.columns else "Annual Income (k$)"
    spending_col = "Spending Score (1-100)"
    income_median = float(display_df[income_col].median())
    spending_median = float(display_df[spending_col].median())

    for c in range(req.k):
        sub = display_df.iloc[labels == c]
        avg_income = round(float(sub[income_col].mean()), 1)
        avg_spending = round(float(sub[spending_col].mean()), 1)
        avg_age = round(float(sub["Age"].mean()), 1)

        # Dynamic human-understandable categorization of what each cluster represents & signifies
        if 40 <= avg_income <= 75 and 35 <= avg_spending <= 65:
            name = "Moderate / Balanced"
            significance = "Average Income · Average Spending"
            interpretation = "Middle-class customers with stable, consistent shopping patterns."
            strategy = "Promote everyday essentials, brand loyalty rewards, and consistent value."
        elif avg_income >= 65 and avg_spending >= 60:
            name = "High-Value VIPs"
            significance = "High Income · High Spending"
            interpretation = "Affluent and active spenders. The most lucrative customer segment."
            strategy = "Exclusive VIP perks, premium product launches, and personalized concierge."
        elif avg_income >= 65 and avg_spending < 45:
            name = "Cautious Savers"
            significance = "High Income · Low Spending"
            interpretation = "High earning power with conservative spending habits. High untapped potential."
            strategy = "Focus on quality, value justification, product longevity, and trust-building."
        elif avg_income < 45 and avg_spending >= 60:
            name = "Impulse Spenders"
            significance = "Low Income · High Spending"
            interpretation = "Highly engaged, trend-seeking shoppers who spend freely despite lower income."
            strategy = "Limited-time offers, flash deals, trend collections, and flexible installments."
        elif avg_income < 45 and avg_spending < 45:
            name = "Budget-Conscious"
            significance = "Low Income · Low Spending"
            interpretation = "Frugal, price-sensitive shoppers who prioritize discounts and necessity items."
            strategy = "Discounts, clearance sales, coupons, and budget-friendly bundle deals."
        else:
            if avg_income >= income_median and avg_spending >= spending_median:
                name = "Affluent Spenders"
                significance = "Higher Income · Higher Spending"
                interpretation = "Above-average earnings and strong shopping activity."
                strategy = "Loyalty tiers, quality recommendations, and seasonal exclusives."
            elif avg_income >= income_median and avg_spending < spending_median:
                name = "Prudent Earners"
                significance = "Higher Income · Lower Spending"
                interpretation = "Above-average earnings with conservative shopping habits."
                strategy = "Value-driven marketing and high-ROI product recommendations."
            elif avg_income < income_median and avg_spending >= spending_median:
                name = "Aspirational Shoppers"
                significance = "Lower Income · Higher Spending"
                interpretation = "Active lifestyle shoppers with lower disposable income."
                strategy = "Attractive financing, bundle discounts, and social-trend promos."
            else:
                name = "Frugal Shoppers"
                significance = "Lower Income · Lower Spending"
                interpretation = "Careful spenders focused on budget economy."
                strategy = "Value packs, essential discounts, and entry-level options."

        cluster_summaries.append({
            "cluster": c,
            "name": name,
            "significance": significance,
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

    # Format points for response with cluster significance
    points = []
    for i, row in display_df.iterrows():
        c_label = int(labels[i])
        c_info = cluster_summaries[c_label]
        inc_val = float(row[income_col])
        pt = {
            "id": int(row["CustomerID"]) if "CustomerID" in row else i,
            "Gender": row.get("Gender", ""),
            "Age": int(row["Age"]),
            "Annual Income": inc_val,
            "Annual Income ($)": inc_val,
            "Annual Income (k$)": inc_val,
            "Spending Score (1-100)": float(row["Spending Score (1-100)"]),
            "x": float(df_encoded.iloc[i][fx]),
            "y": float(df_encoded.iloc[i][fy]),
            "cluster": c_label,
            "cluster_name": c_info["name"],
            "cluster_significance": c_info["significance"]
        }
        if len(features) > 2:
            pt["z"] = float(df_encoded.iloc[i][features[2]])
        points.append(pt)

    centroids = []
    for c_idx, c_vals in enumerate(centroids_raw):
        c_info = cluster_summaries[c_idx]
        c_item = {
            "cluster": c_idx,
            "name": c_info["name"],
            "significance": c_info["significance"],
            "x": float(c_vals[0]),
            "y": float(c_vals[1]),
            "label": f"Cluster {c_idx}: {c_info['name']}"
        }
        if len(c_vals) > 2:
            c_item["z"] = float(c_vals[2])
        centroids.append(c_item)

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
    feature_x: str = "Annual Income ($)"
    feature_y: str = "Spending Score (1-100)"
    additional_features: Optional[List[str]] = []
    scale_features: bool = True
    max_k: int = 10

@app.post("/api/elbow")
def calculate_elbow(req: ElbowRequest):
    _, df_encoded = load_and_preprocess_dataset()
    fx = req.feature_x
    if fx not in df_encoded.columns:
        if fx == "Annual Income ($)" and "Annual Income (k$)" in df_encoded.columns:
            fx = "Annual Income (k$)"
        elif fx == "Annual Income (k$)" and "Annual Income ($)" in df_encoded.columns:
            fx = "Annual Income ($)"

    fy = req.feature_y
    if fy not in df_encoded.columns:
        if fy == "Annual Income ($)" and "Annual Income (k$)" in df_encoded.columns:
            fy = "Annual Income (k$)"
        elif fy == "Annual Income (k$)" and "Annual Income ($)" in df_encoded.columns:
            fy = "Annual Income ($)"

    features = [fx, fy] + [f for f in req.additional_features if f not in [fx, fy]]
    
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
