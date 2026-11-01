# K-Means Customer Clustering Dashboard (React + FastAPI)

An end-to-end Machine Learning web application demonstrating **K-Means Clustering** on the `Mall_Customers.csv` dataset.

---

## Architecture: Why FastAPI over Flask?

| Feature | **FastAPI** | **Flask** |
| :--- | :--- | :--- |
| **Performance** | High-performance asynchronous ASGI powered by Starlette and Uvicorn | WSGI synchronous by default |
| **Data Validation** | Automatic typed validation with Pydantic | Requires manual request parsing or 3rd-party extensions |
| **Interactive API Docs**| Built-in automatic Swagger UI (`/docs`) and ReDoc (`/redoc`) | Requires extra packages (e.g. Flask-RESTX) |
| **Modern ML Stack** | Industry standard for deploying scikit-learn/PyTorch ML microservices | Legacy microframework |

---

## How to Run the Application

The server serves both the **FastAPI REST API** and the **React Dashboard** on a single unified port:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open your browser to:
- **Interactive React Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **FastAPI Interactive Docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Key Features

1. **Interactive 2D Cluster Visualization**:
   - Dynamic scatter plot color-coded by cluster membership.
   - Distinct triangle markers for calculated **Centroids** mapped back to natural units.
   - Hover and click inspection for customer details (Age, Gender, Income, Spending Score).
2. **Dynamic Hyperparameter Tuning**:
   - Real-time slider for $K$ (from 2 to 10).
   - Dynamic axis selection (`Annual Income`, `Spending Score`, `Age`).
   - Toggleable `StandardScaler` feature scaling and 3D clustering with Age.
3. **Elbow Method & Silhouette Optimization**:
   - Automated inertia computation from $K=1$ to $K=10$ to find the optimal "elbow" ($K=5$).
   - Dual-axis Silhouette Score curve to validate cluster cohesion and separation.
4. **Segment Profiling**:
   - Automatically computes cluster demographic summaries (Target/VIP customers, Careful spenders, Balanced spenders, Careless spenders).
