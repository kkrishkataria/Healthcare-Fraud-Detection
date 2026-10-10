# Healthcare Provider Fraud Risk API

This project trains an XGBoost model to estimate provider-level potential-fraud risk and serves predictions through FastAPI. A prediction is a prioritisation signal for review; it does not confirm fraud.

## Setup

Use the same Python environment to run the notebook and API so that the saved model artifacts and installed dependencies stay compatible.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `notebooks/01_data_understanding.ipynb` from top to bottom. Its final cell creates the model, imputer, and metadata in `models/`.

## Run the API

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Run the frontend

In a second terminal, start the React dashboard:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The dashboard uses `http://127.0.0.1:8000` by default. Set `VITE_API_URL` when the API is hosted elsewhere.

## Endpoints

- `GET /health` confirms that the API and model loaded.
- `GET /features` returns the required numeric input fields.
- `POST /predict` returns fraud probability, classification decision, and risk level.

The API accepts 17 direct provider measures from `GET /features`. It calculates the model's totals, ratios, and log transforms automatically before prediction. `AvgLengthOfStay`, `MedianLengthOfStay`, and `UniqueProcedureCount` may be supplied as `null`; their saved training medians are used and missing-value indicators are created automatically.
