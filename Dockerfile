FROM python:3.12-slim

WORKDIR /app

# Install the exact package versions the model was trained with
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs
COPY api.py database.py ./
COPY models/fraud_model.pkl models/fraud_model.pkl

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
