# Imagen para Google Cloud Run
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Entrena todos los modelos al construir (usa los CSV de data/, no llama APIs externas)
RUN python run_pipeline.py
ENV PORT=8080
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
