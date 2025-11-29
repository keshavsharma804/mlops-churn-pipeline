FROM python:3.10

WORKDIR /app

# Copy top-level requirements.txt
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy UI folder
COPY ui/ /app/ui/

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
