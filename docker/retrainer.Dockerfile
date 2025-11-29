FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Install cron
RUN apt-get update && apt-get install -y cron

# Copy cron job
COPY docker/retrain-cron /etc/cron.d/retrain-cron

# Give execution permission
RUN chmod 0644 /etc/cron.d/retrain-cron

# Apply cron job
RUN crontab /etc/cron.d/retrain-cron

CMD ["cron", "-f"]
