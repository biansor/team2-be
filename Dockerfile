FROM python:3.11

WORKDIR /app
COPY requirements.txt .

# Update pip and install dependencies with increased timeout and retries
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
EXPOSE 5000

CMD ["python", "app.py"]