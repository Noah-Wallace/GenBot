# Use Python 3.10 (matches your runtime.txt)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the HF Spaces default port
ENV PORT 7860
EXPOSE 7860

# Run your Flask app via Gunicorn on the correct host/port
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1"]
