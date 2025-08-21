# Custom Vertex AI Container for Neuromodulation
FROM rockylinux:9

# Set working directory
WORKDIR /app

# Install system dependencies and Python 3.11
RUN dnf update -y && dnf install -y --allowerasing \
    git \
    curl \
    python3.11 \
    python3.11-pip \
    python3.11-devel \
    gcc \
    && dnf clean all

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy neuromodulation system from project root
COPY ../neuromod/ /app/neuromod/

# Copy packs from project root
COPY ../packs/ /app/packs/

# Copy prediction server
COPY prediction_server.py .

# Copy test script
COPY test_neuromodulation.py .

# Test neuromodulation system
RUN python3.11 test_neuromodulation.py

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the prediction server
CMD ["python3.11", "prediction_server.py"]
