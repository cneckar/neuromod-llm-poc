# Use Rocky Linux 9 with Python 3.11
FROM rockylinux:9

# Set working directory
WORKDIR /app

# Install system dependencies and Python 3.11
RUN dnf update -y && dnf install -y \
    git \
    curl \
    python3.11 \
    python3.11-pip \
    python3.11-devel \
    gcc \
    && dnf clean all

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3.11", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
