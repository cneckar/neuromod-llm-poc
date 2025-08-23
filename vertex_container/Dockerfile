# Custom Vertex AI Container with FULL NEUROMODULATION PROBE SYSTEM
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

COPY vertex_container/requirements.txt .

# Install Python dependencies
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Install additional neuromodulation dependencies for probe system
RUN python3.11 -m pip install --no-cache-dir \
    torch \
    transformers \
    numpy \
    scipy \
    matplotlib \
    seaborn

# Copy neuromodulation system from project root
COPY neuromod/ /app/neuromod/

# Copy packs from project root
COPY packs/ /app/packs/

# Copy vertex container files
COPY vertex_container/prediction_server.py .
COPY vertex_container/test_neuromodulation.py .

# Test neuromodulation system
RUN python3.11 test_neuromodulation.py

# Verify probe system imports
RUN python3.11 -c "import sys; sys.path.append('/app'); from neuromod import NeuromodTool; from neuromod.testing.simple_emotion_tracker import SimpleEmotionTracker; print('✅ FULL PROBE SYSTEM imported successfully'); print('✅ Emotion tracking system imported successfully')"

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables for neuromodulation
ENV NEUROMODULATION_ENABLED=true
ENV PROBE_SYSTEM_ENABLED=true
ENV EMOTION_TRACKING_ENABLED=true
ENV VERTEX_AI_MODE=true

# Run the prediction server with FULL PROBE SYSTEM
CMD ["python3.11", "prediction_server.py"]
