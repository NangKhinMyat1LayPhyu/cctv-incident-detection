# Runs the pipeline on CPU: analyse clips, run live on RTSP streams, serve the dashboard.
#   docker build -t incident-recon .
#   docker run --rm -v "$PWD/data:/data" -v "$HOME/models:/models" incident-recon \
#       analyze /data/clip.mp4 --weights /models/best.pt --out /data/out
#   docker run --rm -p 8000:8000 -v "$PWD/outputs:/outputs" -v "$HOME/models:/models" incident-recon \
#       serve --host 0.0.0.0 --live-dir /outputs/live --jobs-dir /outputs/jobs --weights /models/best.pt
FROM python:3.12-slim

# OpenCV needs these shared libraries even in its headless build
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only PyTorch first: the default wheel bundles CUDA and is ~10x larger
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY results/split_v2.json ./results/
RUN pip install --no-cache-dir -e ".[service]"
EXPOSE 8000

ENTRYPOINT ["incident-recon"]
CMD ["--help"]
