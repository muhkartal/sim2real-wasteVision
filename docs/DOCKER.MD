# Docker Implementation for İstanbul Waste Detection

This document explains how to use the Docker implementation of the İstanbul Waste Detection project.

## Overview

The Docker implementation provides a containerized environment for:

1. Running the waste detection API
2. Executing the synthetic data generation pipeline
3. Serving the API through NGINX

## Requirements

-  [Docker](https://docs.docker.com/get-docker/) (v20.10+)
-  [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)
-  At least 8GB RAM
-  NVIDIA GPU (optional, for faster performance)

## Quick Start

### Start the API Service

```bash
# Build and start the API service
docker-compose up -d api nginx
```

This will start:

-  The waste detection API on port 8000
-  An NGINX proxy on port 80

### Check API Status

Once the services are running, you can check the API status:

```bash
curl http://localhost/health
```

### Access the Demo UI

Open your browser and navigate to:

```
http://localhost/demo
```

This will show a simple web interface for testing the waste detection API.

## Using GPU Acceleration

To use NVIDIA GPU acceleration, you need to:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Create a `docker-compose.override.yml` file with the following content:

```yaml
version: "3.8"

services:
   api:
      deploy:
         resources:
            reservations:
               devices:
                  - driver: nvidia
                    count: 1
                    capabilities: [gpu]
```

3. Start the services normally:

```bash
docker-compose up -d api nginx
```

## Running the Full Pipeline

To run the full data generation, training, and evaluation pipeline:

```bash
# Run the pipeline with default settings
docker-compose --profile generator up generator

# Or run with specific options
docker-compose run generator python run_pipeline.py --small
```

## Container Structure

-  **api**: Runs the FastAPI application
-  **generator**: Runs the synthetic data generation pipeline
-  **nginx**: Serves as a reverse proxy for the API

## Data Persistence

The following volumes are used for data persistence:

-  `./weights:/app/weights`: Model weights
-  `./dataset:/app/dataset`: Dataset files
-  `./logs:/app/logs`: Log files

## Environment Variables

You can customize the API service using environment variables:

-  `MODEL_PATH`: Path to the model weights (default: `/app/weights/synthetic_only.pt`)
-  `DATASET_PATH`: Path to the dataset (default: `/app/dataset`)
-  `HOST`: API host address (default: `0.0.0.0`)
-  `PORT`: API port (default: `8000`)
-  `DEBUG`: Enable debug mode (default: `false`)

## Building a Custom Image

To build a custom image:

```bash
docker build -t istanbul-waste-detection:custom .
```

## Troubleshooting

### API Not Responding

Check the container logs:

```bash
docker-compose logs api
```

### Model Not Found

Ensure your model weights are in the correct location:

```bash
ls -la weights/
```

If the weights are missing, you need to train the model first:

```bash
docker-compose run generator python run_pipeline.py --skip-render
```

### Out of Memory

If you encounter out of memory errors, increase the memory allocated to Docker, or use the `--small` flag:

```bash
docker-compose run generator python run_pipeline.py --small
```
