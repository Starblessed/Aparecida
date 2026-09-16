# Development and Implementation Plan for Project Aparecida

- **Version:** alpha
- **Date:** 16 sep. 2026
- **By:** Dannylo C. Maurício (Starblessed)

## Goal

Develop a system capable of detecting and recognizing potential missing children from a camera mesh/network by cross-reference with a known and reliable source database.

## Main Components - Name (Potential Solutions)

- AI Model (CLIP)
- Training Pipeline (DVC, Mlflow, FifyOne, LanceDB)
- Missing Children Picture Database (LanceDB, Chroma, QDrant, Pinecone)
- Detection System (UNK)
- Classification System (UNK)
- Filtering and Flagging System (UNK)
- RTSP-inflow pipeline (Gstreamer, OpenCV)
- Access API (FastAPI, Django)
- Alert PubSub Endpoint (Kafka, rabbitMQ, Redis)
