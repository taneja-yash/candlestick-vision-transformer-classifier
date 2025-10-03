# Real-Time Candlestick Pattern Classification with Vision Transformers

## Overview
This project is an end-to-end framework for recognizing and classifying candlestick chart patterns using a custom Vision Transformer (ViT) architecture. It integrates deep learning, data augmentation, synthetic data generation, and real-time inference to deliver actionable insights for algorithmic trading.

## Features
- ***Custom Vision Transformer***: Multi-module ViT with patch embedding, positional encoding, multi-head self-attention, and MLP blocks.
- ***Advanced Data Pipeline***: PyTorch-based loader with Albumentations augmentations (resizing, cropping, color jitter, normalization) and custom collate functions incorporating CutMix and MixUp for training robustness.
- ***Synthetic Data Generation & Annotation***: Scripts for creating diverse candlestick images (Doji, Engulfing, Morning/Evening Star) and a Python-based GUI annotation tool with OpenCV for hand-labeled screenshots.
- ***Real-Time Inference***: Capture live candlestick streams using MSS and OpenCV with prediction overlays and probability bars.
- ***Performance Visualization***: TensorBoard-style summaries, Matplotlib grids, and torchvision plots for monitoring model metrics.
- ***Interactive Dashboard***: Dash application for dynamic candlestick charting with integrated live model predictions.

## Achievements
- Successfully classified five major candlestick patterns with high reliability.
- Curated a balanced training dataset of over 1,000 images, reducing manual labeling time by 60% and improving model generalization.
- Delivered a live demo environment enabling traders to interpret model predictions in under one second.

## Tech Stack
- Python, PyTorch, Albumentations, OpenCV, MSS, Dash, Plotly, Matplotlib

## Project Impact
This project demonstrates the application of state-of-the-art deep learning techniques in algorithmic trading, combining synthetic data generation, advanced augmentation strategies, and real-time visualization to create a complete ML workflow from data to actionable insights.

## Author
Yash Taneja
- [LinkedIn](https://linkedin.com/in/yash-taneja-07) | [GitHub](https://github.com/taneja-yash)
