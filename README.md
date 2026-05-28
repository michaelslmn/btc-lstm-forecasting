## Quick Start

```bash
git clone https://github.com/michaelslmn/btc-lstm-forecasting.git
cd btc-lstm-forecasting
docker build -t btc-lstm .
docker run --rm -v ${PWD}:/app btc-lstm --date 2025-07-01

# BTC/USDT Bidirectional LSTM Forecasting

A deep learning project for forecasting Bitcoin (BTC/USDT) prices using a Bidirectional LSTM neural network built with TensorFlow and Keras.

## Features

* Bidirectional LSTM forecasting
* EarlyStopping + ReduceLROnPlateau
* Dockerized environment
* CLI-based forecasting
* Forecast visualization export
* Reproducible ML workflow

---

## Project Structure

```bash
btc-lstm-forecasting/
│
├── data/
├── outputs/
├── src/
│   └── forecast.py
├── requirements.txt
├── Dockerfile
├── README.md
```

---

## Requirements

* Docker Desktop
* Python 3.12 (optional for local runs)

---

## Run with Docker

### Build image

```bash
docker build -t btc-lstm .
```

### Run forecast

PowerShell:

```powershell
docker run --rm -v ${PWD}:/app btc-lstm --date 2025-07-01
```

CMD:

```cmd
docker run --rm -v %cd%:/app btc-lstm --date 2025-07-01
```

---

## Output

Forecast charts are saved automatically inside:

```bash
outputs/
```

Example:

```bash
outputs/forecast_2025-07-01.png
```

---

## Model Architecture

* Bidirectional LSTM (64 units)
* Bidirectional LSTM (32 units)
* Dense layers
* Adam optimizer
* Mean Squared Error loss

---

## Dataset

Binance BTC/USDT futures historical data.

---

## Future Improvements

* Real-time BTC forecasting
* FastAPI deployment
* Streamlit dashboard
* Multi-step forecasting
* Model checkpoint saving
* GPU acceleration
