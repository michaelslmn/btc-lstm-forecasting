```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC/USDT Bidirectional LSTM Forecasting for Any Target Date
Improved version with:
- EarlyStopping
- ReduceLROnPlateau
- Gradient clipping
- Docker-safe plotting
- Output folder support
"""

import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# =========================
# CONFIG
# =========================
CSV_PATH = "data/binance6months_futures_2025-02-07_to_2025-08-06.csv"

OUTPUT_DIR = "outputs"

SEQUENCE_LENGTH = 50
HORIZON = 1

EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 5

LEARNING_RATE = 0.001


# =========================
# CREATE OUTPUT DIRECTORY
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# FORECAST FUNCTION
# =========================
def forecast(target_date):

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv(CSV_PATH)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df = df[['timestamp', 'close']].dropna()

    df.set_index('timestamp', inplace=True)

    # =========================
    # NORMALIZE DATA
    # =========================
    scaler = MinMaxScaler()

    df['scaled_close'] = scaler.fit_transform(df[['close']])

    # =========================
    # CREATE SEQUENCES
    # =========================
    data = df['scaled_close'].values

    X, y = [], []

    for i in range(SEQUENCE_LENGTH, len(data) - HORIZON):

        X.append(data[i - SEQUENCE_LENGTH:i])

        y.append(data[i:i + HORIZON])

    X = np.array(X)
    y = np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # =========================
    # VALIDATE DATE
    # =========================
    date_str = target_date

    available_dates = df.index.strftime('%Y-%m-%d').values

    if date_str not in available_dates:
        raise ValueError(
            f"Target date {target_date} not found in dataset"
        )

    split_index = (
        df.index.get_loc(f"{date_str} 00:00:00")
        - SEQUENCE_LENGTH
    )

    if split_index <= 0:
        raise ValueError(
            "Not enough historical data before target date"
        )

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    X_train, X_test = X[:split_index], X[split_index:]

    y_train, y_test = y[:split_index], y[split_index:]

    # =========================
    # MODEL
    # =========================
    model = Sequential([

        Input(shape=(X.shape[1], X.shape[2])),

        Bidirectional(
            LSTM(64, return_sequences=True)
        ),

        Bidirectional(
            LSTM(32)
        ),

        Dense(64),

        Activation('relu'),

        Dense(HORIZON)
    ])

    # =========================
    # OPTIMIZER
    # =========================
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss='mse'
    )

    # =========================
    # CALLBACKS
    # =========================
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # =========================
    # TRAIN MODEL
    # =========================
    model.fit(
        X_train,
        y_train,

        validation_split=0.1,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stop,
            reduce_lr
        ],

        verbose=1
    )

    # =========================
    # PREDICT
    # =========================
    predicted_scaled = model.predict(X_test)

    predicted = scaler.inverse_transform(
        predicted_scaled
    )

    # =========================
    # TIMESTAMPS
    # =========================
    timestamps = df.index[
        SEQUENCE_LENGTH + split_index + HORIZON:
    ]

    timestamps = timestamps[:len(predicted)]

    # =========================
    # RESULTS DATAFRAME
    # =========================
    df_result = pd.DataFrame({

        "timestamp": timestamps,

        "actual_close": df['close'].iloc[
            SEQUENCE_LENGTH + split_index + HORIZON:
            SEQUENCE_LENGTH + split_index + HORIZON + len(predicted)
        ].values,

        "predicted_close": predicted.flatten()
    })

    # =========================
    # FILTER TARGET DATE
    # =========================
    df_day = df_result[
        df_result['timestamp'].dt.strftime('%Y-%m-%d')
        == date_str
    ]

    # =========================
    # PLOT
    # =========================
    plt.figure(figsize=(14, 6))

    plt.plot(
        df_day['timestamp'],
        df_day['actual_close'],
        label="Actual",
        linewidth=2
    )

    plt.plot(
        df_day['timestamp'],
        df_day['predicted_close'],
        label="Predicted",
        linestyle='--'
    )

    plt.title(
        f"BTC/USDT LSTM Forecast vs Actual ({date_str})"
    )

    plt.xlabel("Time")

    plt.ylabel("Price (USD)")

    plt.grid(True)

    plt.legend()

    plt.xticks(rotation=45)

    plt.tight_layout()

    # =========================
    # SAVE PLOT
    # =========================
    output_file = (
        f"{OUTPUT_DIR}/forecast_{date_str}.png"
    )

    plt.savefig(output_file)

    plt.close()

    # =========================
    # SUCCESS MESSAGE
    # =========================
    print("\n✅ Forecast completed successfully")

    print(f"📊 Saved plot: {output_file}")


# =========================
# CLI ENTRY
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="BTC/USDT LSTM Forecast"
    )

    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Target date format: YYYY-MM-DD"
    )

    args = parser.parse_args()

    forecast(args.date)
```
