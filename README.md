# Advanced Trading Bot with Hybrid ML Model

This is a sophisticated cryptocurrency trading bot that uses a hybrid machine learning model (Random Forest + LSTM) to predict market movements and execute trades automatically on the Binance exchange. It features robust risk management, a client-side trailing stop-loss, and real-time notifications via Telegram.

## Features

  * **Hybrid AI Model:** Combines a Random Forest Classifier and an LSTM neural network for predictive analysis.
  * **Automated Trading:** Connects to the Binance API (Testnet or Mainnet) to execute trades 24/7.
  * **Advanced Risk Management:**
      * Calculates position size as a percentage of the total portfolio.
      * Uses dynamic leverage based on model confidence.
      * Protects open positions with a client-side trailing stop-loss based on the Average True Range (ATR).
  * **Persistent State:** Uses an SQLite database to track open trades, ensuring the bot can resume and manage positions even after a restart.
  * **Real-time Notifications:** Sends instant alerts for new trades and closed positions to a Telegram chat.
  * **Highly Configurable:** All trading strategies, risk parameters, and API keys are managed in a simple `config.yaml` file.
  * **Dockerized:** Includes a `Dockerfile` for easy, consistent deployment on any machine or cloud server.

## Technology Stack

  * **Python 3.10+**
  * **CCXT:** For connecting to the Binance exchange API.
  * **Pandas:** For data manipulation and analysis.
  * **TA (Technical Analysis Library):** For generating trading indicators.
  * **Scikit-learn:** For the Random Forest model.
  * **TensorFlow/Keras:** For the LSTM neural network model.
  * **Docker:** For containerized deployment.

## Project Structure

```
MyTradingBot/
│
├── 📂 bot_core/
│   ├── __init__.py
│   ├── model_handler.py
│   ├── risk_manager.py
│   ├── state_manager.py
│   └── telegram_handler.py
│
├── 📂 data/
│   ├── trades.db
│   └── ... (Historical CSV files will be generated here)
│
├── 📂 models/
│   ├── lstm.weights.h5
│   ├── rf_model.pkl
│   └── scaler.pkl
│
├── 📂 venv/
│   └── ...
│
├── 📜 config.yaml
├── 📜 download_data.py
├── 📜 Dockerfile
├── 📜 main.py
├── 📜 requirements.txt
└── 📜 trainer.py
```

## Local Setup and Installation

Follow these steps to set up and run the bot on your local machine.

### 1\. Prerequisites

  * Python 3.10 or higher
  * Git

### 2\. Clone the Repository

(If your code is not already in a folder, you can use this standard step)

```bash
git clone <your-repository-url>
cd MyTradingBot
```

### 3\. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment.

```bash
# For macOS / Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4\. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage Workflow

This is the standard workflow for setting up and running the bot for the first time.

### Step 1: Configure the Bot

Fill in your API keys, Telegram details, and trading parameters.

1.  Make a copy of the example configuration if one exists, or edit `config.yaml` directly.
2.  Open `config.yaml` and enter your credentials:
      * `binance`: Your API Key and Secret Key. Set `is_testnet: True` to start.
      * `telegram`: Your bot token and your personal chat ID.
3.  Review the `strategy` and `risk` parameters to match your preferences.

### Step 2: Download Historical Data

Run the download script to fetch the historical data needed for training the AI model. You can configure the pairs and number of years inside the script.

```bash
python download_data.py
```

### Step 3: Train the AI Model

This is a crucial one-time step. Run the trainer script to use the downloaded data to train and save the machine learning models. This may take several minutes.

```bash
python trainer.py
```

### Step 4: Run the Bot

You are now ready to start the bot.

```bash
python main.py
```

The bot will connect to Binance and start its trading cycle. To stop the bot, press `Control + C` in the terminal.

## Deployment with Docker

Using Docker is the recommended way to run the bot 24/7, as it creates a self-contained, reliable environment.

### 1\. Prerequisites

  * [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### 2\. Build the Docker Image

From your main project folder (`MyTradingBot`), run the following command to build the image.

```bash
docker build -t trading-bot .
```

### 3\. Run the Container

Start the bot in a detached (background) container.

```bash
docker run -d --name my-live-bot trading-bot
```

### 4\. Manage the Running Bot

  * **View live logs:**
    ```bash
    docker logs -f my-live-bot
    ```
  * **Stop the bot:**
    ```bash
    docker stop my-live-bot
    ```
  * **Restart the bot:**
    ```bash
    docker start my-live-bot
    ```
  * **Remove the container (after stopping):**
    ```bash
    docker rm my-live-bot
    ```

## Disclaimer

**This is not financial advice.** Trading cryptocurrencies is extremely risky and can result in significant financial loss. This bot is provided as an educational tool. You are solely responsible for your own trading decisions and any resulting profits or losses. Always test thoroughly on a **Testnet** account before considering live trading. The creators and contributors of this project are not liable for any financial outcomes.
