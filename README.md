# Welcome to our Short Term Load Forecasting (STLF) project for Victoria, Australia!👋

- Aim is to predict the next 8 hours and 168 hours to obtain optimum operational efficiency for energy industry

- The data is gathered from https://aemo.com.au/

- A user-friendly user dashboard is provided to present prediction result as well as historical data analysis.

## 🌐 Folder structure under this repository:


### model:
- This includes XGBoost pickle files (which is found to be best performing model among ARIMA, SARIMA, LSTM, linear regression) for 8 and 168 hours prediction models.


### pipeline:
- This is the intermediate process to enable dashboard code to use prediction result. 
- Model outputs are processed to be ready for visualization.
- Also, predictions are made continously as we simulate past data to as if it is happening in real-time.


### dashboard:
- It is using Streamlit and with the update button, predictions are updated to take most recent predicted values.
This button is only usable on the production version of the dashboard deployed on GCP Compute engine.

## 💻 Our Solution Architecture on Production

<img src="https://github.com/ecemsever/demand_forecasting/assets/64542150/49fbf9b7-dbdb-4a70-b042-8308edfef7fa)https://github.com/ecemsever/demand_forecasting/assets/64542150/49fbf9b7-dbdb-4a70-b042-8308edfef7fa" style="width:600px;"/>
