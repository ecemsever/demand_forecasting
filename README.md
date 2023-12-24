Welcome to our Short Term Load Forecasting (STLF) project!
This is conducted for Victoria state of Australia to predict the next 8 hours and 168 hours.
Also, a user-friendly user dashboard is provided to present prediction result as well as historical data analysis.

Folder structure under this repository:


model:
This includes pickle files for 8 and 168 hours prediction models.


pipeline:
This is the intermediate process to enable dashboard code to use prediction result. 
Model outputs are processed to be ready for visulalizing.
Also, predictions are made continously as we simulate past data to as if it is happening in real-time. 


dashboard:
Codebase for our energy analytics dashboard. 
It is using Streamlit and with the update button, predictions are updated to take most recent predicted values.
