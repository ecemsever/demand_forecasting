import pandas as pd
import plotly.express as px
import streamlit as st
import os
import datetime
from PIL import Image
import plotly.graph_objects as go


dir = os.path.dirname(__file__)

# Text to show on the browser tab and page config to be widely fitted
st.set_page_config(
    page_title="Welcome to PowerRangers Dashboard",
    page_icon="👋",
    layout='wide'
    )

# Show contact information on the sidebar with powerranger photo for each team member along with a button that directs to their Linkedin page
with st.sidebar:
    red = Image.open(dir+'/images/red.png')
    yellow = Image.open(dir+'/images/yellow.png')
    green = Image.open(dir+'/images/green.png')
    black = Image.open(dir+'/images/black.png')
    st.markdown('<h1><center> 📬 Contact Us </center></h1>', unsafe_allow_html=True)
    st.write("##")

    # First row
    with st.container():
        col1, col2 = st.columns(2)

        col1.image(red)
        col1.link_button("Muratcan", "https://www.linkedin.com/in/muratcankaplan?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAADye9q4BjqgGfVqmNgKQS-HhQalOfJZBKyU&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_all%3BXhxGWeQVQZWMxiKi0iy1xg%3D%3D", use_container_width=True)

        col2.image(yellow)
        col2.link_button("Ecem", "https://www.linkedin.com/in/ecemsever-bilkent/", use_container_width=True)

    # Second row
    with st.container():
        col1, col2 = st.columns(2)
        col1.image(green)
        col1.link_button("Ahmet Emre", "https://www.linkedin.com/in/ahmet-emre-topba%C5%9F-267391121?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAAB4iVS8BN9tfUXZo2j0AcuCVnR603hSF_4A&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_all%3BleRDOKiKSZSdvj%2BtTawywA%3D%3D", use_container_width=True)

        col2.image(black)
        col2.link_button("Gürkan", "https://www.linkedin.com/in/g%C3%BCrkan-g%C3%BCnd%C3%BCz-970753193?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAAC1-sqQBNxIm1OVJyqtMhIZrD9Gs8PIj-yU&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_all%3BktNP0zL%2BSuqRXnutJ5IYEg%3D%3D", use_container_width=True)

# a little trick to shorten & center logo on the center of the page
col1, col2 ,col3 = st.columns(3)
powerrangers = Image.open(dir+'/images/powerrangers.png')
col2.image(powerrangers)

# Page title
st.markdown("<h3> <p><center> Energy Analytics for Victoria, Australia </center></p> </h3>", unsafe_allow_html=True)

# Read historical total demand data and construct various time granularity suc as year, month, week
def history():
    history_data = pd.read_csv(dir+'/historical_demand.csv')    
    date_col = 'timestamp'
    # Construct date granularity features for past data
    history_data['Day'] = pd.to_datetime(history_data[date_col]).dt.date
    history_data['Year'] = pd.to_datetime(history_data[date_col]).dt.year
    history_data['Month'] = pd.to_datetime(history_data[date_col]).dt.month
    history_data['Week'] = pd.to_datetime(history_data[date_col]).dt.strftime('%W')
    # Month and week values is filled with 0 for 1 digit values as the first digit to get the sorting correct in the plot
    history_data[['Month']]=history_data[['Month']].astype(str).apply(lambda x: x.str.zfill(2))
    history_data[['Week']]=history_data[['Week']].astype(str).apply(lambda x: x.str.zfill(2))
    # Month and week numbers are concatenated with year to differentiate different years
    history_data['Year/Month'] = history_data['Year'].astype(str) + '/' + history_data.Month.astype(str)
    history_data['Year/Week'] = history_data['Year'].astype(str) + '/' + history_data.Week.astype(str)
    history_data = history_data.rename(columns={"timestamp": "Date"})
    return history_data

#read 8 hours prediction
def data_8_hours():
    pred_8_hours = pd.read_csv(dir+'/8_hour_predictions.csv')    
    pred_8_hours=pred_8_hours.rename(columns={"timestamp": "Date"})
    return pred_8_hours

#read 168 hours prediction
def data_168_hours():
    pred_168_hours = pd.read_csv(dir+'/168_hour_predictions.csv')    
    pred_168_hours=pred_168_hours.rename(columns={"timestamp": "Date"})
    return pred_168_hours

# build dataframes
history_data = history()
pred_8_hours = data_8_hours()
pred_168_hours = data_168_hours()

# Mape calculation function
def calculate_mape(actual, predicted):
    return (1 / len(actual)) * sum(abs((actual - predicted) / actual) * 100)

# Function for converting to csv for download
def convert_df(df):
    return df.to_csv().encode('utf-8')

# Processing prediction data to prepare it for the mape calculation
# Get the latest num_hours values in the data and flatten num_hours predictions that corresponds to that actual values
def t_minus_1_data_process(df, num_hours):
    # get latest 8 values
    t_minus_1_all = df.iloc[(len(df)-(num_hours)):(len(df)+1)]
    # hold actual values in one dataframe
    t_minus_1_actual = t_minus_1_all[['Date','total_demand']].iloc[0:(num_hours+1)]
    # flatten 8 predictions that corresponds to that actual values 
    t_minus_1_pred = pd.DataFrame(t_minus_1_all.iloc[0,2:(num_hours+2)]).reset_index(drop=True)
    t_minus_1_pred = t_minus_1_pred.rename(columns={t_minus_1_pred.columns[-1] : "demand_prediction"})
    # merge actual and predicted values
    t_minus_1 = pd.concat([t_minus_1_actual.reset_index(drop=True), t_minus_1_pred.reset_index(drop=True)], axis=1)
    t_minus_1 = pd.DataFrame(t_minus_1)
    t_minus_1['demand_prediction'] = t_minus_1['demand_prediction'].astype(float)
    return t_minus_1
    
# Preparing data for the prediction plot. Flatten the last row of the data to get prediction of unseen data
def pred_data_process(df, num_hours):

    latest_pred = pd.DataFrame(df.iloc[-1,2:(num_hours+2)]).reset_index(drop=True)
    latest_pred = latest_pred.rename(columns={latest_pred.columns[-1] : "demand_prediction"})
    latest_pred_date = df['Date'].iloc[-1]
    # construct prediction dates by adding num_hours to the last observed date in the dataset
    pred_dates = pd.date_range(start=(pd.to_datetime(latest_pred_date)+ pd.Timedelta(hours=1)), end=(pd.to_datetime(latest_pred_date) + pd.Timedelta(hours=num_hours)),freq='H')
    # column bind unseen dates with predictions
    pred_w_dates = pd.concat([latest_pred.reset_index(drop=True), pd.DataFrame(pred_dates).reset_index(drop=True)], axis=1).rename(columns={0 : "Date", "demand_prediction" : "Prediction"})

    return pred_w_dates

# Name the tabs
tab1, tab2, tab3 = st.tabs(["🌍 Historical Analytics", "📈 Shift Based Predictive Analytics", "📊 Weekly Predictive Analytics"])

with tab1:
    
    history_data_daily = history_data.groupby(['Day'])['total_demand'].agg('sum').reset_index(name='Total Demand')

    col1, col2 = st.columns(2)
    #Default start date is the minimum date in the data
    start_date = min(history_data_daily["Day"])
    
    #Start Date filter
    sd = col1.date_input("📆 Select a starting date",start_date)

    #Default end date is the maximum date in the data
    end_date = max(history_data_daily["Day"])

    # End date filter
    ed = col2.date_input("📆 Select a end date",end_date)

    # Time granularity selectbox. 
    with col1:
        option = st.selectbox(
        'Please select a time granularity',
        ('Hourly', 'Daily', 'Weekly','Monthly'))

    # Depending on the selection, graph is updated. Also, for all cases dates are filtered with the selection of start/end date
    if option=='Daily' :
        # Create the line graph for daily
        line_graph = px.line(
            data_frame = history_data_daily[history_data_daily["Day"].between(sd, ed)], title='Total Demand Daily', 
            x='Day', y='Total Demand',labels={"Total Demand": "Total Demand - MW"})
        st.plotly_chart(line_graph,use_container_width = True)
        
    elif option == 'Hourly':
        # Create the line graph for hourly
        line_graph2 = px.line(
            data_frame = history_data[history_data["Day"].between(sd, ed)], title='Total Demand Hourly', 
            x='Day', y='total_demand',labels={"total_demand": "Total Demand - MW"},
            hover_data = ["Date"]
            )
        st.plotly_chart(line_graph2,use_container_width = True)
        
    elif option == 'Weekly' :
        pred_8_hours_weekly = history_data[history_data["Day"].between(sd, ed)].groupby(['Year/Week'])['total_demand'].agg('sum').reset_index(name='Total Demand')
        # Create the line graph for weekly
        line_graph4 = px.line(
             data_frame = pred_8_hours_weekly, title='Total Demand Weekly', 
             x='Year/Week', y='Total Demand',labels={"Total Demand": "Total Demand - MW"})
        st.plotly_chart(line_graph4,use_container_width = True)

    else:
        pred_8_hours_monthly = history_data[history_data["Day"].between(sd, ed)].groupby(['Year/Month'])['total_demand'].agg('sum').reset_index(name='Total Demand')
        # Create the line graph for monthly
        line_graph3 = px.line(
            data_frame = pred_8_hours_monthly, title='Total Demand Monthly', 
            x='Year/Month', y='Total Demand',labels={"Total Demand": "Total Demand - MW"})
        st.plotly_chart(line_graph3,use_container_width = True)


with tab2:
    # define 8 hours data points for prediction
    t_minus_1= t_minus_1_data_process(pred_8_hours, 8)
    pred_w_dates = pred_data_process(pred_8_hours, 8)
    
    # Download predictions
    csv = convert_df(pred_w_dates)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predictions_8hours.csv',
        mime='text/csv')
    
    # Update button to read csv and construct the dataframes all over again to get most up-to-date data
    if st.button('Update the 8 hours Data', type="primary"):
        pred_8_hours = data_8_hours()
        t_minus_1= t_minus_1_data_process(pred_8_hours, 8)
        pred_w_dates = pred_data_process(pred_8_hours, 8)

    #calculate mape by calling the function on 8 hours data
    mape_8 = calculate_mape(t_minus_1['total_demand'],t_minus_1['demand_prediction'])

    # Publishing the first row
    col1, col2, col3 = st.columns(3)
    # 8 hours is added to the last observed value as it is the last prediction value
    col1.metric("Prediction Available till", str(pd.to_datetime(pred_8_hours['Date'].iloc[-1]) + pd.Timedelta(hours=8)), "")
    col2.metric("Last Update", str(pd.Timestamp.today().strftime('%Y-%m-%d')) , "", delta_color="off")
    col3.metric("MAPE", str(round(mape_8,2))+"%")

    # Actual vs Prediction Graph
    line_graph0 = px.line(
        data_frame = t_minus_1, title = 'Actual vs Predicted Hourly Demand for the last 8-hour period', 
        x='Date', y = ['total_demand','demand_prediction'],
        hover_data = ["Date"],
        labels = {"timestamp": "Hour","variable":"","Date":"Hour"},
        color_discrete_sequence = ['red', 'blue']
        )
    
    # change the namings
    new = {"demand_prediction": "Prediction","total_demand": "Total Demand - MW"}
    line_graph0.for_each_trace(lambda t: t.update(name = new[t.name]))
    line_graph0.update_layout(yaxis=dict(range = [6000, max(t_minus_1['total_demand']*1.3)]))
    # change the position of legend
    line_graph0.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1),
                    yaxis_title="Total Demand - MW")

    st.plotly_chart(line_graph0,use_container_width = True)
  
    # Prediction Graph for the next 8 hours
    # For upper/lower ranges average MAPE that shown in the page is added/subracted from the prediction as error probabilities
    pred_8hours_graph = go.Figure()
    pred_8hours_graph.add_trace(go.Scatter(x=pred_w_dates['Date'], y=pred_w_dates['Prediction']+pred_w_dates['Prediction']*mape_8/100,
                    name='Prediction Upper Range'))
    pred_8hours_graph.add_trace(go.Scatter(x=pred_w_dates['Date'], y=pred_w_dates['Prediction'],
                    name='Prediction',fill='tonexty',line_color='red',text=pred_w_dates['Prediction']))
    pred_8hours_graph.add_trace(go.Scatter(x=pred_w_dates['Date'], y=pred_w_dates['Prediction']-pred_w_dates['Prediction']*mape_8/100,
                    name='Prediction Lower Range',fill='tonexty',line_color='green'))
    pred_8hours_graph.update_layout(title='Next 8 hours Hourly Predictions',
                      legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1),
                    yaxis_title = "Total Demand - MW",
                    xaxis_title = "Hour")
    pred_8hours_graph.update_yaxes(tickformat="~s")

    st.plotly_chart(pred_8hours_graph,use_container_width = True)

with tab3:
   
    # define 168 hours data points for prediction
    t_minus_1_168hours= t_minus_1_data_process(pred_168_hours, 168)
    pred_w_dates_168hours = pred_data_process(pred_168_hours, 168)
    
    # Download predictions
    csv = convert_df(pred_w_dates_168hours)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predictions_168hours.csv',
        mime='text/csv')

    # Update button to read csv and construct the dataframes all over again to get most up-to-date data
    if st.button('Update the Weekly Data', type="primary"):
        pred_168_hours = data_168_hours()
        t_minus_1_168hours= t_minus_1_data_process(pred_168_hours, 168)
        pred_w_dates_168hours = pred_data_process(pred_168_hours, 168)

    #calculate mape by calling the function on 168 hours data
    mape_168 = calculate_mape(t_minus_1_168hours['total_demand'],t_minus_1_168hours['demand_prediction'])

    # Publishing the first row
    col1, col2, col3 = st.columns(3)
    # 168 hours is added to the last observed value as it is the last prediction value
    col1.metric("Prediction Available till", str(pd.to_datetime(pred_168_hours['Date'].iloc[-1]) + pd.Timedelta(hours=168)), "")
    col2.metric("Last Update", str(pd.Timestamp.today().strftime('%Y-%m-%d')) , "", delta_color="off")
    col3.metric("MAPE", str(round(mape_168,2))+"%")

    # Actual vs Prediction Graph
    line_graph168 = px.line(
        data_frame = t_minus_1_168hours, title='Actual vs Predicted Hourly Demand for the last 168-hour period', 
        x='Date', y=['total_demand','demand_prediction'],
        hover_data=["Date"],
        labels={"timestamp": "Date","variable":""},
        color_discrete_sequence=['red', 'blue']
        )
    # change the namings
    new = {"demand_prediction": "Prediction","total_demand": "Actual Demand", "value": "Total Demand - MW"}
    line_graph168.for_each_trace(lambda t: t.update(name = new[t.name]))

    # change the position of legend
    line_graph168.update_layout(legend = dict(
                    orientation = "h",
                    yanchor = "bottom",
                    y = 1.02,
                    xanchor = "right",
                    x = 1),
                    yaxis_title="Total Demand - MW")
    st.plotly_chart(line_graph168,use_container_width = True)

    # Prediction Graph for the next 168 hours
    # For upper/lower ranges average MAPE that shown in the page is added/subracted from the prediction as error probabilities
    pred_168hours_graph = go.Figure()
    pred_168hours_graph.add_trace(go.Scatter(x = pred_w_dates_168hours['Date'], 
                                             y = pred_w_dates_168hours['Prediction'] + pred_w_dates_168hours['Prediction']*mape_168/100,
                    name='Prediction Upper Range'))
    pred_168hours_graph.add_trace(go.Scatter(x = pred_w_dates_168hours['Date'], 
                                             y = pred_w_dates_168hours['Prediction'],
                    name='Prediction',fill='tonexty',line_color='red'))
    pred_168hours_graph.add_trace(go.Scatter(x = pred_w_dates_168hours['Date'], 
                                             y = pred_w_dates_168hours['Prediction'] - pred_w_dates_168hours['Prediction']*mape_168/100,
                    name='Prediction Lower Range', fill = 'tonexty', line_color='green'))
    pred_168hours_graph.update_layout(title='Next 168 hours Hourly Predictions',
                      legend = dict(
                        orientation = "h",
                        yanchor = "bottom",
                        y = 1.02,
                        xanchor="right",
                        x = 1),
                    yaxis_title = "Total Demand - MW",
                    xaxis_title = "Date")
    pred_168hours_graph.update_yaxes(tickformat="~s")

    st.plotly_chart(pred_168hours_graph,use_container_width = True)
