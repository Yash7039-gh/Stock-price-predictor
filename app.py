import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from pandas_datareader import data as pdr
import yfinance as yf
import datetime
from keras.models import load_model
import streamlit as st
import sklearn
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

start = '2010-01-01'

x = datetime.datetime.now()
date = x.strftime("%Y-%m-%d")


st.title('Stock Price Prediction')
st.subheader('Mahesh Wagh')

link = 'To Study Stock Graph Trends: <a href="https://stock-trend-prediction-yashyadav.streamlit.app/" target="_blank">Click Here</a>'
st.markdown(link, unsafe_allow_html=True)



#SIDEBAR
def load_data():
    # Load data from CSV file
    data = pd.read_csv("dataset1.csv")
    return data

def search_ticker(company_name, data):
    # Search for the company name in the data
    filtered_data = data[data["Company_name"].str.contains(company_name, case=False)]
    if not filtered_data.empty:
        return filtered_data.iloc[0]["Symbol"]  # Return the ticker symbol of the first matching company
    else:
        return None


    
    # Add widgets to the sidebar
st.sidebar.title("Sidebar")
data = load_data()
    
    # Search box for filtering companies
search_term = st.sidebar.text_input("Search for a company:")
    
    # Filter data based on search term
filtered_data = data[data["Company_name"].str.contains(search_term, case=False)]
    
if not filtered_data.empty:
        st.sidebar.write(filtered_data)
else:
        st.sidebar.write("No matching companies found.")

ticker = search_ticker(search_term, data)
        
if ticker:
        st.sidebar.write(f"The ticker symbol for {search_term} is **{ticker}**")
        # user_input = f"The ticker symbol for {search_term} is {ticker}"
else:
        st.write("Company not found.")
    




user_input = st.text_input('Enter Stock Ticker', 'TSLA')
days_input = st.number_input('Enter Number of days to be predicted', step=1)


data = yf.download()
data = pdr.get_data_yahoo(user_input, period='5y', interval='1d')

df = data.reset_index()

st.write(df.describe())

opn = data[['Open']]
close = data[['Close']]

ds = opn.values
dc = close.values
#Open Value
normalizer = MinMaxScaler(feature_range=(0,1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))


train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size

ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

#Close Vaue
normalizer = MinMaxScaler(feature_range=(0,1))
dc_scaled = normalizer.fit_transform(np.array(dc).reshape(-1,1))

train_size_close = int(len(dc_scaled)*0.70)
test_size_close = len(dc_scaled) - train_size_close

dc_train, dc_test = dc_scaled[0:train_size,:], dc_scaled[train_size_close:len(dc_scaled),:1]

#LSTM
def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)- step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i+step,0])
    return np.array(Xtrain), np.array(Ytrain)


def create_dc(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)- step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i+step,0])
    return np.array(Xtrain), np.array(Ytrain)

time_stamp = 100

#Open
X_train_open, y_train_open = create_ds(ds_train, time_stamp)
X_test_open, y_test_open = create_ds(ds_test,time_stamp)

X_train_open =  X_train_open.reshape( X_train_open.shape[0],  X_train_open.shape[1], 1)
X_test_open = X_test_open.reshape(X_test_open.shape[0], X_test_open.shape[1], 1)


#Close
X_train_close, y_train_close = create_dc(dc_train, time_stamp)
X_test_close, y_test_close = create_dc(dc_test,time_stamp)

X_train_close =  X_train_open.reshape( X_train_close.shape[0],  X_train_close.shape[1], 1)
X_test_close = X_test_open.reshape(X_test_close.shape[0], X_test_close.shape[1], 1)

from keras.models import load_model
model = load_model("keras_model.h5")

#Open
train_predict_open = model.predict(X_train_open)
test_predict_open = model.predict(X_test_open)

train_predict_open = normalizer.inverse_transform(train_predict_open)
test_predict_open = normalizer.inverse_transform(test_predict_open)

test_open = np.vstack((train_predict_open,test_predict_open))


#close
train_predict_close = model.predict(X_train_close)
test_predict_close = model.predict(X_test_close)

train_predict_close = normalizer.inverse_transform(train_predict_close)
test_predict_close = normalizer.inverse_transform(test_predict_close)

test_close = np.vstack((train_predict_close,test_predict_close))

#Getting last 100 Days OPEN VALUES
future_input_open = ds_test[len(ds_test)-100:]
future_input_open = future_input_open.reshape(1,-1)
tmp_input_open = list(future_input_open)


#Getting last 100 Days Close VALUES
future_input_close = dc_test[len(ds_test)-100:]
future_input_close = future_input_close.reshape(1,-1)
tmp_input_close = list(future_input_close)

#Creating 100 days list
tmp_input_open = tmp_input_open[0].tolist()
tmp_input_close = tmp_input_close[0].tolist()

#Prediciting next 30 days price using current data
last_output_open = []
last_output_close = []
n_steps = 100
i=0

while(i<days_input):
    if(len(tmp_input_open)>100):
        future_input_open = np.array(tmp_input_open[1:])
        future_input_open = future_input_open.reshape(1,-1)
        future_input_open = future_input_open.reshape((1,n_steps,1))
        yhat = model.predict(future_input_open, verbose=0)
        tmp_input_open.extend(yhat[0].tolist())
        tmp_input_open = tmp_input_open[1:]
        last_output_open.extend(yhat.tolist())
        i=i+1
    else:
        future_input_open = future_input_open.reshape((1,n_steps,1))
        yhat = model.predict(future_input_open, verbose=0)
        tmp_input_open.extend(yhat[0].tolist())
        last_output_open.extend(yhat.tolist())
        i=i+1

while(i<days_input):
    if(len(tmp_input_close)>100):
        future_input_close = np.array(tmp_input_close[1:])
        future_input_close = future_input_close.reshape(1,-1)
        future_input_close = future_input_close.reshape((1,n_steps,1))
        yhat = model.predict(future_input_close, verbose=0)
        tmp_input_close.extend(yhat[0].tolist())
        tmp_input_close = tmp_input_close[1:]
        last_output_close.extend(yhat.tolist())
        i=i+1
    else:
        future_input_close = future_input_close.reshape((1,n_steps,1))
        yhat = model.predict(future_input_close, verbose=0)
        tmp_input_close.extend(yhat[0].tolist())
        last_output_close.extend(yhat.tolist())
        i=i+1
        
print(last_output_open)
print(last_output_close)

plot_new = np.arange(1,101)
plot_pred = np.arange(101,101+days_input)

#plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[len(ds_scaled)-100: ]))
#plt.plot(plot_pred, normalizer.inverse_transform(last_output_open))

ds_new = ds_scaled.tolist()

ds_new.extend(last_output_open)
# plt.plot(ds_new[1200: ])

final_graph = normalizer.inverse_transform(ds_new).tolist()

roundoff_val = round(float(*final_graph[len(final_graph)-1]),2)

st.markdown(f"## Open Value: {roundoff_val}")

st.set_option('deprecation.showPyplotGlobalUse', False)
fig = plt.plot(final_graph, )
plt.ylabel("Price")
plt.xlabel("Time")
plt.title("{0} predicted Open Price".format(user_input))
plt.axhline(y=final_graph[len(final_graph)-1], color='red', linestyle=":", label='NEXT Predictions: {0}'.format(roundoff_val))
plt.legend()
st.pyplot()

fig = px.line(final_graph)
fin = final_graph[len(final_graph)-1]
print(type(fin))
fig.add_hline(y = fin[0],
                annotation_text='NEXT Predictions: {0}'.format(roundoff_val),
                annotation_position='top left',
                annotation=dict(font_size=20, font_family="Times New Roman"),
                annotation_font_color='red',
                fillcolor = "red", 
                line_dash = "dash" )
st.plotly_chart(fig, use_container_width=True)



#Close
dc_new = dc_scaled.tolist()

dc_new.extend(last_output_close)
# plt.plot(ds_new[1200: ])

final_graph_close = normalizer.inverse_transform(dc_new).tolist()

roundoff_val_close = round(float(*final_graph_close[len(final_graph_close)-1]),2)

st.markdown(f"## Close Value: {roundoff_val_close}")

st.set_option('deprecation.showPyplotGlobalUse', False)
fig_close = plt.plot(final_graph_close, )
plt.ylabel("Price")
plt.xlabel("Time")
plt.title("{0} predicted Close Price".format(user_input))
plt.axhline(y=final_graph[len(final_graph)-1], color='red', linestyle=":", label='NEXT Closing Predictions: {0}'.format(roundoff_val_close))
plt.legend()
st.pyplot()

fig_close = px.line(final_graph_close)
fin_close = final_graph_close[len(final_graph_close)-1]
print(type(fin_close))
fig_close.add_hline(y = fin_close[0],
                annotation_text='NEXT Closing Predictions: {0}'.format(roundoff_val_close),
                annotation_position='top left',
                annotation=dict(font_size=20, font_family="Times New Roman"),
                annotation_font_color='red',
                fillcolor = "red", 
                line_dash = "dash" )
st.plotly_chart(fig_close, use_container_width=True)


diff = round((roundoff_val_close - roundoff_val),2)

diff_Per = round(((roundoff_val_close-roundoff_val)/(roundoff_val/100)),2)

positive = "green"
negative = "red"
if(diff>=0):
    st.markdown(f"<h3 style='color:{positive}'>+{diff} ({diff_Per}%)</h3>", unsafe_allow_html=True)
else:
     st.markdown(f"<h3 style='color:{negative}'>{diff} ({diff_Per}%)</h3>", unsafe_allow_html=True)

