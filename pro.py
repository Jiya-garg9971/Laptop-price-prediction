import numpy as np
import streamlit as st
import pickle
import pandas as pd

with open('df2.pkl', 'rb') as f:
    df = pd.read_pickle(f)
st.title("Laptop Price Predictor")
with open('pipe2.pkl', 'rb') as t:
    # u = pickle._Unpickler(t)
    # u.encoding = 'latin1'
    # pipe = list(pickle.load(f, encoding='latin1'))
    pipe = pd.read_pickle(t)

company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

weight = st.number_input('Weight of the Laptop')

touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])

ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2800x1800', '2560x1600', '2560x1440', '2304x1440'])

cpu = st.selectbox('CPU', df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())

x = df.drop(columns=['Price'])
print(x.iloc[1])

if st.button('Predict Price'):
    # pass
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # calculating the ppi
    # need to split resolution
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2)+(Y_res**2))**0.5/screen_size
    query = np.array([company, type, ram, weight, touchscreen,
                     ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)
    st.title("The predicted price of this configuration is " +
             str(int(np.exp(pipe.predict(query)[0]))))
