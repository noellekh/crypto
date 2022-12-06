import streamlit as st
import pandas as pd
import numpy as np



#image backgroud
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-vector/abstract-background-technology-style-white-low-poly-connection-with-nodes-global-data-blockchain-plexus-future-perspective-backdrop-vector-illustration_78474-1029.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url()

#add title

st.title("Hello Team Crypto")


#add head
add_bg_from_url()
st.header("What crypto do you already have ? ")


genres =['Green', 'Yellow', 'Red', 'Blue','Orange','Noelle', 'JohnyCrypt','GuiCrypt']

with st.form(key='my_form'):
    genre=st.multiselect('Select crypto', genres)

    submit_button = st.form_submit_button(label='Submit')



#add head
add_bg_from_url()
st.header("What do you have to do with those one...")

result=st.button("Predict")

st.header("list what to do Keep/Sell/Rebuy..")
##list what to do Keep/Sell/Rebuy

df = pd.DataFrame(np.random.randn(5, 3),
    columns=['past', 'today', 'future'])

st.dataframe(df)


##give shart from each crypto
#for crypto in options:
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['past', 'today', 'future'])

st.area_chart(chart_data)

st.header("The sentiment of each crypt")
###

st.write("___________________________________________________________________")

#add head
add_bg_from_url()
st.header("Want to invest in new crypto ?")


df = pd.DataFrame(np.random.randn(5, 3),
    columns=['Propostion', 'DL/FT'])

st.dataframe(df)
