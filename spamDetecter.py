import pickle
import streamlit as st
import numpy as np

model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

def main():
    st.title("Email Spam Classification Apps")
    st.subheader("Build With Streamlit & Python")
    msg = st.text_input("Enter Text : ")
    if st.button("Predict"):
        data = [msg]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]
        if result == 1:
            st.error("This is Spam Email")
        else:
            st.success("This is Not Spam Email")

main()