import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Simple training data
spam_texts = [
    "Win free money now click here",
    "Congratulations you won lottery claim prize",
    "Earn cash fast limited offer",
    "Free entry win iPhone click link",
    "You have won $1000 send bank details"
]
ham_texts = [
    "Hi, let's meet for lunch tomorrow",
    "Can you send me the project report?",
    "Happy birthday! Hope you have a great day",
    "Meeting scheduled at 10am in conference room",
    "Please review the attached document"
]

texts = spam_texts + ham_texts
labels = [1]*len(spam_texts) + [0]*len(ham_texts)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

st.title("Spam Classifier")
st.write("Enter a message to check if it's Spam or Not Spam")

msg = st.text_area("Your message:")

if st.button("Check"):
    if msg.strip() == "":
        st.warning("Please enter a message")
    else:
        pred = model.predict(vectorizer.transform([msg]))[0]
        if pred == 1:
            st.error("SPAM")
        else:
            st.success("NOT SPAM (Ham)")
