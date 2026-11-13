import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

st.set_page_config(page_title="Titanic Survival", page_icon="🚢", layout="wide")
st.title("🚢 Titanic Survival Explorer & Predictor")

@st.cache_data
def load_data():
    return pd.read_csv(URL)

def preprocess(df):
    d = df.copy()
    d["Age"] = d["Age"].fillna(d["Age"].median())
    d["Embarked"] = d["Embarked"].fillna(d["Embarked"].mode()[0])
    d["Sex"] = d["Sex"].map({"male": 0, "female": 1})
    d = d[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    return pd.get_dummies(d, columns=["Embarked"])

@st.cache_resource
def train_model(df):
    data = preprocess(df)
    X, y = data.drop("Survived", axis=1), data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, list(X.columns), acc

df = load_data()
model, feature_cols, acc = train_model(df)

tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Analysis", "🔮 Predict"])

with tab1:
    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Passengers", len(df))
    c2.metric("Survival rate", f"{df['Survived'].mean():.1%}")
    c3.metric("Missing ages", int(df["Age"].isna().sum()))

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Survival by Sex**")
        st.bar_chart(df.groupby("Sex")["Survived"].mean())
        st.write("**Survival by Class**")
        st.bar_chart(df.groupby("Pclass")["Survived"].mean())

    with col2:
        st.write("**Age distribution by outcome**")
        fig, ax = plt.subplots()
        ax.hist(df[df.Survived == 1]["Age"].dropna(), bins=20, alpha=0.6, label="Survived")
        ax.hist(df[df.Survived == 0]["Age"].dropna(), bins=20, alpha=0.6, label="Died")
        ax.set_xlabel("Age")
        ax.legend()
        st.pyplot(fig)

    st.write("**Feature importance**")
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    st.bar_chart(imp)

with tab3:
    st.subheader("Would this passenger survive?")
    st.caption(f"Model: Random Forest — test accuracy {acc:.1%}")

    c1, c2, c3 = st.columns(3)
    pclass = c1.selectbox("Class", [1, 2, 3], index=2)
    sex = c1.radio("Sex", ["male", "female"], horizontal=True)
    age = c2.slider("Age", 0, 80, 30)
    fare = c2.number_input("Fare", 0.0, 520.0, 32.0)
    sibsp = c3.number_input("Siblings/Spouses aboard", 0, 8, 0)
    parch = c3.number_input("Parents/Children aboard", 0, 6, 0)
    embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"])

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": 1 if sex == "female" else 0,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked_C": embarked == "C",
            "Embarked_Q": embarked == "Q",
            "Embarked_S": embarked == "S",
        }])[feature_cols]

        prob = model.predict_proba(row)[0][1]
        if prob >= 0.5:
            st.success(f"✅ Likely to survive ({prob:.1%})")
        else:
            st.error(f"❌ Unlikely to survive ({prob:.1%})")
        st.progress(float(prob))
