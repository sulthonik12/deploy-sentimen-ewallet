# ==========================================================
# DASHBOARD ANALISIS SENTIMEN DOMPET DIGITAL INDONESIA
# FINAL VERSION
# STEP 1 - CORE SYSTEM
# ==========================================================


import streamlit as st

import pandas as pd
import numpy as np

import os
import re
import string

import joblib

import plotly.express as px
import matplotlib.pyplot as plt

from wordcloud import WordCloud



# ==========================================================
# CONFIG
# ==========================================================


st.set_page_config(

    page_title="Analisis Sentimen",

    page_icon="📊",

    layout="wide"

)



# ==========================================================
# CSS
# ==========================================================


st.markdown(

"""
<style>


.main-title{

font-size:42px;
font-weight:800;
color:#1e3a8a;

}


.subtitle{

font-size:18px;
color:#64748b;

}


.card{

background:white;

padding:20px;

border-radius:15px;

box-shadow:
0 4px 15px rgba(0,0,0,0.08);

text-align:center;

}



.card-value{

font-size:32px;

font-weight:bold;

color:#2563eb;

}



.card-label{

font-size:15px;

color:#64748b;

}


</style>

""",

unsafe_allow_html=True

)



# ==========================================================
# PATH DEPLOYMENT
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset_final_lexicon.csv"
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "svm_baseline.pkl"
)


TFIDF_PATH = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)


LABEL_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)


RESULT_DIR = os.path.join(
    BASE_DIR,
    "result"
)





# ==========================================================
# LOAD DATA
# ==========================================================


@st.cache_data

def load_dataset():

    return pd.read_csv(

        DATA_PATH,

        low_memory=False

    )





# ==========================================================
# LOAD MODEL
# ==========================================================


@st.cache_resource

def load_model():


    model = joblib.load(

        MODEL_PATH

    )


    tfidf = joblib.load(

        TFIDF_PATH

    )


    encoder = joblib.load(

        LABEL_PATH

    )


    return model, tfidf, encoder





# ==========================================================
# INITIALIZATION
# ==========================================================


try:


    df = load_dataset()


    model, tfidf, encoder = load_model()


    system_ready=True



except Exception as e:


    system_ready=False


    st.error(

        f"Loading Error : {e}"

    )





# ==========================================================
# SIDEBAR
# ==========================================================


st.sidebar.title(

    "📊 Navigation"

)



menu = st.sidebar.radio(

    "Pilih Menu",

    [

        "🏠 Dashboard",

        "🤖 Prediksi",

        "📈 Evaluasi",

        "📂 Batch Prediction",

        "ℹ️ Tentang"

    ]

)





st.sidebar.divider()



st.sidebar.info(

"""

Model:

TF-IDF + Support Vector Machine


Dataset:

Dompet Digital Indonesia


"""

)





# ==========================================================
# DASHBOARD BASIC
# ==========================================================


if menu=="🏠 Dashboard":



    st.markdown(

    """

    <div class="main-title">

    Dashboard Analisis Sentimen

    </div>


    <div class="subtitle">

    Analisis sentimen ulasan aplikasi dompet digital

    </div>

    """,

    unsafe_allow_html=True

    )



    if system_ready:



        total_review=len(df)



        app_count="-"



        for col in [

            "app",

            "application",

            "aplikasi",

            "nama_app"

        ]:


            if col in df.columns:


                app_count=df[col].nunique()

                break





        c1,c2,c3,c4=st.columns(4)




        with c1:


            st.markdown(

            f"""

            <div class="card">

            <div class="card-value">

            {total_review:,}

            </div>


            <div class="card-label">

            Total Review

            </div>

            </div>

            """,

            unsafe_allow_html=True

            )





        with c2:


            st.markdown(

            f"""

            <div class="card">

            <div class="card-value">

            {app_count}

            </div>


            <div class="card-label">

            Jumlah Aplikasi

            </div>

            </div>

            """,

            unsafe_allow_html=True

            )





        with c3:


            st.markdown(

            """

            <div class="card">

            <div class="card-value">

            90.35%

            </div>


            <div class="card-label">

            Accuracy

            </div>

            </div>

            """,

            unsafe_allow_html=True

            )





        with c4:


            st.markdown(

            """

            <div class="card">

            <div class="card-value">

            SVM

            </div>


            <div class="card-label">

            Best Model

            </div>

            </div>

            """,

            unsafe_allow_html=True

            )





        st.divider()



        st.subheader(

            "Preview Dataset"

        )



        st.dataframe(

            df.head(10),

            width="stretch"

        )
# ==========================================================
# STEP 2
# DASHBOARD VISUALIZATION FINAL
# ==========================================================



def find_column(df, candidates):

    for col in candidates:

        if col in df.columns:

            return col

    return None





# ==========================================================
# DASHBOARD ANALYTICS
# ==========================================================


if menu == "🏠 Dashboard":



    if system_ready:



        st.divider()



        st.subheader(

            "📊 Analisis Dataset"

        )




        # --------------------------------------------------
        # DETEKSI KOLOM
        # --------------------------------------------------


        sentiment_col = find_column(

            df,

            [

                "sentiment",

                "sentiment_final",

                "sentiment_lexicon",

                "label",

                "kelas"

            ]

        )



        app_col = find_column(

            df,

            [

                "app",

                "application",

                "aplikasi",

                "nama_app"

            ]

        )



        rating_col = find_column(

            df,

            [

                "rating",

                "score",

                "Rating"

            ]

        )



        text_col = find_column(

            df,

            [

                "review",

                "content",

                "text",

                "ulasan"

            ]

        )





        # --------------------------------------------------
        # SENTIMENT DISTRIBUTION
        # --------------------------------------------------


        if sentiment_col:



            st.subheader(

                "😊 Distribusi Sentimen"

            )



            sentiment_df = (

                df[sentiment_col]

                .astype(str)

                .value_counts()

                .reset_index()

            )



            sentiment_df.columns=[

                "Sentimen",

                "Jumlah"

            ]



            fig_sentiment = px.pie(

                sentiment_df,

                names="Sentimen",

                values="Jumlah",

                hole=0.45,

                title="Distribusi Sentimen"

            )



            st.plotly_chart(

                fig_sentiment,

                width="stretch"

            )




        else:



            st.warning(

                "Kolom sentimen tidak ditemukan"

            )





        # --------------------------------------------------
        # RATING DISTRIBUTION
        # --------------------------------------------------


        if rating_col:



            st.subheader(

                "⭐ Distribusi Rating"

            )



            rating_df = (

                df[rating_col]

                .value_counts()

                .sort_index()

                .reset_index()

            )



            rating_df.columns=[

                "Rating",

                "Jumlah"

            ]



            fig_rating = px.bar(

                rating_df,

                x="Rating",

                y="Jumlah",

                text="Jumlah",

                title="Distribusi Rating"

            )



            st.plotly_chart(

                fig_rating,

                width="stretch"

            )





        # --------------------------------------------------
        # APPLICATION COMPARISON
        # --------------------------------------------------


        if app_col and sentiment_col:



            st.subheader(

                "📱 Perbandingan Sentimen Aplikasi"

            )



            app_df=(

                df.groupby(

                    [

                        app_col,

                        sentiment_col

                    ]

                )

                .size()

                .reset_index(

                    name="Jumlah"

                )

            )



            fig_app = px.bar(

                app_df,

                x=app_col,

                y="Jumlah",

                color=sentiment_col,

                barmode="group",

                title="Sentimen Berdasarkan Aplikasi"

            )



            st.plotly_chart(

                fig_app,

                width="stretch"

            )





        # --------------------------------------------------
        # WORDCLOUD
        # --------------------------------------------------


        if text_col:



            st.subheader(

                "☁ WordCloud Review"

            )



            all_text = " ".join(

                df[text_col]

                .dropna()

                .astype(str)

            )



            if len(all_text)>0:



                wc = WordCloud(

                    width=1000,

                    height=400,

                    background_color="white",

                    max_words=100

                ).generate(all_text)



                fig_wc,ax = plt.subplots(

                    figsize=(12,5)

                )



                ax.imshow(

                    wc,

                    interpolation="bilinear"

                )



                ax.axis(

                    "off"

                )



                st.pyplot(

                    fig_wc

                )





        # --------------------------------------------------
        # DATA SUMMARY
        # --------------------------------------------------


        st.subheader(

            "📌 Statistik Dataset"

        )



        col1,col2,col3 = st.columns(3)




        with col1:



            st.metric(

                "Jumlah Data",

                len(df)

            )





        with col2:



            st.metric(

                "Jumlah Kolom",

                len(df.columns)

            )





        with col3:



            st.metric(

                "Missing Value",

                int(

                    df.isna()

                    .sum()

                    .sum()

                )

            )
# ==========================================================
# STEP 3
# PREDIKSI SENTIMEN FINAL
# ==========================================================


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)

from nltk.tokenize import word_tokenize




# ==========================================================
# NLP COMPONENT
# ==========================================================


stem_factory = StemmerFactory()

stemmer = stem_factory.create_stemmer()



stop_factory = StopWordRemoverFactory()

stop_words = set(

    stop_factory.get_stop_words()

)



stop_words.update([

    "yg",

    "nya",

    "aja",

    "sih",

    "nih",

    "kok",

    "udah"

])




# ==========================================================
# SLANG NORMALIZATION
# ==========================================================


slang_dict = {


    "gak":"tidak",

    "ga":"tidak",

    "gk":"tidak",

    "nggak":"tidak",

    "ngga":"tidak",

    "tdk":"tidak",


    "yg":"yang",

    "bgt":"banget",

    "udh":"sudah",

    "dgn":"dengan",

    "dr":"dari"


}





# ==========================================================
# PREPROCESS FUNCTION
# ==========================================================


def preprocess_text(text):


    detail={}



    detail["Input"] = text



    # lowercase

    text=str(text).lower()



    # remove url

    text=re.sub(

        r"http\S+",

        "",

        text

    )



    # remove non alphabet

    text=re.sub(

        r"[^a-zA-Z\s]",

        " ",

        text

    )



    # remove extra space

    text=re.sub(

        r"\s+",

        " ",

        text

    ).strip()



    detail["Cleaning"]=text




    # normalisasi slang


    words=text.split()



    words=[

        slang_dict.get(

            word,

            word

        )

        for word in words

    ]



    text=" ".join(words)



    detail["Normalisasi"]=text





    # tokenisasi


    try:


        tokens=word_tokenize(text)


    except:


        tokens=text.split()



    detail["Tokenisasi"]=tokens





    # stopword removal


    tokens=[

        word

        for word in tokens

        if word not in stop_words

    ]



    detail["Stopword Removal"]=tokens





    # stemming


    stemmed=[

        stemmer.stem(word)

        for word in tokens

    ]



    detail["Stemming"]=stemmed




    final_text=" ".join(stemmed)



    detail["Final Text"]=final_text



    return detail





# ==========================================================
# PREDIKSI PAGE
# ==========================================================


if menu=="🤖 Prediksi":



    st.markdown(

    """

    <div class="main-title">

    🤖 Prediksi Sentimen

    </div>

    """,

    unsafe_allow_html=True

    )



    st.write(

    """
    Masukkan ulasan pengguna.
    Sistem akan melakukan preprocessing
    dan klasifikasi menggunakan model SVM.
    """

    )




    review_input = st.text_area(

        "Review Pengguna",

        height=150,

        placeholder=

        "Contoh: aplikasi sangat membantu dan mudah digunakan"

    )





    if st.button(

        "🔍 Analisis Sentimen"

    ):



        if review_input.strip()=="":



            st.warning(

                "Review masih kosong"

            )



        else:



            with st.spinner(

                "Melakukan preprocessing..."

            ):



                # preprocessing

                result = preprocess_text(

                    review_input

                )



                final_text=result[

                    "Final Text"

                ]




                # TF-IDF

                vector=tfidf.transform(

                    [

                        final_text

                    ]

                )





                # model prediction

                pred=model.predict(

                    vector

                )




                # angka -> label

                sentiment=encoder.inverse_transform(

                    pred

                )[0]





            st.divider()



            # OUTPUT


            if str(sentiment).lower()=="positif":


                st.success(

                    f"🟢 SENTIMEN : {sentiment.upper()}"

                )



            elif str(sentiment).lower()=="negatif":


                st.error(

                    f"🔴 SENTIMEN : {sentiment.upper()}"

                )



            else:


                st.warning(

                    f"🟡 SENTIMEN : {sentiment.upper()}"

                )




            st.divider()



            st.subheader(

                "🔎 Detail Preprocessing"

            )



            for key,value in result.items():



                with st.expander(key):


                    st.write(value)




            st.divider()



            st.info(

            """

            Model:

            TF-IDF + Support Vector Machine


            Pipeline:

            Review

            ↓

            Cleaning

            ↓

            Normalisasi

            ↓

            Tokenisasi

            ↓

            Stopword Removal

            ↓

            Stemming

            ↓

            TF-IDF

            ↓

            SVM

            ↓

            Sentimen


            """

            )
# ==========================================================
# STEP 4
# EVALUASI MODEL FINAL
# ==========================================================


# ==========================================================
# FILE RESULT PATH
# ==========================================================


CONFUSION_PATH = os.path.join(

    RESULT_DIR,

    "confusion_matrix.png"

)



COMPARISON_PATH = os.path.join(

    RESULT_DIR,

    "comparison_model.csv"

)



REPORT_PATH = os.path.join(

    RESULT_DIR,

    "classification_report.txt"

)




# ==========================================================
# EVALUATION PAGE
# ==========================================================


if menu=="📈 Evaluasi":



    st.markdown(

    """

    <div class="main-title">

    📈 Evaluasi Model

    </div>

    """,

    unsafe_allow_html=True

    )



    st.write(

    """

    Evaluasi performa model berdasarkan
    hasil eksperimen machine learning.

    """

    )



    # ======================================================
    # COMPARISON MODEL
    # ======================================================


    st.subheader(

        "🏆 Perbandingan Model"

    )



    if os.path.exists(COMPARISON_PATH):



        comparison_df = pd.read_csv(

            COMPARISON_PATH

        )



        st.dataframe(

            comparison_df,

            width="stretch"

        )



        # grafik otomatis

        numeric_cols = comparison_df.select_dtypes(

            include=np.number

        ).columns



        if len(numeric_cols)>0:



            metric=numeric_cols[0]



            fig_compare = px.bar(

                comparison_df,

                x=comparison_df.columns[0],

                y=metric,

                title="Perbandingan Performa Model"

            )



            st.plotly_chart(

                fig_compare,

                width="stretch"

            )



    else:


        st.warning(

            "comparison_model.csv tidak ditemukan"

        )





    st.divider()





    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================


    st.subheader(

        "📋 Classification Report"

    )



    if os.path.exists(REPORT_PATH):



        with open(

            REPORT_PATH,

            "r",

            encoding="utf-8"

        ) as f:


            report=f.read()



        st.code(

            report

        )



    else:


        st.info(

            "classification_report.txt belum tersedia"

        )





    st.divider()





    # ======================================================
    # CONFUSION MATRIX
    # ======================================================


    st.subheader(

        "🎯 Confusion Matrix"

    )



    if os.path.exists(CONFUSION_PATH):


        st.image(

            CONFUSION_PATH,

            caption="Confusion Matrix SVM"

        )



    else:


        st.warning(

            "confusion_matrix.png tidak ditemukan"

        )
# ==========================================================
# STEP 5
# FINAL FEATURE
# BATCH PREDICTION + ABOUT
# ==========================================================



# ==========================================================
# BATCH PREDICTION
# ==========================================================


if menu=="📂 Batch Prediction":



    st.markdown(

    """

    <div class="main-title">

    📂 Batch Prediction

    </div>

    """,

    unsafe_allow_html=True

    )



    st.write(

    """

    Upload file CSV yang berisi kumpulan review.
    Sistem akan memprediksi sentimen secara otomatis.

    """

    )



    uploaded_file = st.file_uploader(

        "Upload CSV",

        type=["csv"]

    )



    if uploaded_file:



        upload_df = pd.read_csv(

            uploaded_file,

            low_memory=False

        )



        st.subheader(

            "Preview Data"

        )



        st.dataframe(

            upload_df.head(),

            width="stretch"

        )



        # cari kolom review

        review_col=find_column(

            upload_df,

            [

                "review",

                "content",

                "text",

                "ulasan"

            ]

        )



        if review_col is None:



            st.error(

                "Kolom review tidak ditemukan"

            )



        else:



            if st.button(

                "🚀 Mulai Prediksi"

            ):



                predictions=[]



                progress=st.progress(0)



                total=len(upload_df)



                for i,text in enumerate(

                    upload_df[review_col]

                ):



                    try:



                        result=preprocess_text(

                            str(text)

                        )



                        vector=tfidf.transform(

                            [

                                result["Final Text"]

                            ]

                        )



                        pred=model.predict(

                            vector

                        )



                        label=encoder.inverse_transform(

                            pred

                        )[0]



                        predictions.append(

                            label

                        )



                    except:



                        predictions.append(

                            "error"

                        )



                    progress.progress(

                        (i+1)/total

                    )





                upload_df["prediction"]=predictions




                st.success(

                    "Prediksi selesai"

                )



                st.subheader(

                    "Hasil Prediksi"

                )



                st.dataframe(

                    upload_df,

                    width="stretch"

                )





                # download



                csv_result=upload_df.to_csv(

                    index=False

                )



                st.download_button(

                    label="⬇ Download Hasil",

                    data=csv_result,

                    file_name="hasil_prediksi_sentimen.csv",

                    mime="text/csv"

                )

# ==========================================================
# ABOUT PAGE SIMPLE
# ==========================================================


if menu=="ℹ️ Tentang":


    st.title("ℹ️ Tentang Aplikasi")


    st.write(
        """
        ## Dashboard Analisis Sentimen Dompet Digital Indonesia

        Aplikasi ini digunakan untuk melakukan analisis sentimen
        terhadap ulasan pengguna aplikasi dompet digital
        menggunakan metode Machine Learning.
        """
    )



    st.divider()



    # Output

    st.subheader(
        "📊 Output Sistem"
    )


    col1,col2 = st.columns(2)



    with col1:

        st.success(
        """
        😊

        ## Sentimen Positif


        Contoh kata:

        - bagus
        - membantu
        - mudah
        - cepat

        """
        )



    with col2:

        st.error(
        """
        😡

        ## Sentimen Negatif


        Contoh kata:

        - buruk
        - error
        - gagal
        - lambat

        """
        )



    st.divider()



    st.subheader(
        "📌 Informasi Model"
    )


    col1,col2,col3 = st.columns(3)



    with col1:

        st.metric(
            "Model",
            "TF-IDF + SVM"
        )


    with col2:

        st.metric(
            "Akurasi",
            "90.35%"
        )


    with col3:

        st.metric(
            "Framework",
            "Streamlit"
        )
