import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, LabelEncoder, OrdinalEncoder, OneHotEncoder 
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_object_dtype, is_categorical_dtype
import warnings
from scipy import stats
import base64
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.cluster import KMeans
import xgboost as xgb
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from interpret import show
import os
import shutil
import zipfile
import tempfile
import plotly.io as pio
from datetime import datetime
import plotly.graph_objects as go

try:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
except ImportError:
    ExplainableBoostingClassifier = None
    ExplainableBoostingRegressor = None

from sklearn.utils.multiclass import unique_labels

def get_cleaned_df_copy():
    df = st.session_state.get("cleaned_df", None)
    return df.copy() if df is not None else None





def get_clickable_markdown_download_link(df, text="Download Processed Dataset", filename="cleaned_dataset.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"""
    <a href="data:file/csv;base64,{b64}" 
       download="{filename}" 
       style="text-decoration: none; font-size: 1.1rem; color: #00cfff;">
       {text}
    </a>
    """
    return href

def inject_dark_theme_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Set page config
st.set_page_config(
    page_title="AutoNexus: Automate. Explore. Model. Explain.",
    layout="wide",
    initial_sidebar_state="expanded"
)



inject_dark_theme_css()

# Session state initialization
if "started" not in st.session_state:
    st.session_state["started"] = False
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = 0
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None

if not st.session_state["started"]:
    st.session_state["active_tab"] = 0



# --------- LANDING PAGE ----------
if not st.session_state["started"] and st.session_state.active_tab == 0:
    st.markdown(
        """
        <div style='text-align: center; padding-top: 150px;'>
            <h1 style='font-size: 6em; font-weight: bold;'>AutoNexus</h1>
            <p style='font-size: 1.3em; max-width: 600px; margin: 0 auto;'>
                Upload. Explore. Model. Explain.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Get Started", use_container_width=True):
            st.session_state["started"] = True
            st.rerun()

    # Signature footer
    st.markdown("""
        <div style='position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
                color: gray; font-size: 0.9em; text-align: center;'>
            © Srilekha Tirumala Vinjamoori
        </div>
    """, unsafe_allow_html=True)

    st.stop()


if st.session_state["started"]:
    # Optional: Home button visible only after Get Started
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("Home"):
            st.session_state["started"] = False
            st.session_state["active_tab"] = 0
            st.rerun()

# -------- MAIN APP WITH TABS --------
tab_labels = [
    "Upload, Preview & Clean",
    "Exploratory Data Analysis",
    "Preprocessing & Export",
    "Modeling",
    "Explainability"
]

# Show tab name
st.markdown(f"## {tab_labels[st.session_state.active_tab]}")

# ---- Tab 0 ----
if st.session_state.active_tab == 0:

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file to get started", type=["csv"])

    # Reset session if file is removed
    if "uploaded_file" in st.session_state and uploaded_file is None:
        st.session_state.clear()
        st.rerun()

    # Load file only if new upload
    if uploaded_file is not None and uploaded_file != st.session_state.get("uploaded_file"):
        st.session_state.uploaded_file = uploaded_file

        try:
            df = pd.read_csv(uploaded_file)
            if df.empty or df.shape[1] == 0:
                st.error("Uploaded file is empty or has no columns.")
                st.stop()
            df = df.convert_dtypes().infer_objects()

        except Exception as e:
            st.error(f"Failed to read the CSV file: {e}")
            st.stop()

        st.session_state["df"] = df
        st.session_state["cleaned_df"] = df.copy()
        st.session_state["original_df"] = df.copy()

        # Warn if the dataset is large
        if df.shape[0] > 5000:
            st.warning(f"Your dataset has {df.shape[0]} rows. Previews and processing may be slower.")

    # Access dataset or show fallback
    df = st.session_state.get("df", None)
    df_cleaned = st.session_state.get("cleaned_df", None)

    # -------------------------------
    # Data Preview
    # -------------------------------
    st.subheader("Data Preview")

    if df is None:
        st.warning("No dataset uploaded")
    else:
        col1, col2 = st.columns([6, 1])
        with col2:
            show_full_data = st.checkbox("Show full dataset", value=False)

        st.dataframe(df if show_full_data else df.head(50))
        st.write(f"Shape of dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # -------------------------------
    # Data Cleaning
    # -------------------------------
    st.subheader("Data Cleaning")

    cleaning_task = None
    if df_cleaned is None:
        st.info("Upload a dataset to enable cleaning options.")
    else:
        with st.expander("Data Cleaning Tasks"):
            cleaning_task = st.selectbox("Select a cleaning task:", [
                "No Action", "Remove Duplicate Rows", "Drop Unnecessary Columns", "Rename Columns",
                "Standardize Null-Like Text", "Convert Comma-Separated Numbers to Numeric",
                "Trim & Standardize Text Columns"
            ])

        if cleaning_task == "Remove Duplicate Rows":
            total_dupes = df_cleaned.duplicated().sum()
            if total_dupes == 0:
                st.success("No exact duplicate rows found.")
            else:
                st.warning(f"Found {total_dupes} exact duplicate rows.")

                with st.expander("Preview duplicate rows"):
                    st.dataframe(df_cleaned[df_cleaned.duplicated(keep=False)].head(10))

                dup_mode = st.radio("Method:", [
                    "Remove exact duplicates (all columns)",
                    "Remove based on specific columns"
                ])

                keep_option = st.selectbox("Keep which duplicate?", ["first", "last", "none"])
                keep_value = False if keep_option == "none" else keep_option

                if dup_mode == "Remove exact duplicates (all columns)":
                    if st.button("Remove Duplicates"):
                        df_cleaned.drop_duplicates(keep=keep_value, inplace=True)
                        st.success("Duplicates removed.")
                else:
                    subset_cols = st.multiselect("Select columns to define duplicates:", df_cleaned.columns)
                    if st.button("Remove Duplicates by Subset") and subset_cols:
                        before = df_cleaned.shape[0]
                        df_cleaned.drop_duplicates(subset=subset_cols, keep=keep_value, inplace=True)
                        after = df_cleaned.shape[0]
                        st.success(f"Removed {before - after} duplicates based on selected columns.")
                
            

        elif cleaning_task == "Drop Unnecessary Columns":
            cols_to_drop = st.multiselect("Select columns to drop:", df_cleaned.columns, key = "drop_cols") 
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if cols_to_drop and st.button("Drop Selected Columns"):
                    df_cleaned.drop(columns=cols_to_drop, inplace=True)
                    st.session_state["cleaned_df"] = df_cleaned.copy()  # save updated df
                    st.success(f"Dropped: {', '.join(cols_to_drop)}. New shape: {df_cleaned.shape}")
                    st.rerun()

            with col2:
                if st.button("Reset Drop Action"):
                    if "drop_cols" in st.session_state:
                        del st.session_state["drop_cols"]

                    # ❗ Reset df_cleaned to the original uploaded dataset
                    if "df" in st.session_state:
                        st.session_state["cleaned_df"] = st.session_state["df"].copy()

                    st.rerun()

        elif cleaning_task == "Rename Columns":
            st.markdown("Edit the column names below:")
            rename_map = {}
            for col in df_cleaned.columns:
                new_name = st.text_input(f"Rename `{col}` to:", value=col, key=f"rename_{col}")
                if new_name and new_name != col:
                    rename_map[col] = new_name
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if rename_map and st.button("Apply Renaming"):
                    df_cleaned.rename(columns=rename_map, inplace=True)
                    st.session_state["cleaned_df"] = df_cleaned.copy()
                    st.success("Renaming applied.")

            with col2:
                if st.button("Reset Renaming"):
                    for col in df_cleaned.columns:
                        rename_key = f"rename_{col}"
                        if rename_key in st.session_state:
                            del st.session_state[rename_key]
                    if "df" in st.session_state:
                        st.session_state["cleaned_df"] = st.session_state["df"].copy()
                    st.rerun()

        elif cleaning_task == "Standardize Null-Like Text":
            text_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype == 'object' or df_cleaned[col].dtype.name == 'string']
            selected_cols = st.multiselect("Select columns to normalize null-like values:", text_cols)

            if st.button("Replace Common Null Tokens with NaN"):
                for col in selected_cols:
                    df_cleaned[col] = df_cleaned[col].replace(
                        ["", "None", "none", "NA", "N/A", "na", "null", "NULL", "--"],
                        np.nan
                    )
                st.success("Common null placeholders replaced with NaN.")
                st.dataframe(df_cleaned[selected_cols].head())

        elif cleaning_task == "Convert Comma-Separated Numbers to Numeric":
            candidate_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype == 'object']
            selected_cols = st.multiselect("Select columns to clean & convert to numeric:", candidate_cols)

            if st.button("Convert Selected Columns"):
                for col in selected_cols:
                    try:
                        df_cleaned[col] = df_cleaned[col].str.replace(",", "")
                        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
                    except Exception as e:
                        st.warning(f"Failed to convert {col}: {e}")
                st.success("Selected columns converted to numeric.")
                st.dataframe(df_cleaned[selected_cols].head())

        elif cleaning_task == "Trim & Standardize Text Columns":
            text_cols = [
                col for col in df_cleaned.columns
                if pd.api.types.is_string_dtype(df_cleaned[col]) or pd.api.types.is_object_dtype(df_cleaned[col])
                or pd.api.types.is_categorical_dtype(df_cleaned[col])
            ]

            st.write("Detected text columns:", text_cols)

            if not text_cols:
                st.info("No text columns available for cleaning.")
            else:
                selected_cols = st.multiselect("Select text columns to clean:", text_cols)
                case_option = st.radio("Choose case transformation:", ["lowercase", "uppercase", "capitalize"])

                if selected_cols and st.button("Apply Cleaning"):
                    for col in selected_cols:
                        df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                        if case_option == "lowercase":
                            df_cleaned[col] = df_cleaned[col].str.lower()
                        elif case_option == "uppercase":
                            df_cleaned[col] = df_cleaned[col].str.upper()
                        elif case_option == "capitalize":
                            df_cleaned[col] = df_cleaned[col].str.capitalize()

                    st.success("Whitespace trimmed and casing standardized.")
                    st.dataframe(df_cleaned[selected_cols].head())

        st.session_state.cleaned_df = df_cleaned.copy()
        st.dataframe(df_cleaned.head(50))


# ---- Tab 1 ----
elif st.session_state.active_tab == 1:
    
    # Place your tab2 logic here
    df_cleaned = get_cleaned_df_copy()

    st.subheader("Dataset Summary")

    if df_cleaned is None or df_cleaned.empty:
        st.warning("Upload a dataset to enable EDA.")
    else:
        st.dataframe(df_cleaned.dtypes.astype(str).rename("Data Type"))

        # 🔹 Missing Value Summary
        st.subheader("Handle Missing Values by Column")

        df_cleaned = st.session_state.get("cleaned_df", None)

        if df_cleaned is not None:
            # Get columns with missing values
            missing = df_cleaned.isnull().sum()
            missing = missing[missing > 0]

            if not missing.empty:
                st.write("Here’s a summary of missing values:")
                st.dataframe(missing.rename("Missing Values"))

                for col in missing.index:
                    col_dtype = df_cleaned[col].dtype
                    st.markdown(f"#### {col} ({col_dtype})")

                    # Show sample values
                    st.write("Example values:", df_cleaned[col].dropna().unique()[:5])

                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        strategy = st.selectbox(f"Choose strategy for `{col}`",
                                                ["Fill with mean", "Fill with median", "Drop rows (only if missing in this column)"],
                                                key=f"strat_{col}")
                        if st.button(f"Apply to `{col}`", key=f"btn_{col}"):
                            if strategy == "Fill with mean":
                                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                            elif strategy == "Fill with median":
                                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                            elif strategy == "Drop rows (only if missing in this column)":
                                df_cleaned.dropna(subset=[col], inplace=True)
                            st.success(f"Missing values in `{col}` handled with **{strategy}**.")
                    else:
                        strategy = st.selectbox(f"Choose strategy for `{col}`",
                                                ["Fill with mode", "Fill with 'Unknown'", "Drop rows (only if missing in this column)"],
                                                key=f"strat_{col}")
                        if st.button(f"Apply to `{col}`", key=f"btn_{col}"):
                            if strategy == "Fill with mode":
                                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                            elif strategy == "Fill with 'Unknown'":
                                df_cleaned[col].fillna("Unknown", inplace=True)
                            elif strategy == "Drop rows (only if missing in this column)":
                                df_cleaned.dropna(subset=[col], inplace=True)
                            st.success(f"Missing values in `{col}` handled with **{strategy}**.")

                # Update session state
                st.session_state["cleaned_df"] = df_cleaned

            else:
                st.success("No missing values found in the dataset.")




        # 🔹 Categorical and Numerical Columns
        st.subheader("Column Breakdown")
        cat_cols = [col for col in df_cleaned.columns if is_object_dtype(df_cleaned[col]) or isinstance(df_cleaned[col].dtype, pd.CategoricalDtype) or df_cleaned[col].dtype.name == "string"]
        num_cols = [col for col in df_cleaned.columns if is_numeric_dtype(df_cleaned[col])]
        if len(num_cols) == 0:
            st.warning("No numerical columns detected. Visualizations and modeling may be limited.")
        if len(cat_cols) == 0:
            st.warning("No categorical columns detected. Encoding and some EDA features may be unavailable.")
        st.write(f"Numerical Columns ({len(num_cols)}):", num_cols)
        st.write(f"Categorical Columns ({len(cat_cols)}):", cat_cols)

        # 🔹 Descriptive Statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(df_cleaned.describe(include='all').astype(str))

        # 🔹 Correlation Heatmap
        st.subheader("Correlation Heatmap (Numerical Features Only)")
        valid_numeric = df_cleaned[num_cols].dropna(axis=1, how='all')
        valid_numeric = valid_numeric.loc[:, valid_numeric.nunique() > 1]
        if valid_numeric.shape[1] < 2:
            st.warning("Not enough valid numeric columns to generate correlation heatmap.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(valid_numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # -------------------------------
    # Visual EDA Section
    # -------------------------------
    st.subheader("Visual EDA")

    if df_cleaned is None or df_cleaned.empty:
        st.info("Upload and clean a dataset to enable visualizations.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            plot_type = st.selectbox("Select Plot Type", [
                "Histogram", "Boxplot", "Countplot", "Scatter Plot", "Line Plot", "Pairplot", "Heatmap"
            ])
            dropdown_count = 1
            numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
            cat_cols = df_cleaned.select_dtypes(include=['object', 'category', 'string']).columns

            # Column selectors based on plot type
            if plot_type in ["Histogram", "Boxplot"]:
                if len(numeric_cols) == 0:
                    st.warning("No numeric columns available for this plot type.")
                    column_for_plot = None
                else:
                    column_for_plot = st.selectbox("Select column", numeric_cols)
                    dropdown_count += 1

            elif plot_type == "Countplot":
                if len(cat_cols) == 0:
                    st.warning("No categorical columns available.")
                    column_for_plot = None
                else:
                    column_for_plot = st.selectbox("Select column", cat_cols)
                    dropdown_count += 1

            elif plot_type in ["Scatter Plot", "Line Plot"]:
                if len(numeric_cols) < 2:
                    st.warning("Need at least two numeric columns.")
                    x_col, y_col = None, None
                else:
                    x_col = st.selectbox("X-axis", numeric_cols, key="x_axis")
                    y_col = st.selectbox("Y-axis", numeric_cols, key="y_axis")
                    dropdown_count += 2

        with col2:
            padding_px = 10 * dropdown_count
            st.markdown(f"<div style='margin-top: {padding_px}px;'>", unsafe_allow_html=True)

            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                if plot_type == "Histogram" and column_for_plot:
                    sns.histplot(df_cleaned[column_for_plot], kde=True, ax=ax)
                    ax.set_title(f"Histogram of {column_for_plot}")

                elif plot_type == "Boxplot" and column_for_plot:
                    sns.boxplot(x=df_cleaned[column_for_plot], ax=ax)
                    ax.set_title(f"Boxplot of {column_for_plot}")

                elif plot_type == "Countplot" and column_for_plot:
                    if df_cleaned[column_for_plot].nunique() > 50:
                        st.warning(f"'{column_for_plot}' has too many unique values. Plot may be unreadable.")
                    sns.countplot(x=df_cleaned[column_for_plot], ax=ax)
                    ax.set_title(f"Countplot of {column_for_plot}")
                    plt.xticks(rotation=45)

                elif plot_type == "Scatter Plot" and x_col and y_col:
                    sns.scatterplot(x=df_cleaned[x_col], y=df_cleaned[y_col], ax=ax)
                    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")

                elif plot_type == "Line Plot" and x_col and y_col:
                    sns.lineplot(x=df_cleaned[x_col], y=df_cleaned[y_col], ax=ax)
                    ax.set_title(f"Line Plot: {x_col} vs {y_col}")
                    plt.xticks(rotation=45)

                elif plot_type == "Heatmap":
                    sns.heatmap(valid_numeric.corr(), annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title("Correlation Heatmap")

                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating plot: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        if plot_type == "Pairplot":
            pair_data = df_cleaned.select_dtypes(include=['float64', 'int64'])
            if pair_data.shape[1] < 2:
                st.warning("Not enough numeric columns for pairplot.")
            elif pair_data.shape[1] > 8:
                st.warning("Too many numeric columns for pairplot. Please reduce the number of features.")
            else:
                try:
                    st.info("Generating pairplot...")
                    pairplot_fig = sns.pairplot(pair_data)
                    pairplot_fig.fig.set_size_inches(8, 6)
                    st.pyplot(pairplot_fig.fig)
                except Exception as e:
                    st.error(f"Error generating pairplot: {e}")

        st.session_state.cleaned_df = df_cleaned.copy()




# ---- Tab 2 ----
elif st.session_state.active_tab == 2:

    if "cleaned_df" not in st.session_state or st.session_state["cleaned_df"] is None:
        st.warning("Please upload and clean a dataset in Tab 0 before preprocessing.")
        st.stop()
    # 🧠 Start Tab 2 logic
    df_working = st.session_state["cleaned_df"].copy()

    # ✅ Place this message here — and ONLY here:
    st.info("You're working on a preview copy of the data. No changes are permanent until you click 'Apply All Preprocessing Steps'.")

    # -------------------------------
    # Feature Scaling
    # -------------------------------
    with st.expander("Feature Scaling", expanded=False):
        scaling_option = st.selectbox("Choose a scaling method:", [
            "No Action", "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer"
        ], key="scaling_option")

        try:
            if scaling_option != "No Action":
                scaler_map = {
                    "StandardScaler": StandardScaler(),
                    "MinMaxScaler": MinMaxScaler(),
                    "RobustScaler": RobustScaler(),
                    "MaxAbsScaler": MaxAbsScaler(),
                    "Normalizer": Normalizer()
                }
                scaler = scaler_map[scaling_option]
                numeric_cols = [col for col in df_working.columns if is_numeric_dtype(df_working[col])]
                if numeric_cols:
                    df_working[numeric_cols] = scaler.fit_transform(df_working[numeric_cols])
                    st.success(f"{scaling_option} applied to numeric columns.")
                    st.dataframe(df_working.head())
                else:
                    st.warning("No numeric columns found to scale.")
        except Exception as e:
            st.error(f"Scaling failed: {e}")

    # -------------------------------
    # Categorical Encoding
    # -------------------------------
    with st.expander("Categorical Encoding", expanded=False):
        cat_cols = [col for col in df_working.columns if is_object_dtype(df_working[col])
                    or isinstance(df_working[col].dtype, pd.CategoricalDtype)
                    or df_working[col].dtype.name == "string"]

        high_card_cols = [col for col in cat_cols if df_working[col].nunique() > 50]
        if high_card_cols:
            st.warning(f"High-cardinality columns: {', '.join(high_card_cols)}. Consider Label Encoding instead of One-Hot.")

        if not cat_cols:
            st.info("No categorical columns detected.")
        else:
            encoding_type = st.selectbox("Encoding method:", ["No Action", "Label Encoding", "Ordinal Encoding", "One-Hot Encoding"], key="encoding_type")
            selected_cols = st.multiselect("Columns to encode:", cat_cols, key="encoding_cols")

            try:
                for col in selected_cols:
                    if is_numeric_dtype(df_working[col]):
                        st.warning(f"Column {col} appears already numeric. Skipping.")
                        continue

                    if encoding_type == "Label Encoding":
                        le = LabelEncoder()
                        df_working[col] = le.fit_transform(df_working[col].astype(str))
                    elif encoding_type == "Ordinal Encoding":
                        custom_order = st.text_input(f"Order for `{col}` (comma-separated):", key=f"order_{col}")
                        categories = [x.strip() for x in custom_order.split(",")] if custom_order else sorted(df_working[col].dropna().unique())
                        oe = OrdinalEncoder(categories=[categories])
                        df_working[col] = oe.fit_transform(df_working[[col]].astype(str))
                    elif encoding_type == "One-Hot Encoding":
                        if df_working[col].nunique() > 50:
                            st.warning(f"Skipping `{col}` — too many categories.")
                            continue
                        df_working = pd.get_dummies(df_working, columns=[col], drop_first=True)
                if selected_cols and encoding_type != "No Action":
                    st.success("Encoding applied.")
                    st.dataframe(df_working.head())
            except Exception as e:
                st.error(f"Encoding failed: {e}")

    # -------------------------------
    # Outlier Detection & Handling
    # -------------------------------
    from scipy import stats
    with st.expander("Outlier Detection & Handling", expanded=False):
        numeric_cols = df_working.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.info("No numeric columns available for outlier detection.")
        else:
            method = st.radio("Choose outlier detection method:", ["IQR", "Z-Score"])
            selected_cols = st.multiselect("Select columns to check for outliers:", numeric_cols)
            action = st.radio("What should be done with detected outliers?", ["Remove", "Cap (Clip)"])

            if selected_cols:
                try:
                    if method == "IQR":
                        for col in selected_cols:
                            Q1 = df_working[col].quantile(0.25)
                            Q3 = df_working[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5 * IQR
                            upper = Q3 + 1.5 * IQR

                            if action == "Remove":
                                df_working = df_working[(df_working[col] >= lower) & (df_working[col] <= upper)]
                            elif action == "Cap (Clip)":
                                df_working[col] = df_working[col].clip(lower, upper)

                    elif method == "Z-Score":
                        threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0)
                        z_scores = np.abs(stats.zscore(df_working[selected_cols]))
                        mask = (z_scores < threshold).all(axis=1)

                        if action == "Remove":
                            df_working = df_working[mask]
                        elif action == "Cap (Clip)":
                            for col in selected_cols:
                                col_z = stats.zscore(df_working[col])
                                df_working[col] = np.where(
                                    np.abs(col_z) > threshold,
                                    np.sign(col_z) * threshold * df_working[col].std() + df_working[col].mean(),
                                    df_working[col]
                                )
                    st.success("Outlier handling applied successfully.")
                    st.dataframe(df_working.head())
                except Exception as e:
                    st.error(f"Outlier handling failed: {e}")
        
    # -------------------------------
    # Feature Selection (Supervised + Unsupervised)
    # -------------------------------
    with st.expander("Feature Selection (Supervised + Unsupervised)", expanded=False):
        st.markdown("You can apply both **unsupervised** (statistical) and **supervised** (target-aware) feature selection methods.")

        method_type = st.radio("Choose method type:", ["Unsupervised", "Supervised"])
        numeric_cols = df_working.select_dtypes(include=np.number).columns.tolist()

        if method_type == "Unsupervised":
            st.subheader("Unsupervised Filters")

            low_var_thresh = st.slider("Variance threshold (remove features below this)", 0.0, 1.0, 0.0)
            low_var_cols = []
            if low_var_thresh > 0:
                for col in numeric_cols:
                    if df_working[col].var() < low_var_thresh:
                        low_var_cols.append(col)
                if low_var_cols:
                    st.warning(f"Low variance columns: {low_var_cols}")
                    if st.button("Remove Low Variance Columns"):
                        df_working.drop(columns=low_var_cols, inplace=True)
                        st.success("Low variance columns removed.")

            corr_thresh = st.slider("Correlation threshold", 0.5, 1.0, 0.9)
            if corr_thresh < 1.0 and len(numeric_cols) > 1:
                corr_matrix = df_working[numeric_cols].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
                if to_drop:
                    st.warning(f"Highly correlated columns: {to_drop}")
                    if st.button("Remove Correlated Columns"):
                        df_working.drop(columns=to_drop, inplace=True)
                        st.success("Correlated columns removed.")

        elif method_type == "Supervised":
            st.subheader("Supervised Selection (requires target)")

            target_col = st.selectbox("Select target column:", df_working.columns)
            if not target_col:
                st.warning("Please select a target column.")
                st.stop()

            if df_working[target_col].isnull().sum() > 0:
                st.error("Target column has missing values. Handle them before applying feature selection.")
                st.stop()

            supervised_features = [col for col in df_working.columns if col != target_col and is_numeric_dtype(df_working[col])]
            selection_method = st.radio("Select method", ["ANOVA F-test", "Mutual Information"])
            k = st.slider("Number of top features to keep", 1, len(supervised_features), min(5, len(supervised_features)))

            try:
                X = df_working[supervised_features]
                y = df_working[target_col]

                if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                    st.error("Missing values in features or target. Please clean your data.")
                else:
                    if y.dtype in ['object', 'category', 'string']:
                        y = LabelEncoder().fit_transform(y.astype(str))

                    selector = SelectKBest(score_func=f_classif if selection_method == "ANOVA F-test" else mutual_info_classif, k=k)
                    selector.fit(X, y)
                    selected = X.columns[selector.get_support()].tolist()

                    st.success(f"Top {k} selected features: {selected}")
                    if st.button("Keep Selected Features Only"):
                        df_working = df_working[selected + [target_col]]
                        st.success("Filtered dataset to selected features.")
            except Exception as e:
                st.error(f"Feature selection failed: {e}")

        st.markdown("#### Preview After Feature Selection")
        st.dataframe(df_working.head())

    # -------------------------------
    # Feature Engineering
    # -------------------------------
    with st.expander("Feature Engineering", expanded=False):
        st.markdown("Transform existing columns or extract new ones to boost model performance.")

        numeric_cols = df_working.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            st.markdown("##### 🔢 Numerical Transformations")
            selected_numeric = st.multiselect("Select numeric columns to transform:", numeric_cols)

            if selected_numeric:
                transform_ops = st.multiselect("Select transformations to apply:", ["Square", "Square Root", "Log"])

                for col in selected_numeric:
                    if "Square" in transform_ops:
                        df_working[f"{col}_squared"] = df_working[col] ** 2
                    if "Square Root" in transform_ops:
                        df_working[f"{col}_sqrt"] = df_working[col].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)
                    if "Log" in transform_ops:
                        df_working[f"{col}_log"] = df_working[col].apply(lambda x: np.log1p(x) if x >= 0 else np.nan)
                st.success("Numeric transformations applied.")

        text_cols = [col for col in df_working.columns if is_object_dtype(df_working[col]) and df_working[col].nunique() < len(df_working)]
        if text_cols:
            st.markdown("##### 🔤 Text Features")
            selected_text_cols = st.multiselect("Select text columns to extract features from:", text_cols)

            for col in selected_text_cols:
                non_empty = df_working[col].dropna().astype(str).str.strip()
                if non_empty.eq("").all():
                    st.warning(f"Column `{col}` is empty or only missing. Skipping.")
                    continue
                df_working[f"{col}_char_count"] = df_working[col].astype(str).apply(len)
                df_working[f"{col}_word_count"] = df_working[col].astype(str).apply(lambda x: len(str(x).split()))
                df_working[f"{col}_avg_word_len"] = df_working[col].astype(str).apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)
            if selected_text_cols:
                st.success("Text features extracted.")
        st.dataframe(df_working.head())

    # -------------------------------
    # Drop High-Cardinality Non-Numeric Columns (ID-like)
    # -------------------------------
    with st.expander("Drop ID-like / High-Cardinality String Columns", expanded=False):
        text_cols = [col for col in df_working.columns if is_object_dtype(df_working[col]) or df_working[col].dtype.name in ['string', 'category']]
        id_like_cols = [col for col in text_cols if df_working[col].nunique() > 0.9 * len(df_working)]

        if id_like_cols:
            st.warning(f"High-cardinality string columns detected: {', '.join(id_like_cols)}")
            selected = st.multiselect("Select columns to drop", id_like_cols, default=id_like_cols)

            if selected and st.button("Drop Selected Columns"):
                df_working.drop(columns=selected, inplace=True)
                st.success("Selected columns dropped.")
                st.dataframe(df_working.head())
        else:
            st.info("No ID-like columns detected.")

    # -------------------------------
    # Class Imbalance Handling
    # -------------------------------

    with st.expander("Class Imbalance Handling", expanded=False):
        st.markdown("Handle datasets where one class heavily outnumbers another (like fraud detection).")

        all_cols = df_working.columns.tolist()
        target_col = st.selectbox("Select target column:", all_cols)

        # Detect classification task only
        if is_numeric_dtype(df_working[target_col]) and df_working[target_col].nunique() > 15:
            st.warning("This looks like a regression problem — imbalance handling is only for classification tasks.")
        else:
            imbalance_method = st.radio("Choose balancing method:", ["SMOTE", "Random OverSampling", "Random UnderSampling"])

            # Show class distribution before
            st.markdown("**Class distribution before balancing:**")
            st.write(dict(Counter(df_working[target_col])))

            if st.button("Apply Balancing"):
                try:
                    with st.spinner("Applying class balancing..."):
                        X = df_working.drop(columns=[target_col])
                        y = df_working[target_col]

                        # Convert categorical y if needed
                        if y.dtype == "object" or y.dtype.name == "category" or y.dtype == "string":
                            y = LabelEncoder().fit_transform(y)

                        if imbalance_method == "SMOTE":
                            sampler = SMOTE(random_state=42)
                        elif imbalance_method == "Random OverSampling":
                            sampler = RandomOverSampler(random_state=42)
                        else:
                            sampler = RandomUnderSampler(random_state=42)

                        X_resampled, y_resampled = sampler.fit_resample(X, y)

                        df_working = pd.DataFrame(X_resampled, columns=X.columns)
                        df_working[target_col] = y_resampled

                        st.success("Class balancing applied successfully.")
                        st.markdown("**Class distribution after balancing:**")
                        st.write(dict(Counter(y_resampled)))
                        st.dataframe(df_working.head())
                except Exception as e:
                    st.error(f"Balancing failed: {e}")


    with st.expander("Preview: Original vs Modified", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Cleaned Data:**")
            st.dataframe(st.session_state["cleaned_df"].head())

        with col2:
            st.markdown("**Modified Preprocessed Data:**")
            st.dataframe(df_working.head())
   
    #Final Apply of Pre-Processing Steps

    st.subheader("Finalize Preprocessing")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Apply All Preprocessing Steps"):
            # 🔁 Backup before overwrite
            st.session_state["cleaned_df_backup"] = st.session_state["cleaned_df"].copy()
            st.session_state["cleaned_df"] = df_working.copy()
            st.success("Changes have been applied to the cleaned dataset.")
            st.rerun()

    with col2:
        if st.button("Discard Changes"):
            st.warning("All changes discarded. Reloaded the last committed version.")
            st.rerun()

    with col3:
        if "cleaned_df_backup" in st.session_state:
            if st.button("Revert to Previous Committed Version"):
                st.session_state["cleaned_df"] = st.session_state["cleaned_df_backup"].copy()
                st.success("Restored previous version of the dataset.")
                st.rerun()

    
# TODO: Missing value handling, scaling, encoding, download


# -------------------------------
# Tab 3: Modeling
# -------------------------------


# ---- Tab 3 ----
elif st.session_state.active_tab == 3:
    
    # Place your tab4 logic here

    #st.subheader("Train & Evaluate Models")

    def clean_column_names(df):
        df = df.copy()
        df.columns = [re.sub(r"[^\w]", "_", col) for col in df.columns]
        return df

    df_model = st.session_state.get("cleaned_df")
    if df_model is not None:
        df_model = df_model.copy()

    if df_model is None:
        st.warning("Upload and Pre-Process a dataset to enable Modeling.")
    else:
        target_col = st.selectbox("Select Target Column", df_model.columns, key="tab4_target")
        if df_model[target_col].nunique() <= 1:
            st.error("The selected target column has only one unique value. Please choose another target or check your data.")
            st.stop()
        if df_model[target_col].isnull().sum() > 0:
            st.error("Target column contains missing values. Please impute or remove them in preprocessing.")
            st.stop()

        if df_model[target_col].dtype in ['object', 'category', 'bool', 'string'] or df_model[target_col].nunique() <= 10:
            task_type = "Classification"
        elif df_model[target_col].nunique() > 15:
            task_type = "Regression"
        else:
            task_type = st.radio("Select Task Type:", ["Classification", "Regression"])

        st.info(f"Detected Task Type: {task_type}")

        X = df_model.drop(columns=[target_col])
        y = df_model[target_col]

        if task_type == "Classification" and y.dtype in ['object', 'category', 'string']:
            y = LabelEncoder().fit_transform(y)
        else:
            y = y.to_numpy().squeeze()

        test_size = st.slider("Test Set Size (%)", 10, 90, 20)
        train_size = 100 - test_size
        st.write(f"Train Size: {train_size}%, Test Size: {test_size}%")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

        if task_type == "Classification":
            classifiers = [
                "Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier",
                "KNN", "Naive Bayes", "GaussianNB"
            ]
            if xgb:
                classifiers.insert(3, "XGBoost Classifier")
            if ExplainableBoostingClassifier:
                classifiers.append("Explainable Boosting Classifier (EBM)")
            model_choice = st.selectbox("Select Classification Model", classifiers)
        else:
            regressors = [
                "Linear Regression", "Lasso", "Ridge"
            ]
            if xgb:
                regressors.append("XGBoost Regressor")
            if ExplainableBoostingRegressor:
                regressors.append("Explainable Boosting Regressor (EBM)")
            model_choice = st.selectbox("Select Regression Model", regressors)

        # Optional enhancement: allow users to choose interaction count for EBM
        interaction_count = 10
        if "Boosting" in model_choice:
            interaction_count = st.slider("Max Interaction Terms (EBM only)", 0, 50, 10)

        model_map = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "GaussianNB": GaussianNB()
        }

        if xgb:
            model_map["XGBoost Classifier"] = xgb.XGBClassifier()
            model_map["XGBoost Regressor"] = xgb.XGBRegressor()
        if ExplainableBoostingClassifier:
            model_map["Explainable Boosting Classifier (EBM)"] = ExplainableBoostingClassifier(interactions=interaction_count, random_state=42)
        if ExplainableBoostingRegressor:
            model_map["Explainable Boosting Regressor (EBM)"] = ExplainableBoostingRegressor(interactions=interaction_count, random_state=42)

        model = model_map.get(model_choice)

        if st.button("Train Model"):
            try:
                if "XGBoost" in model_choice:
                    X_train = clean_column_names(X_train.astype(int))
                    X_test = clean_column_names(X_test.astype(int))

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.success(f"{model_choice} trained successfully!")
                st.markdown("### Model Evaluation")

                if task_type == "Classification":
                    metrics = {
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                        "Precision": precision_score(y_test, y_pred, zero_division=0),
                        "Recall": recall_score(y_test, y_pred, zero_division=0),
                        "MCC": matthews_corrcoef(y_test, y_pred)
                    }
                    for metric, score in metrics.items():
                        st.write(f"**{metric}:** {score:.3f}")

                    with st.expander("Full Classification Report"):
                        report_text = classification_report(y_test, y_pred, zero_division=0)
                        st.code(report_text, language="text")

                    with st.expander("Confusion Matrix", expanded=False):
                        try:
                            cm = confusion_matrix(y_test, y_pred)
                            labels = unique_labels(y_test, y_pred)

                            col1, col2 = st.columns([1, 2])
                            with col1:
                                fig, ax = plt.subplots(figsize=(4, 3))
                                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                                ax.set_title("Confusion Matrix")
                                ax.set_xlabel("Predicted Labels")
                                ax.set_ylabel("True Labels")
                                st.pyplot(fig)
                            with col2:
                                st.markdown("**Confusion Matrix Explanation**")
                                st.markdown("""
                                    - **Rows** → Actual values  
                                    - **Columns** → Predicted values  
                                    - Diagonal → Correct predictions  
                                    - Off-diagonal → Errors
                                """)

                        except Exception as e:
                            st.warning(f"Confusion matrix could not be displayed: {e}")

                else:
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                    r2 = r2_score(y_test, y_pred)

                    st.write(f"MAE: {mae:.2f}")
                    st.write(f"MSE: {mse:.2f}")
                    st.write(f"RMSE: {rmse:.2f}")
                    st.write(f"R² Score: {r2:.2f}")

                st.markdown("### Prediction Preview")
                pred_df = pd.DataFrame({
                    "Actual": y_test[:20],
                    "Predicted": y_pred[:20]
                })
                st.dataframe(pred_df)

                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                model_filename = model_choice.replace(" ", "_").lower() + "_model.pkl"
                st.download_button("Download Trained Model", buffer.getvalue(), file_name=model_filename)

            except Exception as e:
                st.error(f"Training failed: {e}")




# ---- Tab 4 ----
elif st.session_state.active_tab == 4:
    
    # Place your tab logic here
    #st.subheader("Model Explainability")

    df = st.session_state.get("cleaned_df")

    if df is None:
        st.warning("Upload a dataset to enable Explainability.")
        st.empty()  # keeps layout rendering going (so buttons can show)
    else:

        uploaded_model = st.file_uploader("Upload Trained Model (.pkl)", type=["pkl"])
        if uploaded_model:
            try:
                model = joblib.load(uploaded_model)
                st.success("Model loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()
        else:
            st.info("Upload a model trained from Tab 4.")
            st.stop()

        target_col = st.selectbox("Select Target Column", df.columns, key="explainability_target")
        if target_col is None:
            st.stop()

        X = df.drop(columns=[target_col])
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes
        X = X.astype("float64")

        y = df[target_col]
        if y.dtype in ["object", "category", "string"]:
            y = LabelEncoder().fit_transform(y)
        else:
            y = y.to_numpy().squeeze()

        X.columns = [str(col).replace("[", "_").replace("]", "_").replace("<", "_") for col in X.columns]

        if isinstance(model, (ExplainableBoostingClassifier, ExplainableBoostingRegressor)):
            st.subheader("EBM Global Explanation Plots")
            try:
                ebm_global = model.explain_global()
                plots = ebm_global.visualize()
                term_names = ebm_global.data()['names']

                if isinstance(plots, list):
                    st.success(f"{len(plots)} plots generated (main effects + interactions).")
                    for i, fig in enumerate(plots):
                        name = term_names[i]
                        is_interaction = isinstance(name, tuple)
                        title = "Interaction" if is_interaction else "Feature"
                        st.markdown(f"### {title} Plot {i + 1}: {name}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Only 1 plot generated — possibly no interaction terms detected.")
                    st.plotly_chart(plots, use_container_width=True)

            except Exception as e:
                st.error(f"Global explanation failed: {e}")


            except Exception as e:
                st.error(f"Export failed: {e}")


        else:
            st.subheader("SHAP Explanations")
            try:
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)

                st.markdown("#### SHAP Summary - Bar")
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.plots.bar(shap_values, max_display=10, show=False)
                st.pyplot(fig)

                st.markdown("#### SHAP - Waterfall Plot")
                try:
                    idx = st.slider("Select row index for Waterfall", 0, len(X) - 1, 0, key="shap_waterfall_idx")

                # Ensure SHAP values are from Explainer and have .values and .data
                    if hasattr(shap_values, "values") and hasattr(shap_values, "data"):
                        plt.clf()
                        shap.plots.waterfall(shap_values[idx], show=False)
                        st.pyplot(plt.gcf())
                    else:
                        st.warning("SHAP values are not explainer-based (may not support waterfall).")

                except Exception as e:
                    st.error(f"SHAP waterfall plot failed: {e}")
            
            
                st.markdown("#### SHAP Force Plot")
                idx = st.slider("Select row index", 0, len(X) - 1, 0, key="shap_force_idx")
                shap.initjs()
                plt.clf()
                shap.plots.force(shap_values[idx], matplotlib=True, show=False)
                st.pyplot(plt.gcf())

            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")

        st.subheader("LIME Local Explanation")
        lime_idx = st.slider("Select row index for LIME", 0, len(X) - 1, 0, key="lime_slider")
        if hasattr(model, "predict_proba"):
            try:
                lime_explainer = LimeTabularExplainer(
                    training_data=X.values,
                    feature_names=X.columns.tolist(),
                    class_names=np.unique(y).astype(str).tolist(),
                    mode='classification' if len(np.unique(y)) <= 10 else 'regression'
                )
                lime_exp = lime_explainer.explain_instance(X.iloc[lime_idx].values, model.predict_proba, num_features=10)
                st.components.v1.html(lime_exp.as_html(), height=400, scrolling=True)
            except Exception as e:
                st.error(f"LIME explanation failed: {e}")
        else:
            st.warning("Selected model does not support probability predictions required by LIME.")


# --- Fixed Position Buttons Using Streamlit Columns (Styled + Positioned) ---
# Define 5-column layout: Left spacer, Back button, center spacer, Next button, Right spacer
col_left_spacer, col_back, col_center_spacer, col_next, col_right_spacer = st.columns([1, 1, 6, 1, 1])

# Inject styles (optional override to match your style)
st.markdown("""
    <style>
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        font-weight: 500;
        font-size: 14px;
        border-radius: 8px;
        padding: 6px 16px;
        border: none;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ⬅️ Back Button (left-aligned)
with col_back:
    if st.session_state.active_tab > 0:
        if st.button("Back", key=f"back_{st.session_state.active_tab}"):
            st.session_state.active_tab -= 1
            st.rerun()

# ➡️ Next Button (right-aligned)
with col_next:
    if st.session_state.active_tab < len(tab_labels) - 1:
        if st.button("Next", key=f"next_{st.session_state.active_tab}"):
            st.session_state.active_tab += 1
            st.rerun()
    else:
        st.markdown("Done", unsafe_allow_html=True)
