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

# App Title & Intro

st.title("AutoNexus: Automate. Explore. Model. Explain.")



st.markdown(
    """

    Whether you're prepping data for machine learning or just making sense of a new dataset — this app gives you an intuitive, powerful interface to handle it all.

    **Clean. Format. Explore. Prep — in one seamless flow:**
    - Smart data cleaning  
    - Auto date handling  
    - Visual insights  
    - Encoding & scaling made simple
    
    _Give it a spin — your data deserves better._ 

    """
)

st.markdown("---")

# Initialize session state
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Upload, Preview & Clean",
    "Exploratory Data Analysis",
    "Preprocessing & Export",
    "Modeling",
    "Explainability",
])

with tab1:
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
        st.warning("No dataset uploaded yet. Please upload a CSV file above.")
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
                    keep_value = None if keep_option == "none" else keep_option

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
                cols_to_drop = st.multiselect("Select columns to drop:", df_cleaned.columns)
                if cols_to_drop and st.button("Drop Selected Columns"):
                    df_cleaned.drop(columns=cols_to_drop, inplace=True)
                    st.success(f"Dropped: {', '.join(cols_to_drop)}. New shape: {df_cleaned.shape}")

            elif cleaning_task == "Rename Columns":
                st.markdown("Edit the column names below:")
                rename_map = {}
                for col in df_cleaned.columns:
                    new_name = st.text_input(f"Rename `{col}` to:", value=col, key=f"rename_{col}")
                    if new_name and new_name != col:
                        rename_map[col] = new_name
                if rename_map and st.button("Apply Renaming"):
                    df_cleaned.rename(columns=rename_map, inplace=True)
                    st.success("Renaming applied.")

            
            elif cleaning_task == "Standardize Null-Like Text to NaN":
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
                # Detect text-like columns: object, string, or categorical
                text_cols = [
                    col for col in df_cleaned.columns
                    if pd.api.types.is_string_dtype(df_cleaned[col]) or pd.api.types.is_object_dtype(df_cleaned[col])
                    or pd.api.types.is_categorical_dtype(df_cleaned[col])
                ]

                st.write("Detected text columns:", text_cols)  # Optional: for debugging

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



with tab2:
    df_cleaned = get_cleaned_df_copy()

    st.subheader("Dataset Summary")

    if df_cleaned is None or df_cleaned.empty:
        st.warning("No dataset found. Please upload and clean a dataset in Tab 1 to enable EDA.")
    else:
        st.dataframe(df_cleaned.dtypes.astype(str).rename("Data Type"))

        # 🔹 Missing Value Summary
        st.subheader("Missing Values")
        missing = df_cleaned.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.dataframe(missing.rename("Missing Values"))
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




with tab3:
    st.subheader("Data Preprocessing")

    df_cleaned = get_cleaned_df_copy()

    if df_cleaned is None:
        st.warning("No dataset found. Please upload and clean your dataset in Tab 1.")
    else:
        # -------------------------------
        # Handle Missing Values
        # -------------------------------
        with st.expander("Handle Missing Values", expanded=False):
            missing_option = st.selectbox(
                "Choose a missing value strategy:",
                ["No Action", "Drop rows with missing values", "Impute missing values"]
            )

            if missing_option == "Drop rows with missing values":
                df_cleaned.dropna(inplace=True)
                after_shape = df_cleaned.shape
                st.success(f"Rows with missing values dropped. Shape - {after_shape}")

            elif missing_option == "Impute missing values":
                impute_method = st.radio("Select imputation method:", ["Mean", "Median", "Mode"])
                try:
                    for col in df_cleaned.columns:
                        if df_cleaned[col].isnull().sum() > 0:
                            if df_cleaned[col].dropna().shape[0] == 0:
                                st.warning(f"Column `{col}` has only missing values. Skipping imputation.")
                                continue
                            if is_numeric_dtype(df_cleaned[col]):
                                if impute_method == "Mean":
                                    val = df_cleaned[col].mean()
                                    if is_integer_dtype(df_cleaned[col]):
                                        val = int(round(val))
                                    df_cleaned[col] = df_cleaned[col].fillna(val)
                                elif impute_method == "Median":
                                    val = df_cleaned[col].median()
                                    if is_integer_dtype(df_cleaned[col]):
                                        val = int(round(val))
                                    df_cleaned[col] = df_cleaned[col].fillna(val)
                                elif impute_method == "Mode":
                                    val = df_cleaned[col].mode()[0]
                                    df_cleaned[col] = df_cleaned[col].fillna(val)
                            else:
                                val = df_cleaned[col].mode()[0]
                                df_cleaned[col] = df_cleaned[col].fillna(val)
                    st.success(f"Missing values imputed using {impute_method.lower()}.")
                except Exception as e:
                    st.error(f"Imputation failed: {str(e)}")

            # ✅ Show missing value summary after handling
            st.markdown("###### Missing Values After Handling")
            missing_after = df_cleaned.isnull().sum()
            missing_percent_after = (missing_after / len(df_cleaned)) * 100
            missing_df_after = pd.DataFrame({
                "Missing Values": missing_after,
                "Percent (%)": missing_percent_after
            })
            missing_df_after = missing_df_after[missing_df_after["Missing Values"] > 0]

            if not missing_df_after.empty:
                st.dataframe(missing_df_after.astype(str))
            else:
                st.success("No missing values remaining in the cleaned dataset!")

        if df_cleaned is not None:
            st.session_state.cleaned_df = df_cleaned.copy()

        # -------------------------------
        # Feature Scaling
        # -------------------------------
        with st.expander("Feature Scaling", expanded=False):
            scaling_option = st.selectbox("Choose a scaling method:", [
                "No Action", "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer"])

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
                    numeric_cols = [col for col in df_cleaned.columns if is_numeric_dtype(df_cleaned[col])]
                    if numeric_cols:
                        df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
                        st.success(f"Applied {scaling_option} to numeric columns.")
                        st.markdown("#### Preview After Scaling")
                        st.dataframe(df_cleaned.head())
                    else:
                        st.warning("No numeric columns found for scaling.")
            except Exception as e:
                st.error(f"Scaling failed: {str(e)}")
            
            if df_cleaned is not None:
                st.session_state.cleaned_df = df_cleaned.copy()

        # -------------------------------
        # Categorical Encoding
        # -------------------------------
        with st.expander("Categorical Encoding", expanded=False):
            cat_cols = [col for col in df_cleaned.columns if is_object_dtype(df_cleaned[col])
                        or isinstance(df_cleaned[col].dtype, pd.CategoricalDtype)
                        or df_cleaned[col].dtype.name == "string"]

            # Detect high-cardinality columns
            high_card_cols = [col for col in cat_cols if df_cleaned[col].nunique() > 50]
            if high_card_cols:
                st.warning(f"High-cardinality categorical columns detected: {', '.join(high_card_cols)}. "
                        "One-Hot Encoding may create too many columns. Consider Label Encoding instead.")

            if not cat_cols:
                st.info("No categorical columns detected.")
            else:
                selected_encoding = st.selectbox(
                    "Select encoding method:",
                    ["No Action", "Label Encoding", "Ordinal Encoding", "One-Hot Encoding"]
                )
                selected_columns = st.multiselect("Select categorical columns to encode:", cat_cols)

                if selected_encoding == "Ordinal Encoding":
                    st.markdown("*Optional: Enter custom order for each selected column (comma-separated). Leave blank for default alphabetical order.*")

                try:
                    for col in selected_columns:
                        # 🔒 Skip already encoded (numeric) columns
                        if is_numeric_dtype(df_cleaned[col]):
                            st.warning(f"Column `{col}` appears to be numeric already. Skipping encoding.")
                            continue

                        if selected_encoding == "Label Encoding":
                            le = LabelEncoder()
                            df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
                            st.write(f"`{col}` encoded with **Label Encoding**.")

                        elif selected_encoding == "Ordinal Encoding":
                            custom_order = st.text_input(f"Custom order for `{col}` (comma-separated)", key=f"order_{col}")
                            if custom_order:
                                categories = [x.strip() for x in custom_order.split(",")]
                                oe = OrdinalEncoder(categories=[categories])
                            else:
                                sorted_categories = sorted(df_cleaned[col].dropna().unique())
                                oe = OrdinalEncoder(categories=[sorted_categories])
                            df_cleaned[col] = oe.fit_transform(df_cleaned[[col]].astype(str))
                            st.write(f"`{col}` encoded with **Ordinal Encoding**.")

                        elif selected_encoding == "One-Hot Encoding":
                            if df_cleaned[col].nunique() > 50:
                                st.warning(f"Skipping `{col}` — it has {df_cleaned[col].nunique()} unique categories. One-hot encoding would create too many columns.")
                                continue
                            df_cleaned = pd.get_dummies(df_cleaned, columns=[col], drop_first=True)
                            st.write(f"`{col}` encoded with **One-Hot Encoding**.")

                    if selected_encoding != "No Action" and selected_columns:
                        st.success("Encoding completed.")
                        st.markdown("#### Preview After Encoding")
                        st.dataframe(df_cleaned.head(10))

                except Exception as e:
                    st.error(f"Encoding failed: {str(e)}")



        # ✅ Save the updates back to session state
        if df_cleaned is not None:
            st.session_state.cleaned_df = df_cleaned.copy()

        st.markdown(get_clickable_markdown_download_link(
        st.session_state.cleaned_df, 
        text="Download Cleaned Dataset"
        ), unsafe_allow_html=True)

        # -------------------------------
        # Outlier Detection & Handling
        # -------------------------------
        from scipy import stats

        with st.expander("Outlier Detection & Handling", expanded=False):
            st.markdown("""
            ℹ️ **About Outlier Detection Methods**

            - **IQR (Interquartile Range)**: Flags values that fall **1.5×IQR** outside the 25th–75th percentile range.
            - **Z-Score**: Flags values that are **more than X standard deviations** from the mean.

            > Use IQR for **skewed datasets**.  
            > Use Z-score for **normally distributed data**.
            """)

            numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()

            if not numeric_cols:
                st.info("No numeric columns available for outlier detection.")
            else:
                method = st.radio("Choose outlier detection method:", ["IQR", "Z-Score"])
                selected_cols = st.multiselect("Select columns to check for outliers:", numeric_cols)

                action = st.radio("What should be done with detected outliers?", ["Remove", "Cap (Clip)"])

                if selected_cols:
                    if method == "IQR":
                        for col in selected_cols:
                            Q1 = df_cleaned[col].quantile(0.25)
                            Q3 = df_cleaned[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            st.write(f"`{col}` → Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f}")

                            if action == "Remove":
                                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                            elif action == "Cap (Clip)":
                                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

                    elif method == "Z-Score":
                        threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0)
                        z_scores = np.abs(stats.zscore(df_cleaned[selected_cols]))
                        mask = (z_scores < threshold).all(axis=1)

                        if action == "Remove":
                            df_cleaned = df_cleaned[mask]
                        elif action == "Cap (Clip)":
                            for col in selected_cols:
                                col_z = stats.zscore(df_cleaned[col])
                                df_cleaned[col] = np.where(
                                    np.abs(col_z) > threshold,
                                    np.sign(col_z) * threshold * df_cleaned[col].std() + df_cleaned[col].mean(),
                                    df_cleaned[col]
                                )

                    st.success("Outlier handling applied successfully.")
                    st.markdown("#### Preview After Outlier Handling")
                    st.dataframe(df_cleaned.head(50))

        # ✅ Update session state after outlier step
        if df_cleaned is not None:
            st.session_state.cleaned_df = df_cleaned.copy()
        

        # -------------------------------
        # Feature Selection (Supervised + Unsupervised)
        # -------------------------------
        
        with st.expander("Feature Selection (Supervised + Unsupervised)", expanded=False):
            st.markdown("You can apply both **unsupervised** (statistical) and **supervised** (target-aware) feature selection methods.")

            method_type = st.radio("Choose method type:", ["Unsupervised", "Supervised"])

            numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()

            if method_type == "Unsupervised":
                st.subheader("Unsupervised Filters")

                # Low Variance
                low_var_thresh = st.slider("Variance threshold (remove features below this)", 0.0, 1.0, 0.0)
                low_var_cols = []
                if low_var_thresh > 0:
                    for col in numeric_cols:
                        if df_cleaned[col].var() < low_var_thresh:
                            low_var_cols.append(col)
                    if low_var_cols:
                        st.warning(f"Low variance columns: {low_var_cols}")
                        if st.button("Remove Low Variance Columns"):
                            df_cleaned.drop(columns=low_var_cols, inplace=True)
                            st.success("Low variance columns removed.")

                # High Correlation
                corr_thresh = st.slider("Correlation threshold", 0.5, 1.0, 0.9)
                if corr_thresh < 1.0 and len(numeric_cols) > 1:
                    corr_matrix = df_cleaned[numeric_cols].corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
                    if to_drop:
                        st.warning(f"Highly correlated columns: {to_drop}")
                        if st.button("Remove Correlated Columns"):
                            df_cleaned.drop(columns=to_drop, inplace=True)
                            st.success("Correlated columns removed.")

            elif method_type == "Supervised":
                st.subheader("Supervised Selection (requires target)")

                target_col = st.selectbox("Select target column:", df_cleaned.columns)
                if not target_col:
                    st.warning("Please select a target column for supervised feature selection.")
                    st.stop()

                if df_cleaned[target_col].isnull().sum() > 0:
                    st.error("Target column contains missing values. Please handle them before applying feature selection.")
                    st.stop()

                supervised_features = [col for col in df_cleaned.columns if col != target_col and is_numeric_dtype(df_cleaned[col])]

                selection_method = st.radio("Select method", ["ANOVA F-test", "Mutual Information"])
                k = st.slider("Number of top features to keep", 1, len(supervised_features), min(5, len(supervised_features)))

                try:
                    X = df_cleaned[supervised_features]
                    y = df_cleaned[target_col]

                    # ✅ Check for NaNs first
                    if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                        st.error("Missing values detected. Please handle missing values in the 'Handle Missing Values' section above before applying supervised feature selection.")
                    else:
                        # Encode target if it's categorical
                        if y.dtype == 'object' or y.dtype.name == 'category' or y.dtype == 'string':
                            y = LabelEncoder().fit_transform(y.astype(str))

                        if selection_method == "ANOVA F-test":
                            selector = SelectKBest(score_func=f_classif, k=k)
                        else:
                            selector = SelectKBest(score_func=mutual_info_classif, k=k)

                        selector.fit(X, y)
                        mask = selector.get_support()
                        selected = X.columns[mask].tolist()

                        st.success(f"Top {k} selected features: {selected}")
                        if st.button("Keep Selected Features Only"):
                            df_cleaned = df_cleaned[selected + [target_col]]
                            st.success("Filtered dataset to selected features.")
                except Exception as e:
                    st.error(f"Feature selection failed: {str(e)}")


            st.markdown("#### Preview After Feature Selection")
            st.dataframe(df_cleaned.head(50))

            if df_cleaned is not None:
                st.session_state.cleaned_df = df_cleaned.copy()

        # -------------------------------
        # Feature Engineering
        # -------------------------------
        with st.expander("Feature Engineering", expanded=False):
            st.markdown("Transform existing columns or extract new ones to boost model performance.")

            # -------------------------------
            # 1. Numeric Column Transformations
            # -------------------------------
            numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                st.markdown("##### 🔢 Numerical Transformations")
                selected_numeric = st.multiselect("Select numeric columns to transform:", numeric_cols)

                if selected_numeric:
                    transform_ops = st.multiselect("Select transformations to apply:", ["Square", "Square Root", "Log"])

                    for col in selected_numeric:
                        if "Square" in transform_ops:
                            df_cleaned[f"{col}_squared"] = df_cleaned[col] ** 2
                        if "Square Root" in transform_ops:
                            df_cleaned[f"{col}_sqrt"] = df_cleaned[col].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)
                        if "Log" in transform_ops:
                            df_cleaned[f"{col}_log"] = df_cleaned[col].apply(lambda x: np.log1p(x) if x >= 0 else np.nan)
                    st.success("Numeric transformations applied.")

            if df_cleaned is not None:
                st.session_state.cleaned_df = df_cleaned.copy()

    

            # -------------------------------
            # 3. Text Feature Extraction
            # -------------------------------
            text_cols = [col for col in df_cleaned.columns if is_object_dtype(df_cleaned[col]) and df_cleaned[col].nunique() < len(df_cleaned)]
    
            if text_cols:
                st.markdown("##### 🔤 Text Features")
                selected_text_cols = st.multiselect("Select text columns to extract features from:", text_cols)

                for col in selected_text_cols:
                    non_empty = df_cleaned[col].dropna().astype(str).str.strip()
                    if non_empty.eq("").all():
                        st.warning(f"Column `{col}` is empty or contains only missing values. Skipping text feature extraction.")
                        continue
                    df_cleaned[f"{col}_char_count"] = df_cleaned[col].astype(str).apply(len)
                    df_cleaned[f"{col}_word_count"] = df_cleaned[col].astype(str).apply(lambda x: len(str(x).split()))
                    df_cleaned[f"{col}_avg_word_len"] = df_cleaned[col].astype(str).apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)
                if selected_text_cols:
                    st.success("Text features extracted.")

            # ✅ Update session state
            st.markdown("#### Preview After Feature Engineering")
            st.dataframe(df_cleaned.head(50))

            if df_cleaned is not None:
                st.session_state.cleaned_df = df_cleaned.copy()
           
        

        # -------------------------------
        # Class Imbalance Handling
        # -------------------------------
        

        with st.expander("Class Imbalance Handling", expanded=False):
            st.markdown("Handle datasets where one class heavily outnumbers another (like fraud detection).")

            all_cols = df_cleaned.columns.tolist()
            target_col = st.selectbox("Select target column:", all_cols)

            # Detect classification task only
            if is_numeric_dtype(df_cleaned[target_col]) and df_cleaned[target_col].nunique() > 15:
                st.warning("This looks like a regression problem — imbalance handling is only for classification tasks.")
            else:
                imbalance_method = st.radio("Choose balancing method:", ["SMOTE", "Random OverSampling", "Random UnderSampling"])

                # Show class distribution before
                st.markdown("**Class distribution before balancing:**")
                st.write(dict(Counter(df_cleaned[target_col])))

                if st.button("Apply Balancing"):
                    

                    try:
                        with st.spinner("Applying class balancing..."):
                            X = df_cleaned.drop(columns=[target_col])
                            y = df_cleaned[target_col]

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

                            df_cleaned = pd.DataFrame(X_resampled, columns=X.columns)
                            df_cleaned[target_col] = y_resampled

                            st.success("Class balancing applied successfully.")
                            st.markdown("**Class distribution after balancing:**")
                            st.write(dict(Counter(y_resampled)))
                            st.dataframe(df_cleaned.head())

                            # ✅ Save to session
                            if df_cleaned is not None:
                                st.session_state.cleaned_df = df_cleaned.copy()
                    
                    except Exception as e:
                        st.error(f"Balancing failed: {e}")

    
# TODO: Missing value handling, scaling, encoding, download


# -------------------------------
# Tab 4: Modeling
# -------------------------------


with tab4:
    st.subheader("Train & Evaluate Models")

    def clean_column_names(df):
        df = df.copy()
        df.columns = [re.sub(r"[^\w]", "_", col) for col in df.columns]
        return df

    df_model = st.session_state.get("cleaned_df")
    if df_model is not None:
        df_model = df_model.copy()

    if df_model is None:
        st.warning("No dataset found. Please upload and preprocess data in Tab 1 to activate modeling.")
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
            task_type = st.radio("Select Task Type:", ["Classification", "Regression", "Clustering"])

        st.info(f"Detected Task Type: {task_type}")

        if task_type == "Clustering":
            model_choice = st.selectbox("Select Clustering Model", ["KMeans"])
            num_clusters = st.slider("Number of Clusters", 2, 10, 3)

            if st.button("Train Clustering Model"):
                model = KMeans(n_clusters=num_clusters, random_state=42)
                model.fit(df_model)
                df_model["Cluster"] = model.labels_
                st.success("KMeans clustering applied.")
                st.dataframe(df_model.sample(50))
                st.write("Inertia (Within-cluster sum of squares):", model.inertia_)

        else:
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




with tab5:
    st.subheader("Model Explainability")

    df = st.session_state.get("cleaned_df")
    if df is None:
        st.warning("Please upload and preprocess data in Tab 1.")
        st.stop()

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
                    st.markdown(f"### 🔍 {title} Plot {i + 1}: {name}")
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
            shap.plots.bar(shap_values, max_display=10, show=False, ax=ax)
            st.pyplot(fig)

            st.markdown("#### SHAP - Beeswarm")
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            st.pyplot(plt.gcf())

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







        


    
   



