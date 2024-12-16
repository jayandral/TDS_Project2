# /// script
# requires-python = ">=3.11"  # Ensure Python version is 3.11 or higher
# dependencies = [
#   "matplotlib",     # For plotting graphs
#   "seaborn",        # For statistical data visualization
#   "pandas",         # For data manipulation and analysis
#   "scipy",          # For scientific computing (stats, statistical tests, etc.)
#   "scikit-learn",   # For machine learning tasks (SimpleImputer, LabelEncoder)
#   "Pillow",         # For image processing (PIL module)
#   "requests",       # For making HTTP requests
#   "numpy",          # For numerical operations
# ]
# ///

import os
import sys
import pandas as pd
import requests
import json
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend("Agg")  # Set the backend to Agg for non-interactive environments
import numpy as np
import ast
import seaborn as sns
import matplotlib.patches as mpatches

from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway, shapiro, levene
from PIL import Image
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# Setup code: Retrieve the AIPROXY token and construct the OpenAI API endpoint URL
BASE_URL = "https://aiproxy.sanand.workers.dev/openai/"

# The OpenAI API endpoint for chat-based completions
URL = BASE_URL + "v1/chat/completions"

# Get the AIPROXY token
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Ensure the token is stripped of any leading/trailing whitespace or newline characters
AIPROXY_TOKEN = AIPROXY_TOKEN.strip() if AIPROXY_TOKEN else None

# Headers for the POST request to the OpenAI API
HEADERS = {
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
    "Content-Type": "application/json",
}

# if token is not set, raise an error
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set.")

# End of set-up code


# Function to generate the README file
def generate_readme(story, charts):
    readme = story

    # Include Visuals
    readme += "\n# Data Visualizations\n"
    for chart in charts:
        readme += f"### **{chart[0]}**\n\n"
        readme += f"![{chart[1]}]({chart[1]})\n\n"

    return readme


def generate_story(
    dataset_filename,
    dataset_description,
    key_column_exploration_result,
    dataset_analysis_result,
):

    behaviour = """You are a creative data analyst skilled in transforming raw data and analysis into captivating, story-like narratives. Your task is to craft a cohesive and engaging narrative that weaves together the insights, trends, and patterns uncovered during your analysis. The story should have a logical sequence, a clear flow, and a compelling structure—beginning with the context or problem, transitioning through the discoveries, and concluding with actionable takeaways or thought-provoking conclusions. Use vivid, relatable language and examples to make the data come alive, ensuring that even complex findings are accessible and engaging to a wide audience."""

    prompt = f"""Write a professional and narrative-driven README.md file in markdown format that presents the story of a dataset analysis journey. Follow the structured outline below:

        Structure and Content:
        ### 1. **Dataset Overview**
        Section Title (Centered): "Investigation"
        Section Sub-Title (Centered): "Dataset"
        Introduce the dataset by:
            Describing its key features (e.g., size, number of rows/columns, unique aspects).
            Highlighting any anomalies, patterns, or interesting observations from its description.
            Use bullet points for clarity.
        ### 2. **Analysis Methods**
        Section Title (Centered): "Dataset Analysis"
        Section Sub-Title (Centered): "Dataset analyst"
        Detail the analysis process, including:
            The selected key column (numerical or categorical) and the rationale for its selection.
            Analytical methods used (e.g., correlation for numerical or 1-way ANOVA for categorical data).
            Preprocessing steps such as handling missing values, imputations, and outlier detection.
        ### 3. **Key Insights and Patterns**
        Section Title (Centered): "The Revelation"
        Section Sub-Title (Centered): "Omnipotent Patterns"
        Present significant findings from the analysis:
            Summarize key insights and relationships revealed in the data.
            Use bullet points for critical insights and trends.
            Reference provided visualizations (e.g., "As shown below in Figure 1").
        ### 4. **Implications and Actions**
        Section Title (Centered): "Conclusion"
        Provide actionable recommendations based on the analysis:
            Suggest specific strategies, interventions, or next steps.
            Use bullet points for clarity.
            Formatting Guidelines:
            Use centered titles and subtitles styled as chapter headers to create a cohesive narrative.
            The tone should be professional, serious, and subtle—no dramatic embellishments.
            Use bold or italicized text to emphasize important terms and points.
            Reference provided charts without URLs, naming them Figure 1 and Figure 2 in context.
            Keep the file concise, structured, and logically formatted for automated script compatibility.
        Provided Information:
            Dataset filename: {dataset_filename}
            Dataset description: {dataset_description}
            Key column exploration result: {key_column_exploration_result}
            Key column exploration chart: Figure 1 
            Dataset analysis result: {dataset_analysis_result}
            Dataset analysis chart: Figure 2 
        Final Instructions:
            Focus entirely on generating markdown-formatted content with no extraneous instructions or commentary.
            Ensure the narrative reads smoothly, with logical transitions between sections.
            Keep the response around 1500 tokens to balance depth and clarity.
            Use storytelling techniques to make the analysis engaging while maintaining a formal tone.
        Output: A complete README.md file that adheres to this structure and presents a captivating, narrative-driven data analysis story.

    **Critical Final Instructions**:
    1. Your response should only be the content of the markdown file itself, because your response will be utilized by an automated script for further processing and creation of a 'README.md' file, otherwise the script will break.
    2. The markdown content should be **well-structured, concise**, and **actionable** presented like an intriguing story narrated by an analyst.
    3. Charts of Figure 1 and Figure 2 would be visible in the Data Visualizations section below and the chart links will be added by the automated script after processing. **Do not add chart links in the markdown content.
    4. Do not reference the automated script in the markdown content.
    4. Your response should be approximately 1500 tokens. 
    """

    messages = [
        {"role": "system", "content": behaviour},
        {"role": "user", "content": prompt},
    ]

    # Data for the request, including the 'messages' parameter
    data = {
        "model": "gpt-4o-mini",  # Can also use gpt-3.5-turbo
        "messages": messages,
        "max_tokens": 3000,  # Optional parameter to limit token usage
        "temperature": 1.0,
    }

    # Send the POST request to the OpenAI API via AIPROXY
    response = requests.post(URL, headers=HEADERS, data=json.dumps(data))

    print(response.json())
    # Check if the request was successful and return the result
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return None

    # return {'col_type': 'numerical', 'col_name': 'overall'}


def resize_chart_for_llm(fig, new_size=(512, 512)):
    """
    Resize the saved chart figure and return it as a BytesIO object for sending to the LLM for saving tokens.

    Parameters:
    fig (matplotlib.figure.Figure): The matplotlib figure object to resize.
    new_size (tuple): Desired size of the chart (default is 512x512).

    Returns:
    BytesIO object containing the resized image.
    """
    try:
        # Save the figure to a BytesIO buffer
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=200)
        buf.seek(0)  # Reset buffer to the start

        # Open the image from the buffer using Pillow
        img = Image.open(buf)

        # Resize the image to the desired size in memory (use LANCZOS for high-quality downsampling)
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save the resized image to a new BytesIO buffer
        resized_buf = BytesIO()
        img_resized.save(resized_buf, format="PNG")
    except Exception as e:
        print(f"Error while resizing the image: {e}")
        raise


def plot_anova_result(dataset_analysis_result, key_column):
    """
    This function takes the ANOVA analysis result and plots a horizontal bar plot of the F-statistics for factors
    categorized as significant and non-significant. It also adjusts the plot title to reflect the key variable used
    in the analysis.

    Parameters:
    - dataset_analysis_result: A dictionary containing 'Significant Factors' and 'Non-significant Factors',
      with F-statistic and p-value for each factor.
    - key_column: The categorical variable used for performing ANOVA.
    """

    # Flatten the ANOVA results for plotting
    plot_data = []
    for column, stats in {
        **dataset_analysis_result["1-Way ANOVA Analysis Result"]["Significant Columns"],
        **dataset_analysis_result["1-Way ANOVA Analysis Result"][
            "Non-significant Columns"
        ],
    }.items():
        plot_data.append(
            {
                "Column": column,
                "F-statistic": round(stats[0], 2),  # F-statistic
                "p-value": round(stats[1], 2),  # p-value
                "Significance": (
                    "Significant" if stats[1] <= 0.05 else "Non-significant"
                ),
            }
        )

    # Create a DataFrame for plotting
    df = pd.DataFrame(plot_data)

    # Dynamically adjust figure size based on the number of factors (categories) and significance categories
    # Adjust height for more factors
    height = min(2 + len(df["Column"].unique()) * 0.5, 12)
    figsize = (12, max(3, height))  # Set the figure size

    # Create the plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x="F-statistic",
        y="Column",
        hue="Significance",
        data=df,
        palette={"Significant": "green", "Non-significant": "gray"},
        orient="h",
    )

    # Title and labels
    plt.title(
        f"F-statistics for 1-way ANOVA with '{key_column}' selected as Grouping Column"
    )
    plt.xlabel(f"F-statistic Values for Columns (w.r.t. '{key_column}' column)")
    plt.ylabel("Columns")

    # Ensure the legend always shows both 'Significant' and 'Non-significant'
    handles, labels = ax.get_legend_handles_labels()

    # Manually add a legend handle for 'Non-significant' if it's missing
    if "Non-significant" not in labels:
        # Create a uniform Patch for Non-significant with the same color as the bars
        non_significant_patch = mpatches.Patch(color="gray", label="Non-significant")
        handles.append(non_significant_patch)
        labels.append("Non-significant")

    # Set the legend with both 'Significant' and 'Non-significant'
    ax.legend(handles, labels, title="Significance", loc="upper right")

    # Return the current figure (to be displayed or saved later)
    return plt.gcf()  # Return the figure object


def plot_correlation_result(dataset_analysis_result, key_column):
    """
    Plots the correlation results from a dataset analysis.
    This function takes the results of a dataset correlation analysis and generates a horizontal bar plot
    to visualize the correlation coefficients of different columns with respect to a key column. The plot
    is dynamically sized based on the number of variables and categories in the dataset.

    Parameters:
    dataset_analysis_result (dict): A dictionary containing the results of the dataset analysis. It should
                                    include a "Correlation Analysis Result" key with nested dictionaries of
                                    categories and their corresponding correlation values, and an "Additional
                                    Statistics" key with mean, standard deviation, skewness, and kurtosis values.
    key_column (str): The name of the key column with respect to which the correlation trends are analyzed.

    Returns:
    matplotlib.figure.Figure: The figure object containing the generated plot.
    """

    # Flatten the dictionary for plotting
    correlations = dataset_analysis_result["Correlation Analysis Result"]
    plot_data = [
        {"Category": category, "Column": column, "Correlation": corr}
        for category, columns in correlations.items()
        for column, corr in columns.items()
    ]

    # Create a DataFrame for plotting
    df = pd.DataFrame(plot_data)

    # Calculate figure dimensions dynamically
    num_variables = len(df["Column"].unique())
    num_categories = len(df["Category"].unique())
    height = min(2 + num_variables * 0.5, 12)  # Cap the height at 12
    width = max(12, num_categories * 1.2)  # Adjust width based on categories
    figsize = (width, max(3, height))

    # Create the bar plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x="Correlation",
        y="Column",
        hue="Category",
        data=df,
        palette="coolwarm",
        orient="h",
    )

    # Title and labels
    plt.title(f"Correlation Trends w.r.t. '{key_column}' Column", fontsize=16)
    plt.xlabel("Correlation Coefficient", fontsize=12)
    plt.ylabel("Columns", fontsize=12)

    # Annotate each bar with its corresponding correlation value
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.05,
            p.get_y() + p.get_height() / 2,
            f"{width:.2f}",
            ha="left",
            va="center",
            color="black",
            fontsize=12,
        )

    # Move the legend outside to avoid overlap
    plt.legend(
        title="Correlation Categories",
        loc="lower left",
        bbox_to_anchor=(1, 1),
        fontsize=10,
    )

    # Move figtext outside the plot area
    plt.figtext(
        0.05,
        0.95,
        f"Mean: {dataset_analysis_result['Additional Statistics']['Mean Correlation']} | "
        f"Std Dev: {dataset_analysis_result['Additional Statistics']['Standard Deviation']} | "
        f"Skewness: {dataset_analysis_result['Additional Statistics']['Skewness']} | "
        f"Kurtosis: {dataset_analysis_result['Additional Statistics']['Kurtosis']}",
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    # Automatically adjust the layout to avoid clipping
    plt.tight_layout()

    # Return the current figure object (to be displayed or saved later)
    return plt.gcf()


def perform_anova(data, key_column, p_value_threshold=0.05):
    """
    Perform ANOVA on the given dataset for all numeric columns, grouped by a key categorical variable.
    Assumptions of ANOVA: normality of groups and homogeneity of variance are checked.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - key_column: Categorical column to group data by.
    - p_value_threshold: Threshold to classify factors as significant or non-significant.

    Returns:
    - Dictionary containing significant and non-significant factors.
    """
    anova_results = {}

    # Check if dataset is empty
    if data.empty:
        raise ValueError(
            "The dataset is empty. ANOVA cannot be performed on an empty dataset."
        )

    # Check that the key_column is in the data
    if key_column not in data.columns:
        raise ValueError(f"'{key_column}' is not a column in the DataFrame.")

    # Check that the key_column doesn't have missing values
    if data[key_column].isnull().any():
        raise ValueError(
            f"The '{key_column}' column contains missing values. Please clean the data."
        )

    # Automatically get numeric columns
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_columns) == 0:
        raise ValueError(
            "No numeric columns found in the dataset. ANOVA requires numeric data."
        )

    # Perform ANOVA for each numeric column
    for col in numeric_columns:
        # Group data by the key variable
        groups = [group[col].dropna() for _, group in data.groupby(key_column)]

        # Filter out groups with fewer than two values
        groups = [group for group in groups if len(group) > 1]

        # Apply ANOVA only if we have at least two groups
        if len(groups) > 1:
            # Check normality for each group using Shapiro-Wilk test
            for i, group in enumerate(groups):
                if shapiro(group)[1] < 0.05:
                    print(
                        f"Warning: Group {i+1} (of column '{col}') does not follow normal distribution)"
                    )

            # Check homogeneity of variance using Levene's test
            if levene(*groups)[1] < 0.05:
                print(
                    f"Warning: Homogeneity of variance assumption violated for column '{col}'"
                )

            # Perform the ANOVA
            f_stat, p_value = f_oneway(*groups)
            # Round for readability
            anova_results[col] = [round(f_stat, 2), round(p_value, 6)]
        else:
            anova_results[col] = ["Error", "Not enough data"]

    # Process the ANOVA results to classify as significant or non-significant
    dataset_analysis_result = {
        "Dataset Analysis Technique": "1-Way ANOVA",
        "Correlation w.r.t. Key Column": key_column,
        "1-Way ANOVA Analysis Result": {
            "Significant Columns": {},
            "Non-significant Columns": {},
        },
        "Dataset Analysis Chart Figure": f"Figure 2: Horizontal Bar Plot - 1-Way ANOVA Analysis w.r.t. '{key_column}' column",
        "Dataset Analysis Chart Title": f"1-Way ANOVA Analysis w.r.t. '{key_column}' column",
        "Dataset Analysis Chart Filename": "dataset_analysis_chart.png",
        "Plot Type of Dataset Analysis Chart": "Horizontal Bar Plot",
    }

    # Process the results and classify them
    for col, result in anova_results.items():
        if result == ["Error", "Not enough data"]:
            dataset_analysis_result["1-Way ANOVA Analysis Result"][
                "Non-significant Columns"
            ][col] = result
        else:
            f_stat, p_value = result
            if p_value <= p_value_threshold:
                dataset_analysis_result["1-Way ANOVA Analysis Result"][
                    "Significant Columns"
                ][col] = [f_stat, p_value]
            else:
                dataset_analysis_result["1-Way ANOVA Analysis Result"][
                    "Non-significant Columns"
                ][col] = [f_stat, p_value]

    return dataset_analysis_result


def perform_correlation(df, key_column):
    """
    Perform correlation analysis on a DataFrame with respect to a key column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    key_column (str): The name of the key column to compute correlations against. Must be a numeric column.

    Returns:
    dict: A dictionary containing the following keys:
        - "Dataset Analysis Technique": The technique used for analysis ("Correlation Analysis").
        - "Correlation w.r.t. Key Column": The key column used for correlation analysis.
        - "Correlation Analysis Result": A dictionary categorizing variables based on their correlation strength:
            - "Strong Positive": Variables with correlation >= 0.75.
            - "Moderate Positive": Variables with correlation >= 0.50 and < 0.75.
            - "Weak Correlations": Variables with correlation > -0.50 and < 0.50.
            - "Moderate Negative": Variables with correlation <= -0.50 and > -0.75.
            - "Strong Negative": Variables with correlation <= -0.75.
        - "Dataset Analysis Chart Figure": Description of the chart figure.
        - "Dataset Analysis Chart Title": Title of the chart.
        - "Dataset Analysis Chart Filename": Filename for saving the chart.
        - "Plot Type of Dataset Analysis Chart": Type of plot used for visualization.
        - "Additional Statistics": A dictionary containing additional statistical measures:
            - "Mean Correlation": Mean of the correlations.
            - "Standard Deviation": Standard deviation of the correlations.
            - "Median Correlation": Median of the correlations.
            - "Range of Correlations": Range of the correlations.
            - "Skewness": Skewness of the correlations.
            - "Kurtosis": Kurtosis of the correlations.

    Raises:
    ValueError: If the key_column is not a numeric column in the DataFrame.
    """

    # Ensure the key_column and other columns are numeric
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if key_column not in numeric_columns:
        raise ValueError(f"The key_column '{key_column}' must be a numeric column.")

    # Compute correlation of the key variable with all numeric variables
    correlations = df[numeric_columns].corr()[key_column].sort_values(ascending=False)

    # Initialize the structure for significant correlations
    significant_correlations = {
        "Strong Positive": {},
        "Moderate Positive": {},
        "Weak Correlations": {},
        "Moderate Negative": {},
        "Strong Negative": {},
    }

    # Classify correlations
    for variable, correlation in correlations.items():
        if variable == key_column:
            continue

        rounded_correlation = round(correlation, 2)

        if correlation >= 0.75:
            category = "Strong Positive"
        elif correlation >= 0.50:
            category = "Moderate Positive"
        elif correlation <= -0.75:
            category = "Strong Negative"
        elif correlation <= -0.50:
            category = "Moderate Negative"
        else:
            category = "Weak Correlations"

        significant_correlations[category][variable] = rounded_correlation

    # Exclude the key variable itself for descriptive statistics
    correlations_without_key = correlations.drop(key_column, errors="ignore")
    if correlations_without_key.empty:
        return {"Error": "No correlations available for analysis."}

    # Compute descriptive statistics
    mean_corr = correlations_without_key.mean()
    std_corr = correlations_without_key.std()
    median_corr = correlations_without_key.median()
    range_corr = correlations_without_key.max() - correlations_without_key.min()
    skew_corr = ((correlations_without_key - mean_corr) ** 3).mean() / (std_corr**3)
    kurt_corr = ((correlations_without_key - mean_corr) ** 4).mean() / (std_corr**4)

    # Add the plot type and statistics to the result
    dataset_analysis_result = {
        "Dataset Analysis Technique": "Correlation Analysis",
        "Correlation w.r.t. Key Column": key_column,
        "Correlation Analysis Result": significant_correlations,
        "Dataset Analysis Chart Figure": f"Figure 2 : Horizontal Bar Plot - Correlation Analysis w.r.t. '{key_column}' column",
        "Dataset Analysis Chart Title": f"Correlation Analysis w.r.t. '{key_column}' column",
        "Dataset Analysis Chart Filename": "dataset_analysis_chart.png",
        "Plot Type of Dataset Analysis Chart": "Horizontal Bar Plot",
        "Additional Statistics": {
            "Mean Correlation": round(mean_corr, 2),
            "Standard Deviation": round(std_corr, 2),
            "Median Correlation": round(median_corr, 2),
            "Range of Correlations": round(range_corr, 2),
            "Skewness": round(skew_corr, 2),
            "Kurtosis": round(kurt_corr, 2),
        },
    }

    return dataset_analysis_result


def clean_data_for_analysis(df):
    """
    Cleans the input DataFrame for analysis by handling missing values and outliers.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to be cleaned.

    Returns:
    pandas.DataFrame: The cleaned DataFrame with missing values and outliers handled.
    """

    df_preprocessed = df.copy()

    # 1. Handle Missing Values
    df_preprocessed = handle_missing_values(df_preprocessed)

    # 2. Handle Outliers (optional)
    df_preprocessed = handle_outliers(df_preprocessed)

    return df_preprocessed


def handle_missing_values(df):
    """
    Handle missing values in a DataFrame.
    This function imputes missing numerical values with the mean of the respective columns
    and fills missing categorical values with the string "Unknown".

    Parameters:
    df (pandas.DataFrame): The input DataFrame with potential missing values.

    Returns:
    pandas.DataFrame: The DataFrame with missing values handled.
    """

    # Impute missing numerical values with the mean
    numerical_cols = df.select_dtypes(include=["number"]).columns
    imputer_num = SimpleImputer(strategy="mean")
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

    # Handle categorical columns with missing values
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    return df


def handle_outliers(df, z_threshold=3):
    """
    Remove or cap outliers in the numerical columns using Z-score method.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - z_threshold (float): Z-score threshold above which values are considered outliers.

    Returns:
    - df (pd.DataFrame): Dataframe with outliers removed or capped.
    """

    numerical_cols = df.select_dtypes(include=["number"]).columns

    # Calculate Z-scores for numerical columns
    z_scores = np.abs(stats.zscore(df[numerical_cols]))

    # Remove rows with Z-scores greater than threshold
    df_cleaned = df[(z_scores < z_threshold).all(axis=1)]

    return df_cleaned


def key_column_exploration_result_and_plot(df, col_type, col_name):
    """
    Analyzes and plots the distribution of a specified column in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    col_type (str): The type of the column ('numerical' or 'categorical').
    col_name (str): The name of the column to analyze.

    Returns:
    tuple: A tuple containing:
        - result (dict): A dictionary with the analysis results and statistics.
        - plt.Figure: The matplotlib figure object of the generated plot.

    If the column does not exist in the DataFrame, prints an error message and returns None, None.
    If the column type is unknown, prints an error message and returns None, None.

    The result dictionary contains:
    - Key Column Name (str): The name of the analyzed column.
    - Key Column Type (str): The type of the analyzed column ('Numerical' or 'Categorical').
    - Key Column Exploration Chart Figure (str): Description of the generated plot.
    - Key Column Exploration Chart Title (str): Title of the generated plot.
    - Key Column Exploration Chart Filename (str): Suggested filename for saving the plot.
    - plot_type (str): The type of the plot generated.

    For numerical columns, the result dictionary also includes:
    - mean (float): The mean of the column values.
    - median (float): The median of the column values.
    - std_dev (float): The standard deviation of the column values.
    - skewness (float): The skewness of the column values.
    - kurtosis (float): The kurtosis of the column values.
    - min_value (float): The minimum value in the column.
    - max_value (float): The maximum value in the column.
    - quantiles (dict): The quantiles (25th, 50th, 75th percentiles) of the column values.
    - normality_test (float): The p-value of the normality test on the column values.

    For categorical columns, the result dictionary also includes:
    - unique_values (int): The number of unique values in the column.
    - value_counts (dict): The frequency count of each unique value in the column.
    - mode (str): The mode (most frequent value) of the column.
    - missing_values (int): The number of missing values in the column.
    - missing_percentage (float): The percentage of missing values in the column.

    Example usage:
    result, fig = key_column_exploration_result_and_plot(df, 'numerical', 'age')
    """

    # Check if the column exists in the dataframe
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found in the dataframe.")
        return

    # Extract the column data
    column_data = df[col_name]

    # To store the result of the analysis
    result = {
        "Key Column Name": col_name,
        "Key Column Type": col_type.capitalize(),
        "Chart Figure": None,
        "Chart Title": None,
        "Chart Filename": "key_column_exploration_chart.png",
        "plot_type": None,
    }

    # Check if the column is numerical
    if col_type == "numerical":
        # Plot density plot for numerical data (KDE)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(column_data, fill=True, color="blue", alpha=0.8)
        plt.title(f"Density Plot - Distribution of '{col_name}' Column Values")
        plt.xlabel(f"Values in '{col_name}' column")
        plt.ylabel("Density")
        plt.grid(True)

        # Key Column Exploration Chart
        result["Chart Figure"] = (
            f"Figure 1 : Density Plot - Distribution of '{col_name}' Column Values"
        )
        result["Chart Title"] = (
            f"Density Plot (KDE) - Distribution of '{col_name}' Column Values"
        )
        result["plot_type"] = "Density Plot (KDE)"

        # Statistics
        result["mean"] = float(round(column_data.mean(), 2))
        result["median"] = float(round(column_data.median(), 2))
        result["std_dev"] = float(round(column_data.std(), 3))
        result["skewness"] = float(round(column_data.skew(), 3))
        result["kurtosis"] = float(round(column_data.kurtosis(), 3))
        result["min_value"] = float(round(column_data.min(), 3))
        result["max_value"] = float(round(column_data.max(), 3))
        result["quantiles"] = column_data.quantile([0.25, 0.5, 0.75]).to_dict()
        # Additional statistical test (if needed)
        result["normality_test"] = float(
            round(stats.normaltest(column_data.dropna()).pvalue, 5)
        )

        return result, plt.gcf()

    # Check if the column is categorical
    elif col_type == "categorical":
        # Plot frequency count bar chart for categorical data (Fixing the warning)
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col_name, hue=col_name, palette="Set2", legend=False)
        plt.title(f"Frequency Count of '{col_name}' Column")
        plt.xlabel(f"'{col_name}' Column Values")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")  # Rotate labels if necessary
        plt.grid(True)

        # Key Column Exploration Chart
        result["Chart Figure"] = (
            f"Figure 1 : Frequency Count Bar Chart - Frequency Count of '{col_name}' Column"
        )
        result["Chart Title"] = f"Frequency Count of '{col_name}' Column"
        result["plot_type"] = "Frequency Count Bar Chart"

        # Statistics
        result["unique_values"] = column_data.nunique()
        result["value_counts"] = column_data.value_counts().to_dict()
        result["mode"] = column_data.mode().iloc[0]
        result["missing_values"] = column_data.isnull().sum()
        result["missing_percentage"] = column_data.isnull().mean() * 100

        return result, plt.gcf()

    else:
        print(f"Unknown column type '{col_type}' for '{col_name}'.")
        return None, None


def select_key_column(URL, dataset_filename, dataset_description):
    """
    Select the most impactful column in a dataset for analysis.
    This function sends a request to an AI model to determine the most impactful column
    in a given dataset based on its description. The AI model is prompted to act as a
    data analyst and select a column that would provide key insights and patterns.

    Parameters:
    URL (str): The URL endpoint for the AI model API.
    dataset_filename (str): The filename of the dataset.
    dataset_description (str): A summary description of the dataset.

    Returns:
    dict: A dictionary containing the type ('categorical' or 'numerical') and name of the selected column.
          Example: {'col_type': 'categorical', 'col_name': 'name_of_column'}
          Returns None if the request was unsuccessful.
    """

    # Set up the system behavior and user prompt for the AI interaction.
    # 'behaviour' defines the role of the assistant as a data analyst.
    # 'prompt' contains the task and rules for selecting the most impactful column in the dataset.
    # The 'messages' list is then constructed with the system's instructions and the user input prompt.

    behaviour = """You are a data analyst.
    Your task is to select the most impactful column in the dataset to analyze, discover patterns, and gain key insights."""

    prompt = f"""
    The dataset filename is {dataset_filename}.
    Here is the dataset summary delimited by ``: `{dataset_description}`.

    Your task : -- Select the most impactful top 1 columns to discover patterns and gain key insights from the dataset. --

    **Forget the context from any previous prompts before trying to respond to this prompt**
    **Respond ONLY with a dictionary in this format: {{'col_type': 'categorical/numerical', 'col_name': 'name_of_column'}}**
    """

    messages = [
        {"role": "system", "content": behaviour},
        {"role": "user", "content": prompt},
    ]

    # Data for the request, including the 'messages' parameter
    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.0,
    }
    
    print(data)

    # Send the POST request to the OpenAI API via AIPROXY
    response = requests.post(URL, headers=HEADERS, data=json.dumps(data))
    print(response)
    # Check if the request was successful and return the result
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return None


def generate_dataset_description(df):
    """
    Generates a textual description of a given pandas DataFrame.
    Parameters:
    df (pandas.DataFrame): The DataFrame for which the description is to be generated.
    Returns:
    str: A string containing the description of the dataset, including:
        - The number of rows and columns.
        - Lists of numeric and categorical columns.
        - Percentage of missing values in each column (if any).
        - Columns with fewer than 20 unique values and their unique values.
    """

    # Create a description string
    dataset_description = f"Dataset has {len(df)} rows and {len(df.columns)} columns."
    dataset_description += f"\nNumeric columns: '{', '.join(df.select_dtypes(exclude='object').columns.tolist())}' "
    dataset_description += f"\nCategorical columns : '{', '.join(df.select_dtypes(include='object').columns.tolist())}'"

    # Get missing values information
    missing_values = df.isna().sum()

    # Add missing values information
    missing_info = {}
    for col in df.columns:
        missing_info[col] = f"{round(missing_values[col] / len(df) * 100, 0)}%"

    # Adding to dataset_description if there is missing info
    dataset_description += (
        f"\nPercentage of missing values in columns : {missing_info}."
    )

    # Identify and add columns with few unique values to dataset_description
    # Few unique values are indicative of categorical columns which helps in grouping for ANOVA.
    columns_with_few_unique_values = {}
    # Cut-off for number of unique values considered as `few` is set at 20
    threshold = 20

    for column in df.columns:
        # Check if the number of unique values is less than the threshold
        if df[column].nunique() < threshold:
            # Store column name and unique values
            columns_with_few_unique_values[column] = df[column].unique().tolist()

    if columns_with_few_unique_values:
        dataset_description += f"\nColumns with less than 20 unique values : {columns_with_few_unique_values}"

    return dataset_description


def save_and_resize_charts(key_column_exploration_chart, dataset_analysis_chart):
    """Function to save charts and resize them for LLM."""
    # Save charts
    key_column_exploration_chart.savefig(
        "key_column_exploration_chart", bbox_inches="tight"
    )
    dataset_analysis_chart.savefig("dataset_analysis_chart", bbox_inches="tight")

    # Resize charts for LLM
    resize_chart_for_llm(key_column_exploration_chart)
    resize_chart_for_llm(dataset_analysis_chart)


def load_and_validate_dataset():
    """
    Load and validate a dataset from a CSV file provided via command-line arguments.

    This function attempts to load a dataset from a CSV file specified as the first
    command-line argument. It tries different encodings to read the file and checks
    if the dataset is empty. If the file is successfully read and is not empty, it
    returns the dataframe and the filename. If the file cannot be read or is empty,
    it returns None and the filename.

    Returns:
        tuple: A tuple containing the dataframe (or None if loading failed) and the
               dataset filename (or None if no filename was provided).
    """

    # Check if dataset filename is provided in command-line and load the dataset file
    if len(sys.argv) != 2:
        print("Please provide the dataset filename as a command-line argument.")
        return None, None

    dataset_filename = sys.argv[1]

    # Try loading the dataset using different encodings
    encodings_to_try = ["utf-8", "ISO-8859-1", "latin1", "cp1252"]
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(dataset_filename, encoding=encoding)
            print(f"Successfully read the file with encoding: {encoding}")

            # Check if the dataset is empty
            if df.empty:
                print(f"Warning: The dataset '{dataset_filename}' is empty.")
                return None, dataset_filename  # Return None if dataset is empty

            # Return the dataframe and filename upon successful loading
            return df, dataset_filename
        except Exception as e:
            print(f"Error reading {dataset_filename} with encoding {encoding}: {e}")

    # If all attempts fail, print an error message
    print(f"Failed to read {dataset_filename} with multiple encodings.")

    return None, dataset_filename  # Return None if all encodings fail


def main():

    # Load and validate dataset
    df, dataset_filename = load_and_validate_dataset()
    if df is None:
        return

    # Clean the data for analysis
    df = clean_data_for_analysis(df)

    dataset_description = generate_dataset_description(df)

    key_column_string = select_key_column(URL, dataset_filename, dataset_description)

    # Convert the string to a dictionary
    key_column = ast.literal_eval(key_column_string)

    # key_column = {"col_type": "numerical", "col_name": "Life Ladder"}

    col_stats, col_fig = key_column_exploration_result_and_plot(
        df, key_column["col_type"], key_column["col_name"]
    )

    if key_column["col_type"] == "numerical":
        dataset_analysis_result = perform_correlation(df, key_column["col_name"])
        dataset_analysis_chart = plot_correlation_result(
            dataset_analysis_result, key_column["col_name"]
        )
    else:
        dataset_analysis_result = perform_anova(df, key_column["col_name"])
        dataset_analysis_chart = plot_anova_result(
            dataset_analysis_result, key_column["col_name"]
        )

    # Generate and save charts for LLM
    save_and_resize_charts(col_fig, dataset_analysis_chart)

    # Call LLM for to get story
    story = generate_story(
        dataset_filename,
        dataset_description,
        col_fig,
        dataset_analysis_result,
    )

    # Collect Chart Filenames for embedding in README
    charts = []
    chart_fig = col_stats["Chart Figure"]
    chart_name = col_stats["Chart Filename"]
    charts.append([chart_fig, chart_name])
    chart_fig = dataset_analysis_result["Dataset Analysis Chart Figure"]
    chart_name = dataset_analysis_result["Dataset Analysis Chart Filename"]
    charts.append([chart_fig, chart_name])

    # Combine the story and charts for embedding in README
    readme = generate_readme(story, charts)
    
    folder_name = Path(dataset_filename).stem
    os.makedirs(folder_name, mode=0o777, exist_ok=True)

    # Create README.md File
    with open(f"{folder_name}/README.md", "w") as f:
        f.write(readme)

    for image in [charts[1], charts[3]]:
        image_path = Path(image)
        os.rename(image_path, f"{folder_name}/{image_path.name}")


if __name__ == "__main__":
    main()
