# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests",
#   "pandas",
#   "seaborn",
# ]
# ///

from pathlib import Path
from typing import Any, List

import pandas as pd
import os
import sys
import io
import json
import logging
import requests
import base64
import traceback
import time

# Constants for API interaction
OPENAI_API = os.getenv(
    "OPENAI_API_URL", "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
)

# Check for the API Token, which is necessary for authentication
AIPROXY_API_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_API_TOKEN:
    raise ValueError(
        "AIPROXY_TOKEN environment variable is not set."
    )  # Ensure token is present for authentication

# API headers for authentication
# Headers for making requests to the API, includes authorization header
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_API_TOKEN}",
}

# Prompt template for generating Python code for dataset analysis
# PYTHON_CODE_PROMPT = """I will provide details of a pandas DataFrame named `df`. Write Python code to dynamically generate three graphs that best describe the dataset based on its structure and contents, ensuring that each graph includes at least one categorical and one numeric variable. Analyze the dataset to determine the most relevant visualizations by examining the number of categorical and numeric columns. Select appropriate graphs based on the dataset’s structure, such as bar plots, box plots, scatter plots, or other suitable visualizations. The first graph should show the aggregation (e.g., mean or sum) of a numeric variable grouped by a categorical variable. The second graph should highlight variations or distributions of a numeric variable across categories, such as a box plot. The third graph should visualize interactions between two categorical variables and their relationship with a numeric variable, such as a stacked or grouped bar chart. Dynamically adapt the visualizations to fit the dataset’s specific characteristics, including column names, data types, and unique value counts, ensuring each plot provides meaningful insights. Ensure all plots have appropriate titles, axis labels, and legends where applicable, and save each plot as a .png file with descriptive filenames. Use only matplotlib, seaborn, and pandas for the analysis and visualization. Do not display the graphs or provide explanations. Provide only the Python code for these visualizations, ensuring the code is free from errors."""
PYTHON_CODE_PROMPT = """I will provide details of a pandas DataFrame named `df`. Write Python code to dynamically generate three graphs that best describe the dataset based on its structure and contents. Analyze the dataset to determine the most relevant visualizations by examining the number of categorical and numeric columns. Select appropriate graphs based on the dataset’s structure, such as bar plots, box plots, scatter plots, or other suitable visualizations. Dynamically adapt the visualizations to fit the dataset’s specific characteristics, including column names, data types, and unique value counts, ensuring each plot provides meaningful insights. Ensure all plots have appropriate titles, axis labels, and legends where applicable, and save each plot as a .png file with descriptive filenames. Use only matplotlib, seaborn, and pandas for the analysis and visualization. Do not display the graphs or provide explanations. Provide only the Python code for these visualizations, ensuring the code is free from errors. Additionally, if the number of unique categories is huge for a categorical column try avoiding it."""

# Prompt template for analyzing a given graph and describing its features and patterns
GRAPH_ANALYSIS_PROMPT = """Analyze the provided graph by identifying its type, describing the data it represents (including axes, labels, units, and legends), summarizing key patterns or insights such as trends, clusters, outliers, or relationships, highlighting notable features like peaks or differences, and interpreting the implications of the data clearly and concisely based solely on what is visible in the graph."""

# Prompt template for generating a structured README for the dataset, including analysis, visuals, and findings
README_PROMPT = """"Generate a README.md file for the dataset titled '{filename}' by structuring it as follows:

1. Title the file with the dataset name: '# {filename}'.

2. Include an 'Overview' section that briefly introduces the dataset.

3. Add a 'Column Details' section that explains and analyzes the dataset's columns. Use the info provided to get insights of different columns.
    - '{column_details}'
    
4. Add a 'Numerical Details' section to analyze the numerical columns. Use the info provide to get insights into the numerical data.
    - '{numerical_details}'
    
5. Add a 'Categorical Details' section to explain the categorical columns. Use the info provide to get insights into the categorical data.
    - '{categorical_details}'
    
6. For the visual analysis, include sections with images and analyses with appropriate titles and descriptions.
   - Include the image path '{image1}' with the corresponding analysis.
        - '{analysis1}'
        - Make sure to include "Description of the Graph", "Key Observations", and "Implications"
   - Include the image path '{image2}' with the corresponding analysis.
        - '{analysis2}'
        - Make sure to include "Description of the Graph", "Key Observations", and "Implications"
   - Include the image path '{image3}' with the corresponding analysis.
        - '{analysis3}'
        - Make sure to include "Description of the Graph", "Key Observations", and "Implications"
        
7. Create a 'How the Analysis Was Carried Out' section, describing the steps and methods used in the analysis, based on the provided data.

8. End with a 'Findings' section that summarizes key discoveries from the analysis. Include insights about trends, correlations, outliers, or any interesting data patterns.

Ensure all sections are well-formatted and provide a clear explanation for each part of the dataset. The analysis should be presented in a concise and understandable manner."
"""

# Set up logging configuration to log messages at INFO level
logging.basicConfig(level=logging.INFO)


def validate_input_file(file_path: Path):
    """
    Validates that the input file has a .csv extension.

    Args:
        file_path (Path): The path to the input file.

    Raises:
        SystemExit: If the input file does not have a .csv extension, the program will log a warning and exit.
    """
    if not (file_path.suffix == ".csv" and file_path.is_file()):
        logging.warning("Input file must be a CSV.")
        sys.exit(1)

    # Check if the file exists
    file_path.resolve(strict=True)


def read_csv(file_path: Path) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        file_path (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The contents of the CSV file as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If the file cannot be parsed.
    """

    logging.info("Reading CSV file.")
    try:
        return pd.read_csv(file_path, encoding="unicode_escape")
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        sys.exit(1)


def encode_png_image(image_path: str) -> str:
    """
    Encodes a PNG image to a base64 string, with a data URL prefix.

    This function reads a PNG image from the given file path, encodes its content
    to base64, and returns the encoded string with a 'data:image/png;base64,' prefix
    so that it can be embedded directly in HTML or other web applications.

    Args:
        image_path (str): The file path to the PNG image to be encoded.

    Returns:
        str: The base64 encoded string of the PNG image with a data URL prefix.

    Example:
        >>> encode_png_image("image.png")
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA...'
    """

    if image_path.suffix.lower() != ".png":
        logging.warning("File is not a PNG image.")
        return ""

    try:
        # Open the image file in binary read mode and encode its content to base64
        with open(image_path, "rb") as image_file:
            # Return the base64 string with the appropriate prefix for embedding in web applications
            return "data:image/png;base64," + base64.b64encode(
                image_file.read()
            ).decode("utf-8")
    except Exception as e:
        logging.error(f"Error encoding image {image_path.name}: {e}")
        return ""


def get_column_details(data: pd.DataFrame) -> str:
    """
    Generate detailed information about the columns of a given DataFrame.

    This function provides a summary of the DataFrame's columns, including:
    - General column information (excluding header/footer lines)
    - Numerical details using the describe() method
    - Categorical details such as unique value count, mode, and missing values

    Args:
        data (pd.DataFrame): The input DataFrame to analyze.

    Returns:
        dict: A dictionary containing:
            - "column details": A string with general column information.
            - "numerical details": A string with numerical details of the DataFrame.
            - "categorical details": A string with details of categorical columns.
    """

    # Create a StringIO buffer to capture the DataFrame info output
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_string = buffer.getvalue()

    # Extract column details from the info string (excluding header/footer lines)
    column_details = "\n".join(info_string.split("\n")[3:-3])

    # Generate numerical details using describe() method with formatting for readability
    numerical_details = (
        data.describe().apply(lambda s: s.apply("{0:.3f}".format)).to_string()
    )

    categorical_details = json.loads(
        data.select_dtypes(include="object")
        .apply(
            lambda col: {
                "unique": col.nunique(),
                "mode": col.mode().iloc[0] if not col.mode().empty else None,
                "missing": col.isnull().sum(),
                "missing_percentage": col.isnull().mean() * 100,
            }
        )
        .to_json()
    )

    cumulative_column_details = ""
    for col_name, col_details in categorical_details.items():
        # Create a summary for each column
        column_summary = (
            f"Column '{col_name}':\n  - Unique value count: {col_details['unique']}\n"
        )
        # Add mode details if available
        if col_details["mode"].strip():
            column_summary += f"  - Most frequent value (mode): {col_details['mode']}\n"
        # Add missing values details
        column_summary += f"  - Missing values: {col_details['missing']} ({col_details['missing_percentage']:.2f}% of total)\n\n"

        cumulative_column_details += column_summary

    return {
        "column details": column_details,
        "numerical details": numerical_details,
        "categorical details": cumulative_column_details,
    }


def generate_and_exec_code(data: pd.DataFrame, dataset_info_message: dict) -> bool:
    """
    Generates and executes code using a GPT-based API and retries on failure.

    Args:
        data (pd.DataFrame): The input data to be used for code execution.
        dataset_info_message (dict): Information about the dataset to be sent to the model.

    Returns:
        bool: True if the code was successfully generated and executed, False otherwise.

    The function performs the following steps:
    1. Constructs a JSON message to be sent to the GPT-based API.
    2. Attempts to make a POST request to the API up to 3 times with exponential backoff on failure.
    3. Parses the response and attempts to execute the generated code.
    4. Logs errors and retries if the request or code execution fails.
    5. Updates the message with error information for subsequent attempts if execution fails.
    6. Returns True if the code is successfully executed, otherwise returns False.
    """

    backoff_delay = 2

    # Create the data structure to be sent to the model (GPT-based API)
    message_json = {
        "model": "gpt-4o-mini",  # Specify the model to be used
        "messages": [
            {
                "role": "system",
                "content": PYTHON_CODE_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(dataset_info_message),
            },
        ],
    }

    # Loop to retry the request for code execution up to 3 times
    for attempt in range(3):
        try:
            # Make the POST request to the OpenAI API
            response = requests.post(
                OPENAI_API,
                headers=HEADERS,
                json=message_json,
            )

            # If the request failed, log the error and retry
            if response.status_code != 200:
                logging.error(
                    f"Attempt {attempt + 1} - Failed to get response from AIProxy. "
                    f"Status code: {response.status_code}. Response: {response.text}"
                )
                logging.info(f"Retrying in {backoff_delay} seconds...")
                time.sleep(backoff_delay)
                continue

            # Execute the code from the response
            response_of_code_exec = execute_code(
                response.json()["choices"][0]["message"]["content"], data
            )

            logging.info(f"Attempt {attempt + 1} - {response_of_code_exec}")

            # If execution failed, update the message for the next attempt
            if response_of_code_exec.startswith("Failure"):
                error_message = f"Fix the code as it failed to execute with error: {response_of_code_exec}"
                data["messages"].append({"role": "user", "content": error_message})
            else:
                # If successful, break the loop
                logging.info("Code executed successfully!")
                return True

        except requests.RequestException as e:
            # Log request-related errors (network issues, timeout, etc.)
            logging.error(f"Attempt {attempt + 1} - Request failed: {str(e)}")
            logging.info(f"Retrying in {backoff_delay} seconds...")
            time.sleep(backoff_delay)

    return False


def execute_code(code: str, df: Any) -> str:
    """
    Executes a block of Python code provided within a markdown code block.

    This function extracts the Python code from a string containing markdown-style code blocks,
    checks for a code block with the "python" tag, and then attempts to execute the code using
    Python's `exec()` function. If the code execution is successful, it returns a success message.
    Otherwise, it returns a failure message with the error details.

    Args:
        code (str): A string containing markdown-formatted code, potentially including Python code in a code block.
        df (Any): A parameter to pass along during code execution (e.g., DataFrame if needed).

    Returns:
        str: A message indicating success or failure along with details in case of failure.

    Example:
        >>> execute_code("```python\nx = 5\nprint(x)```", df)
        'Success: Code executed successfully'
    """
    # Initialize code to None, to be assigned later
    extracted_code = None

    # Split the provided string by markdown code block delimiters (```)
    for line in code.split("```"):
        line = line.strip()  # Remove any leading or trailing whitespace
        # Look for a code block starting with "python"
        if line.startswith("python"):
            extracted_code = line.strip(
                "python"
            ).strip()  # Remove "python" and extra spaces
            break

    # If no Python code block is found, return failure message
    if extracted_code is None:
        return "Failure: Couldn't find code block. Please provide the code block with the 'python' tag as ```python ... ```"

    try:
        # Execute the extracted code
        exec(extracted_code)
    except Exception as e:
        # Return error message if code execution fails
        return f"Failure: Error occurred while executing the code: {traceback.format_exc()}"

    # Return success message if code is executed successfully
    return "Success: Code executed successfully"


def generate_graph_analysis(image_paths: List[Path]) -> List[str]:
    """
    Analyzes a list of images by sending them to an AI API and returns the analysis results.
    Args:
        image_paths (List[Path]): A list of paths to the images to be analyzed.
    Returns:
        List[str]: A list of analysis results for each image. If an error occurs during the analysis of an image,
                   an error message is included in the list for that image.
    """

    # List to store the image analysis results
    image_analysis = []

    # Iterate over each image and send to the API for analysis
    for image_path in image_paths:
        # Prepare data for API request
        message_json = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": GRAPH_ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and make sure 'Description of the Graph', 'Key Observations', and 'Implications' are included.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_png_image(image_path),
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
        }

        try:
            # Make the request to the AI API
            response = requests.post(OPENAI_API, headers=HEADERS, json=message_json)

            # Handle response based on status code
            if response.status_code != 200:
                logging.error(
                    f"Failed to get response from AIProxy for Image '{image_path.name}' explanation. "
                    f"Status code: {response.status_code}, Response: {response.text}"
                )
                image_analysis.append(
                    f"Failed to get response from AIProxy for Image '{image_path.name}' explanation."
                )
            else:
                image_analysis.append(
                    response.json()["choices"][0]["message"]["content"]
                )
                logging.info(f"Successfully analyzed image '{image_path.name}'.")

        except requests.RequestException as e:
            # Handle any request-related errors
            logging.error(
                f"Error occurred while sending request for image '{image_path.name}': {e}"
            )
            image_analysis.append(
                f"Error occurred while analyzing image '{image_path.name}'."
            )

    return image_analysis


def generate_readme(
    file_name: str,
    column_details: dict,
    images_path: List[Path],
    image_analysis: List[str],
) -> str:

    if not (len(images_path) == 3 and len(image_analysis) == 3):
        logging.warning("Number of images and analysis do not match.")
        sys.exit(1)

    # Prepare the data for the API request
    message_json = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": README_PROMPT.format(
                    filename=file_name,
                    column_details=column_details.get("column details", ""),
                    numerical_details=column_details.get("numerical details", ""),
                    categorical_details=column_details.get("categorical details", ""),
                    image1=images_path[0],
                    analysis1=image_analysis[0],
                    image2=images_path[1],
                    analysis2=image_analysis[1],
                    image3=images_path[2],
                    analysis3=image_analysis[2],
                ),
            },
            {
                "role": "user",
                "content": "Generate a README.md file for the content provided above",
            },
        ],
    }

    # Send the request to the AI API
    try:
        response = requests.post(OPENAI_API, headers=HEADERS, json=message_json)

        # Check for successful response
        if response.status_code != 200:
            logging.error(
                f"Failed to get response from AIProxy for README generation. "
                f"Status code: {response.status_code}, Response: {response.text}"
            )
        else:
            logging.info("Successfully generated README.md content.")
            # Retrieve the README content from the response
            readme_content = response.json()["choices"][0]["message"]["content"]

            # Extract the markdown content from the response
            for line in readme_content.strip().split("```"):
                line = line.strip()
                if line.startswith("markdown"):
                    return line.strip("markdown").strip()

    except requests.RequestException as e:
        logging.error(
            f"Error occurred while sending the request for README generation: {e}"
        )

    return None


def analyse(file_path: Path) -> dict:
    """
    Analyzes the dataset by reading the CSV file, extracting column details,
    numerical statistics, and categorical column summaries. Generates Python code
    for dynamic visualizations and prepares the content for generating a structured README.
    Args:
        file_path (Path): The file path to the CSV dataset.
    Returns:
        dict: A dictionary containing column, numerical, and categorical details of the dataset.
    """

    validate_input_file(file_path)

    # Read the CSV file into a pandas DataFrame
    data = read_csv(file_path)

    # Create a directory named after the file (without extension) for output purposes
    logging.info("Creating a new directory for the file")
    os.makedirs(file_path.stem, mode=0o777, exist_ok=True)

    # Prepare a dictionary to hold all extracted details about the dataset
    dataset_info_message = get_column_details(data)

    # Generate Python code for dynamic visualizations based on the dataset
    has_plot_generated = generate_and_exec_code(data, dataset_info_message)

    if not has_plot_generated:
        logging.warning("Failed to generate plots.")
        sys.exit(1)

    image_paths = list(Path("./").glob("*.png"))

    # Get the list of images generated from the code execution
    image_analysis = generate_graph_analysis(image_paths)

    readme_content = generate_readme(
        file_path.stem, dataset_info_message, image_paths, image_analysis
    )

    # exit if readme content is None
    if readme_content is None:
        logging.warning("README content not generated.")
        sys.exit(1)

    # Save the README content to a file
    with open(f"{file_path.stem}/README.md", "w") as f:
        f.write(readme_content)

    # Move and rename images
    for image in image_paths:
        os.rename(image, f"{file_path.stem}/{image.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.warning("Usage: python script.py <csv_file_path>")
        sys.exit(1)

    logging.info(f"Analysing file {sys.argv[1]}")
    analyse(Path(sys.argv[1]))
