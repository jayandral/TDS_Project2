# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests",
#   "pandas",
#   "seaborn",
# ]
# ///

from pathlib import Path
from typing import Any

import pandas as pd
import os
import sys
import io
import json
import logging
import requests
import base64
import traceback

# Constants for API interaction
OPENAI_API = os.getenv(
    "OPENAI_API_URL", "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
)

# Check for the API Token, which is necessary for authentication
AIPROXY_API_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_API_TOKEN:
    raise ValueError("AIPROXY_API_TOKEN environment variable is not set.") # Ensure token is present for authentication

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
    # Open the image file in binary read mode and encode its content to base64
    with open(image_path, "rb") as image_file:
        # Return the base64 string with the appropriate prefix for embedding in web applications
        return "data:image/png;base64," + base64.b64encode(image_file.read()).decode("utf-8")


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
    
    # Log the start of file reading
    logging.info("Reading file")

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path, encoding="unicode_escape")

    # Log the creation of a new directory for the file
    logging.info("Creating a new directory for the file")

    # Create a directory named after the file (without extension) for output purposes
    os.makedirs(file_path.stem, mode=0o777, exist_ok=True)

    # Create a StringIO buffer to capture the DataFrame info output
    buffer = io.StringIO()
    data.info(buf=buffer)  # Capture DataFrame info in the buffer
    info_string = buffer.getvalue()  # Get the content of the buffer

    # Extract column details from the info string (excluding header/footer lines)
    column_details = "\n".join(info_string.split("\n")[3:-3])

    # Generate numerical details using describe() method with formatting for readability
    numerical_details = (
        data.describe().apply(lambda s: s.apply("{0:.3f}".format)).to_string()
    )

    # Initialize categorical details as an empty string
    categorical_details = ""

    # Iterate through each column to analyze categorical data
    for col in data.columns:
        if data[col].dtype == "object":  # Check if the column is categorical
            value_counts = data[
                col
            ].value_counts()  # Get the value counts for the column
            value, value_count = value_counts.idxmax().strip(), value_counts.max()

            # Determine mode value and frequency if applicable
            mode_value = value if value_count > 1 and value != "" else None
            mode_frequency = value_count if value_count > 1 and value != "" else None

            # Calculate missing values count and percentage
            missing_count = data[col].isnull().sum()
            total_count = len(data[col])
            missing_percentage = (missing_count / total_count) * 100

            # Build the details string for the categorical column
            column_summary = (
                f"Column '{col}':\n" f"  - Unique value count: {data[col].nunique()}\n"
            )

            # Add mode details if available
            if mode_value is not None:
                column_summary += f"  - Most frequent value (mode): {mode_value} ({mode_frequency} occurrences)\n"

            # Add missing values details
            column_summary += (
                f"  - Missing values: {missing_count} ({missing_percentage:.2f}% of total)\n"
                f"\n"
            )

            # Append the categorical column details to the overall string
            categorical_details += column_summary

    # Prepare a dictionary to hold all extracted details about the dataset
    dataset_info_message = {
        "column details": column_details,
        "numerical details": numerical_details,
        "categorical details": categorical_details,
    }

    # Create the data structure to be sent to the model (GPT-based API)
    message_json = {
        "model": "gpt-4o-mini",  # Specify the model to be used
        "messages": [
            {
                "role": "system",
                "content": PYTHON_CODE_PROMPT,
            },  # System message (prompt for the model)
            {
                "role": "user",
                "content": json.dumps(dataset_info_message),
            },  # User message with dataset info
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
                continue  # Retry if the request fails

            # Parse the response JSON
            response_json = response.json()
            # Execute the code from the response
            response_of_code_exec = execute_code(
                response_json["choices"][0]["message"]["content"], data
            )

            logging.info(f"Attempt {attempt + 1} - {response_of_code_exec}")

            # If execution failed, update the message for the next attempt
            if response_of_code_exec.startswith("Failure"):
                error_message = f"Fix the code as it failed to execute with error: {response_of_code_exec}"
                if len(data["messages"]) == 2:
                    data["messages"].append({"role": "user", "content": error_message})
                else:
                    data["messages"][2]["content"] = error_message
            else:
                # If successful, break the loop
                logging.info("Code executed successfully!")
                break

        except requests.RequestException as e:
            # Log request-related errors (network issues, timeout, etc.)
            logging.error(f"Attempt {attempt + 1} - Request failed: {str(e)}")
            continue

        except KeyError as e:
            # Log key errors if the response doesn't have expected keys
            logging.error(
                f"Attempt {attempt + 1} - KeyError in response parsing: {str(e)}"
            )
            break  # Exiting the loop as the response format is not as expected

    images = list(Path("./").glob("*.png"))
    image_description = []

    # Check if no images are found
    if len(images) == 0:
        logging.warning("No images generated.")
        sys.exit(1)

    # Iterate over each image and send to the API for analysis
    for image in images:
        # Prepare data for API request
        message_json = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": GRAPH_ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_png_image(
                                    image
                                ),  # Ensure this function is correct
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
                    f"Failed to get response from AIProxy for Image '{image.name}' explanation. "
                    f"Status code: {response.status_code}, Response: {response.text}"
                )
                image_description.append(
                    f"Failed to get response from AIProxy for Image '{image.name}' explanation."
                )
            else:
                image_description.append(
                    response.json()["choices"][0]["message"]["content"]
                )
                logging.info(f"Successfully analyzed image '{image.name}'.")

        except requests.RequestException as e:
            # Handle any request-related errors
            logging.error(
                f"Error occurred while sending request for image '{image.name}': {e}"
            )
            image_description.append(
                f"Error occurred while analyzing image '{image.name}'."
            )

    # Format the README content using the provided details
    formatted_readme = README_PROMPT.format(
        filename=file_path.stem.upper(),
        column_details=column_details,
        numerical_details=numerical_details,
        categorical_details=categorical_details,
        image1=images[0],
        analysis1=image_description[0],
        image2=images[1],
        analysis2=image_description[1],
        image3=images[2],
        analysis3=image_description[2],
    )

    # Prepare the data for the API request
    message_json = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": formatted_readme},
            {
                "role": "user",
                "content": "Generate a README.md file for the content provided above",
            },
        ],
    }

    readme_content = None

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

    except requests.RequestException as e:
        logging.error(
            f"Error occurred while sending the request for README generation: {e}"
        )

    # If README content was not generated, log a warning and exit
    if readme_content is None:
        logging.warning("README content not generated.")
        sys.exit(1)

    final_readme_content = None

    # Extract the markdown content from the response
    for line in readme_content.strip().split("```"):
        line = line.strip()
        if line.startswith("markdown"):
            final_readme_content = line.strip("markdown").strip()
            break

    # If no markdown content was found, log a warning and exit
    if final_readme_content is None:
        logging.warning("No markdown content found in the README.")
        sys.exit(1)

    # Save the README content to a file
    with open(f"{file_path.stem}/README.md", "w") as f:
        f.write(final_readme_content)

    # Move and rename images
    for image in images:
        os.rename(image, f"{file_path.stem}/{image.name}")


if __name__ == "__main__":
    args = sys.argv

    if len(args) > 2:
        logging.warning("Too many arguments provided")
    elif len(args) != 2:
        logging.warning("Provide a csv file path")
    else:
        file = args[1]
        if not file.endswith(".csv"):
            logging.warning("Need a csv file but got a different file type")
            sys.exit(1)

        logging.info(f"Analysing file {args[1]}")
        analyse(Path(args[1]))
