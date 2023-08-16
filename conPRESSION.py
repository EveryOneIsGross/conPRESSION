'''Explanation of Error States:

Context Truncation:
If the context provided by the user exceeds a certain character limit (e.g., 2000 characters), the chat agent will truncate it to fit within that limit.
As a result, the agent will consider only the latest portion of the context for generating responses.

Column Limit Determination:
The column limit for shaping the encoded response into blocks is determined based on the number of spaces and the total length of the response.
This column limit is used to create square-shaped encoded response blocks.
In some cases, this approach might not yield perfectly shaped blocks, especially if the response contains an unusual number of spaces.

Encoding and Padding:
The code uses a special marker to denote the beginning and end of encoded content.
When generating encoded response blocks, padding is added using dots to achieve a square slab shape.
Padding may not always yield perfect results, and in some cases, it might add more padding than necessary.
It can grok those errors and wil defs create soup.

User: sumamrise

Pre-encoded Response: ] th[seo] [tomor] ry] ereas] ollmors] erses] nthes] orts] ollcen] tate] ollme] sot] erth] ynapr] olls] ethe] ollst] orch[e] sth] olltea] ryto] ollmors] erses] nthes] orts] ollcen] tate] ollme] sot] erth] ynapr] olls] ethe] ollst] orch[e] sth] olltea] ryto] ollmors] erses] nthes] orts] ollcen] tate] ollme] sot] erth] ynapr] olls] ethe] ollst] 
orch[e] sth] olltea] ryto] ollmors] erses] nthes] orts] ollcen] tate] ollme] sot] erth] ynapr] olls] ethe] ollst] orch[e] 
sth] olltea] ryto] ollmors] er


        "[[19x19-005]-rsthyollteatrytoool]]]",
        "[[19x19-004]-lmorsrersesynthesto]]]",
        "[[19x19-003]-rtsoollcenrtateyoll]]]",
        "[[19x19-002]-metsotoerthrynapryo]]]",
        "[[19x19-001]-llstetheoollstrorch]]]",
        "[[19x19-000]-eysthtollteaorytoro]]]"
    ],
    "Decoded Summary": [
        "",
        "ynaprynaprynaprynaprynaprynaprynaprynaprynaprynaprynaprynaprynaprynaprynaprtomorthtatetatetatetatetatetatetatetatetatetatetatetatetatetateta"

Disclaimer:

This code is a proof of concept and example implementation of a chat agent. 
It demonstrates the use of context management, encoding, and shaping responses into blocks. 
'''
import gpt4all
import json
from rake_nltk import Rake
import string
import math


# Global markers for encoding
marker_start = "[["
marker_end = "]]"

try:
    with open("recorded_data.json", "r") as json_file:
        recorded_data = json.load(json_file)
except FileNotFoundError:
    recorded_data = []


# Initialize models
model = gpt4all.GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")
r = Rake(min_length=1, max_length=3)  # Adjust keyword extraction parameters

def strip_punctuation(text):
    """Removes punctuation, newline characters, and other special characters from the text."""
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove newline characters
    text = text.replace("\n", "")
    
    # Remove other special characters (like non-breaking spaces)
    text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace())
    return text

def encode(text, summary):
    # Strip punctuation from the text before encoding
    text = strip_punctuation(text)
    
    if not summary:
        return text
    
    encoded = list(text)
    insert_indices = [i for i, char in enumerate(encoded) if char in [" ", ",", ".", "!", "?", ";", ":"]]
    for idx, insert_idx in enumerate(insert_indices):
        char_to_insert = summary[idx % len(summary)]
        encoded[insert_idx] = char_to_insert
    return ''.join(encoded)

def trim_content(line, max_length):
    """Trim the content of the line to ensure it doesn't overflow."""
    # Extract the content from the line
    content_start = line.find("]-") + 2
    content_end = line.rfind("]]]")
    content = line[content_start:content_end]

    # Remove special characters
    content = content.replace("\n", " ").replace("[", "").replace("]", "")

    # Trim the content if it's still too long
    while len(line) > max_length:
        trim_amount = min(len(line) - max_length, len(content))
        content = content[:-trim_amount]
        line = line[:content_start] + content + line[content_end:]

    return line

def check_for_overflowing_lines(encoded_lines, max_length):
    """Check for overflowing lines and correct them."""
    overflowing_lines = [line for line in encoded_lines if len(line) > max_length]
    corrected_lines = [trim_content(line, max_length) for line in overflowing_lines]
    return corrected_lines

def generate_encoded_response_blocks(encoded_response, column_limit):
    """Generate encoded response blocks conforming to strict shape dimensions."""
    
    # Split the encoded response into slabs of the given column limit
    encoded_lines = [
        f"[[{column_limit}x{column_limit}-{str(index).zfill(3)}]-{encoded_response[i:i+column_limit]}{marker_end}]"
        for index, i in enumerate(range(0, len(encoded_response), column_limit))
    ]
    
    # Remove the last slab if it contains padding or is shorter than column limit
    if encoded_lines and (encoded_lines[-1][-column_limit-3:-3].count('.') > 0 or len(encoded_lines[-1]) < column_limit):
        encoded_lines = encoded_lines[:-1]

    # Check for overflowing lines and correct them
    max_length = len("[[10x10-000]-\nWeucansus]]]")  # Using one of the lines as a reference for the maximum length
    corrected_lines = check_for_overflowing_lines(encoded_lines, max_length)
    
    return corrected_lines

def basic_keyword_extraction(text):
    keywords = [word for word in text.split() if len(word) > 3]
    return ' '.join(keywords)


def decode(encoded_response, marker_start, marker_end):
    columns = [col[len(marker_start)+4:-len(marker_end)] for col in encoded_response]
    extracted_keywords = [basic_keyword_extraction(col) for col in columns]
    keyword_list = []
    for col, keyword in zip(columns, extracted_keywords):
        for word in keyword.split():
            index = col.find(word)
            punctuation_before = "" if index == 0 or col[index-1].isalnum() else col[index-1]
            punctuation_after = "" if index+len(word) == len(col) or col[index+len(word)].isalnum() else col[index+len(word)]
            keyword_list.append(punctuation_before + word + punctuation_after)
    keyword_string = ' '.join(keyword_list)
    for keyword in keyword_list:
        encoded_response = encoded_response.replace(keyword, '', 1)
    decoded_summary = ''.join(encoded_response)
    return keyword_string, decoded_summary

def truncate_context(context, max_chars=256):
    """
    Truncates the context if it exceeds the specified number of characters.

    Parameters:
    context (str): The context string to be truncated.
    max_chars (int): The maximum number of characters allowed for the context.

    Returns:
    str: The truncated context.
    """
    # Truncate the context if it exceeds max_chars
    if len(context) > max_chars:
        context = context[-max_chars:]  # Retain the latest characters for freshness
    
    return context

class ChatAgentResponse:
    def __init__(self, model):
        self.model = model

    def generate_response(self, prompt, context=""):
        full_prompt = f"{context} {prompt}" if context else prompt
        tokens = [token for token in model.generate(full_prompt, max_tokens=1024, streaming=True)]
        return ''.join(tokens)

class ChatAgentSummary:
    def __init__(self, model):
        self.model = model

    def generate_summary(self, response):
        """
        Generates a concise summary of the response.

        Parameters:
        response (str): The response text to be summarized.

        Returns:
        str: A concise summary of the response.
        """
        # Extract a summary based on the number of words in the response
        summary = ' '.join(response.split()[:response.count(" ")])
        
        # If the summary is still too long, use keyword extraction to make it more concise
        if len(summary) > response.count(" "):
            summary = rake_keyword_extraction(response).replace(" ", "")
        
        return summary


def rake_keyword_extraction(text):
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()
    return ''.join(keywords)

def pad_encoded_response(encoded_response, column_limit):
    """Pad the encoded response to achieve a square slab shape."""
    total_chars = len(encoded_response)
    padding_needed = max(0, column_limit * column_limit - total_chars)
    return encoded_response + '.' * padding_needed  # Padding with dots


def generate_encoded_response_blocks(encoded_response, column_limit):
    """Generate encoded response blocks conforming to strict shape dimensions."""
    
    # Calculate the number of lines needed
    num_lines = math.ceil(len(encoded_response) / column_limit)
    
    # Determine the starting index for the count
    max_lines = num_lines - 1  # The index starts from 0
    
    # Split the encoded response into slabs of the given column limit
    encoded_lines = [
        f"[[{column_limit}x{column_limit}-{str(max_lines - index).zfill(3)}]-{encoded_response[i:i+column_limit]}{marker_end}]"
        for index, i in enumerate(range(0, len(encoded_response), column_limit))
    ]
    
    # Check if the last line is shorter than the column limit and pad it
    if encoded_lines and len(encoded_lines[-1]) < column_limit + len("[[10x10-000]-]]"):
        padding_needed = column_limit - len(encoded_lines[-1]) + len("[[10x10-000]-]]")
        encoded_lines[-1] = encoded_lines[-1][:-len(marker_end)] + '.' * padding_needed + marker_end
    
    return encoded_lines


def remove_markers_from_slab(slab):
    # Strip out the start and end markers
    stripped_slab = slab[len(marker_start)+11:-len(marker_end)]
    # Remove the portion between the "-" and "]"
    content_start = stripped_slab.find("-") + 1
    return stripped_slab[content_start:]

def find_nearest_square_factors_for_length(length):
    """Find two factors of the given length that are closest to each other."""
    for i in range(int(length**0.5), 0, -1):
        if length % i == 0:
            return i, length // i
    return 1, length  # Shouldn't happen, but just in case

def determine_column_limit_based_on_spaces_and_length(encoded_response):
    """Determine the optimal column limit based on the number of spaces and total length."""
    num_spaces = encoded_response.count(" ")
    total_length = len(encoded_response)
    
    width, height = find_nearest_square_factors_for_length(total_length)
    
    # To decide which one (width or height) to use for the column limit, 
    # choose the one that's closer to the square root of total_length.
    if abs(width - total_length**0.5) < abs(height - total_length**0.5):
        column_limit = width
    else:
        column_limit = height

    # Ensure that the column limit is at least 8
    column_limit = max(column_limit, 8)
    
    return column_limit

def chat_agent():
    context = "" 
    summary = "" 

    response_agent = ChatAgentResponse(model)
    summary_agent = ChatAgentSummary(model)

    while True:
        user_input = input("User: ")

        # Check if the user wants to exit
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Truncate context if it's too long
        context = truncate_context(context, 2000)  # Using 2000 characters as an example limit

        
        response = response_agent.generate_response(user_input, context)
        summary = strip_punctuation(summary_agent.generate_summary(response))

        print("\nPre-encoded Response:", response)
        print("Pre-encoded Summary:", summary)

        encoded_response = encode(response, summary)
        column_limit = determine_column_limit_based_on_spaces_and_length(encoded_response)
        
        encoded_response = pad_encoded_response(encoded_response, column_limit)
        encoded_lines = generate_encoded_response_blocks(encoded_response, column_limit)
        encoded_input = encode(user_input, summary)
        encoded_summary = encode(summary, summary)
        
        data = {
            "User": encoded_input,
            "Response": encoded_response,
            "Summary": encoded_summary,
            "Encoded Response": encoded_lines,
            "Decoded Summary": decode(encoded_summary, marker_start, marker_end)
        }

        # Append the latest entry to the history
        recorded_data.append(data)

        with open("recorded_data.json", "w") as json_file:
            json.dump(recorded_data, json_file, indent=4)

        print(json.dumps(data, indent=4))
        
        # Get the decoded summary from the latest entry
        decoded_summary, _ = data["Decoded Summary"]
        
        # Generate the cleaned context from the latest encoded response
        cleaned_encoded_response = ' '.join([remove_markers_from_slab(slab) for slab in data["Encoded Response"]])

        # Combine the decoded summary and cleaned encoded response to form the context
        context = f"{decoded_summary} {cleaned_encoded_response}"


if __name__ == "__main__":
    chat_agent()
