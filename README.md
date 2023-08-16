conPRESSION
---

'''a novel noise-tolerant compression/recovery method for piping truncation protected context into agents'''


```
User: how does one pursue their True Will?

Pre-encoded Response: 
I have a Will, but I don't know what to do with it. What should I do?
Pre-encoded Summary: know
{
    "User": "how does one pursue their True Will?",
    "Response": "IkhavenaoWillwbutkIndontoknowwwhatktondoowithwitkWhatnshouldoIwdo",
    "Summary": "know",
    "Encoded Response": [
        "[[8x9-008]-Ikhavena]]]",
        "[[8x9-007]-oWillwbu]]]",
        "[[8x9-006]-tkIndont]]]",
        "[[8x9-005]-oknowwwh]]]",
        "[[8x9-004]-atktondo]]]",
        "[[8x9-003]-owithwit]]]",
        "[[8x9-002]-kWhatnsh]]]",
        "[[8x9-001]-ouldoIwd]]]",
        "[[8x9-000]-o]]]"
    ],
    "Decoded Summary": [
        "",
        "know"
    ]
}
User:
```

This project encodes chatbot responses into uniquely structured slabs. It takes user input, generates a response via a local llm, summarizes the response, and then encodes both the response and summary into a structured format. This format is beneficial for visual representation, compact storage, and potentially for advanced processing later on.

The code employs several utility functions to process and manipulate text, ensuring it conforms to the desired encoding format. The key features of this project include:

Text Encoding: Text is encoded with markers that embed a summary into the response. This encoding helps preserve the essence of the message even when viewing a condensed version.
Response Generation: Using GPT4all a local llm model is used to generate chatbot responses based on user input and context.

Keyword Extraction: The project uses the rake_nltk library to extract keywords from the text, which are then used in the encoding process.

Structured Slabs: Responses are broken into structured slabs that are of a specific shape (e.g., 10x10, 17x17). This format is beneficial for visual representation and further processing.

How It Works:

User Interaction: The user provides input to the chatbot.
Response Generation: The chatbot, powered by the GPT-4 model, generates a response based on the user's input and the context.
Summarization: The response is summarized, and this summary is used in the encoding process.
Encoding: The response is encoded into structured slabs using markers. The encoding process embeds the summary into the response.
Decoding: The encoded response can be decoded to extract the original response and the summary.
Usage

To use the project:

Ensure you have all required libraries installed.
Execute the script. The chatbot will await user input.
Interact with the chatbot, and view the encoded responses and their corresponding summaries.


Code Structure:

strip_punctuation(): Removes punctuation and special characters from a string.
encode(): Encodes a text using a provided summary.
decode(): Decodes an encoded response to retrieve the original response and summary.
ChatAgentResponse and ChatAgentSummary: Classes to generate chatbot responses and their summaries.
rake_keyword_extraction(): Extracts keywords from a text using the RAKE algorithm.
chat_agent(): The main function that drives the interaction loop, encoding, and decoding processes.


To-Do List:

Enhanced Encoding Logic: Improve the encoding logic to embed more nuanced data into the slabs, such as sentiment or other metadata.
Dynamic Column Limit: Adjust the column limit dynamically based on the complexity or length of the response to optimize the slab shape.
Context Management: Improve how context is managed and fed back into the chatbot for more coherent multi-turn conversations.
Optimization: Enhance performance, especially in the encoding and decoding processes.
Interface: Develop a user-friendly interface for easier interaction and visualization of the encoded slabs.
Error Handling: Incorporate better error handling and validation checks to ensure robustness.
