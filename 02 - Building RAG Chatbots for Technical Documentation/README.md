
#
# Car Manual Chatbot with Retrieval-Augmented Generation (RAG)

This project is a proof-of-concept (POC) context-aware chatbot for a well-known car manufacturer exploring the integration of large language models (LLMs) into vehicles. The chatbot is designed to provide real-time guidance to drivers by interpreting and explaining car warning messages. The project demonstrates how car manuals, specifically from an MG ZS compact SUV, can be used to generate accurate, context-aware responses to driver queries about car warning

##
## Setting Up the OpenAI API Key

This project requires an OpenAI API key to connect to the language model. Follow these steps to create an OpenAI account, generate an API key, and securely store it.

### Steps to Set Up Your API Key

1.  **Create an OpenAI Account**: Go to [OpenAI's API signup page](https://platform.openai.com/signup) and follow the instructions to set up your developer account.
2.  **Generate an API Key**: Access the [API keys page](https://platform.openai.com/account/api-keys) and create a new secret key. **Copy it immediately** as it will only be shown once.
3.  **Store the Key as an Environment Variable**: Set the key as an environment variable named `OPENAI_API_KEY`:
    -   Use your environment settings or add it to a `.env` file if working locally.
4.  **Load the Key in Code**: Load this environment variable within your project to securely access the API.

## Project Approach

1.  **Document Loading**: The car manual, stored as an HTML file (`mg-zs-warning-messages.html`), is loaded into the system. This document contains various warning messages along with their explanations and suggested driver actions.
2.  **Content Splitting**: To enable efficient retrieval, the content of the manual is divided into small, overlapping segments or “chunks.” Each chunk is designed to capture context from adjacent sections for better relevance.
3.  **Vector Embedding**: Each text chunk is transformed into a vector embedding, allowing the system to measure the similarity between the query and document content.
4.  **Retriever**: A retriever component uses these embeddings to locate the most relevant chunks based on the user's question.
5.  **LLM and Prompt Template**: Retrieved content is formatted into a structured prompt, combining relevant manual text with the user’s question. The prompt is passed to an LLM (e.g., OpenAI’s GPT-3.5) to generate a human-friendly response.
6.  **RAG Chain**: A RAG chain orchestrates the entire process—retrieving information, constructing a prompt, and generating a response—into a seamless query-response experience.

### Use Case

A user inquires about specific car warning messages, such as "The Gasoline Particular Filter Full warning has appeared. What does this mean, and what should I do about it?" The chatbot will use relevant sections of the manual to provide a clear, direct response.

    


## Key Components

-   **LangChain**: A framework to handle document loading, content splitting, and pipeline management.
-   **Chroma Vector Store**: A vector store that manages embeddings, enabling quick retrieval of relevant manual sections.
-   **OpenAI's GPT-3.5**: The language model that generates coherent, user-friendly answers based on the retrieved content.

## Benefits of This Approach

-   **Accuracy**: Combining information retrieval with language generation provides specific, manual-based answers.
-   **Scalability**: This RAG approach is adaptable for other document types and query formats.
-   **Context-Aware Responses**: Unlike traditional search, this system provides responses in a conversational and contextually enriched format.

## Example Workflow

A user query about a specific car warning triggers the RAG system to:

1.  Retrieve relevant sections from the manual.
2.  Format the retrieved information into a prompt for the language model.
3.  Generate a response that directly addresses the user’s question.

### Potential Extensions

-   **Multi-Language Support**: Expanding the system to handle queries in multiple languages.
-   **Caching Mechanisms**: Adding caching for frequently asked questions to reduce API usage and improve response times.

## Conclusion

This project is an effective implementation of RAG for querying structured documents, specifically car manuals. By integrating document retrieval and natural language generation, it offers accurate, accessible answers for car warning-related queries.
