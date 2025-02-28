# CDP Support Agent Chatbot

A web-based chatbot application that answers "how-to" questions related to Customer Data Platforms (CDPs): Segment, mParticle, Lytics, and Zeotap. The chatbot extracts relevant information from the official documentation of these CDPs to guide users on performing tasks or achieving specific outcomes within each platform.

![alt text](image.png)

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Data Structures](#data-structures)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Intelligent Question Answering**: Answers questions about how to perform specific tasks in four CDPs.
- **Document Retrieval**: Extracts relevant information from CDP documentation to answer user queries.
- **Cross-CDP Comparisons**: Compares approaches or functionalities between different CDP platforms.
- **Web-Based Interface**: Clean, responsive UI with chat-like interaction.
- **Documentation Caching**: Scraped documentation is cached to improve performance.
- **Markdown Support**: Formats responses with proper code blocks, lists, and links.
- **Example Questions**: Provides clickable example questions to help users get started.

## Tech Stack

### Backend
- **Python**: Core programming language chosen for its rich ecosystem of natural language processing and web development libraries.
- **Flask**: Lightweight web framework that allows for quick development of web applications with minimal boilerplate code.
- **Groq API**: AI language model API used for intelligent document retrieval and response generation - chosen for its speed and capability to handle complex queries.
- **BeautifulSoup4**: HTML parsing library used to extract content from documentation websites.
- **Requests**: HTTP library used to fetch web pages during the scraping process.

### Frontend
- **HTML/CSS/JavaScript**: Core technologies for the web interface.
- **Bootstrap 5**: CSS framework used for responsive design and pre-styled components.
- **Fetch API**: Used for making asynchronous HTTP requests to the Flask backend.

### Deployment & Infrastructure
- **Local File System**: Used for caching scraped documentation.
- **Environment Variables**: Used for storing API keys securely.

## Data Structures

### 1. Documentation Storage

The CDP documentation is stored as a nested dictionary structure:

```python
documentation = {
    "segment": [
        {
            "title": "Page Title",
            "url": "https://segment.com/docs/page-url",
            "content": "The extracted text content of the page...",
            "cdp": "segment"
        },
        # More documents...
    ],
    "mparticle": [
        # Similar structure...
    ],
    # Other CDPs...
}
```

This structure was chosen for:
- **Quick Access**: Allows O(1) access to all documents for a specific CDP
- **Content Association**: Keeps page content associated with its metadata
- **Easy Serialization**: Can be directly converted to/from JSON for caching

### 2. Message Format

Chat messages are structured as:

```javascript
{
    "message": "The content of the message",
    "isUser": true/false  // Whether the message is from the user or the bot
}
```

### 3. Document Retrieval Results

When finding relevant documents, we create a ranked list:

```python
[
    {
        "title": "Page Title",
        "url": "https://segment.com/docs/page-url",
        "content": "The extracted text content of the page...",
        "cdp": "segment",
        "similarity": 0.85  # Similarity score from Groq ranking
    },
    # More documents...
]
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Groq API key

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/cdp-support-agent.git
cd cdp-support-agent
```

### Step 2: Create a virtual environment
```bash
python -m venv venv
```

### Step 3: Activate the virtual environment
On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

### Step 4: Install dependencies
```bash
pip install flask requests beautifulsoup4 groq
```

### Step 5: Set environment variables
```bash
# On Windows (Command Prompt)
set GROQ_API_KEY=your_groq_api_key_here

# On Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_api_key_here"

# On macOS/Linux
export GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Running the application
```bash
python app.py
```

The application will start on `http://localhost:5000`. Open this URL in your browser to use the chatbot.

### First-time Usage

When you first run the application, it will:
1. Create a `templates` directory with the HTML template
2. Scrape the documentation from the CDP websites (this may take a few minutes)
3. Cache the documentation for future use

After the initial setup, the application will load much faster on subsequent runs.

### Example Questions

- "How do I set up a new source in Segment?"
- "How can I create a user profile in mParticle?"
- "How do I build an audience segment in Lytics?"
- "How can I integrate my data with Zeotap?"
- "How does Segment's audience creation process compare to Lytics'?"

## Architecture

### Component Overview

The application consists of three main components:

1. **Documentation Manager**
   - Scrapes and caches CDP documentation
   - Loads documentation from cache when available

2. **Query Processor**
   - Identifies which CDP the query is about
   - Determines if the query is CDP-related
   - Retrieves relevant documentation

3. **Response Generator**
   - Uses Groq API to generate responses based on documentation
   - Formats responses for display in the UI

### Process Flow

1. User submits a question via the web interface
2. Backend identifies if the question is CDP-related
3. Backend retrieves relevant documentation
4. Groq API generates a response based on the documentation
5. Response is returned to the frontend and displayed to the user

## Design Decisions

### Why Retrieval Augmented Generation (RAG)?

We chose a RAG approach (using Groq + retrieved documentation) over traditional search because:

1. **Better Understanding**: LLMs can understand the semantic meaning of questions, not just keywords
2. **Natural Responses**: Generates cohesive, human-like responses that directly address the question
3. **Context Awareness**: Considers multiple documents in context to provide comprehensive answers

### Why Flask?

Flask was chosen for its:
1. **Simplicity**: Minimal framework with little boilerplate code
2. **Flexibility**: Easy to extend with additional features
3. **Compatibility**: Works well with Python-based AI components

### Why Document Caching?

Documentation is cached to:
1. **Reduce Load**: Minimize repeated requests to documentation websites
2. **Improve Speed**: Significantly faster startup times after initial run
3. **Work Offline**: Allow the application to work without internet after initial setup

### Why Groq Over Custom Vector Search?

Using Groq for document ranking and response generation provides:
1. **Simplified Implementation**: Eliminates need for custom vector databases
2. **Better Relevance**: LLM-based ranking often outperforms traditional similarity metrics
3. **Coherent Responses**: Generates well-structured answers from multiple documentation sources

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.