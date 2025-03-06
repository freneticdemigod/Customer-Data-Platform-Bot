# app.py
import os
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
import logging
import json
import groq
from urllib.parse import urlparse
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CDPSupportAgent:
    def __init__(self, api_key=None, cache_dir="./cdp_cache"):
        self.cache_dir = cache_dir
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass as parameter.")
        
        self.groq_client = groq.Client(api_key=self.api_key)
        
        self.cdps = {
            "segment": "https://segment.com/docs/?ref=nav",
            "mparticle": "https://docs.mparticle.com/",
            "lytics": "https://docs.lytics.com/",
            "zeotap": "https://docs.zeotap.com/home/en-us/"
        }
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Load or scrape documentation 
        self.documentation = self.load_documentation()

    def is_cdp_related(self, query: str) -> bool:
        """Check if query is related to CDPs using Groq API"""
        try:
            prompt = f"""Is the following query related to Customer Data Platforms (CDPs) like 
            Segment, mParticle, Lytics, or Zeotap? Answer with just 'yes' or 'no'.
            
            Query: {query}
            """
            
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",  # Using smaller model for efficiency
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            answer = response.choices[0].message.content.strip().lower()
            return "yes" in answer
        
        except Exception as e:
            logger.error(f"Error checking if query is CDP-related: {str(e)}")
            # Fall back to simple keyword check
            cdp_terms = ["cdp", "customer data", "segment", "mparticle", "lytics", "zeotap"]
            return any(term in query.lower() for term in cdp_terms)
    
    def answer_question(self, query: str) -> Dict:
        """Answer a user question using Groq API and retrieved documentation"""
        # Check if query is relevant to CDPs
        if not self.is_cdp_related(query):
            logger.info("Query not related to CDPs")
            return {
                "text": "I'm a CDP support agent focused on helping with questions about Segment, mParticle, Lytics, and Zeotap. Please ask a question related to these Customer Data Platforms.",
                "source": "off-topic"
            }
        
        # Identify which CDP the query is about
        identified_cdp = None
        for cdp_name in self.cdps.keys():
            if cdp_name.lower() in query.lower():
                identified_cdp = cdp_name
                break
        
        # Retrieve relevant documents
        relevant_docs = self.find_relevant_documents(query, cdp=identified_cdp)
        
        # Check if we have synthetic documents (meaning scraping failed)
        has_synthetic = any(doc.get('synthetic', False) for doc in relevant_docs)
        
        if not relevant_docs:
            logger.info("No relevant documents found")
            return {
                "text": "I couldn't find specific information about that in the CDP documentation. Could you please rephrase your question or provide more details?",
                "source": "no-docs"
            }
        
        # If we have synthetic documents or no documents for the identified CDP,
        # we'll send the query directly to Groq without context
        if has_synthetic or len(relevant_docs) < 2:
            logger.info(f"Using direct Groq query for {identified_cdp or 'unknown CDP'}")
            
            # Create a more specific prompt based on the CDP
            cdp_specific_prompt = ""
            if identified_cdp:
                cdp_specific_prompt = f"""
                You are answering a question about {identified_cdp.capitalize()}, which is a Customer Data Platform (CDP).
                
                About {identified_cdp.capitalize()}:
                """
                
                if identified_cdp == "segment":
                    cdp_specific_prompt += """
                    Segment is a customer data platform that helps businesses collect, clean, and control their customer data.
                    It allows companies to collect user events from any source (website, mobile apps, server, etc.),
                    and send that data to marketing, analytics, and data warehouse tools.
                    
                    Key features include:
                    - Sources: Collect data from websites, mobile apps, servers, and cloud apps
                    - Connections: Route customer data to over 300+ tools with the flip of a switch
                    - Protocols: Standardize data collection and governance across the organization
                    - Personas: Build unified customer profiles and audiences
                    - Journeys: Create cross-channel campaigns with personalized messages
                    """
                elif identified_cdp == "mparticle":
                    cdp_specific_prompt += """
                    mParticle is a customer data platform that helps teams collect and connect their data.
                    It enables businesses to collect data from multiple sources and send it to various
                    analytics, marketing, and data warehouse platforms.
                    
                    Key features include:
                    - Data collection from web, mobile, and server
                    - Identity resolution and user profiles
                    - Audience segmentation
                    - Real-time data filtering and forwarding
                    - Data governance and quality tools
                    """
                elif identified_cdp == "lytics":
                    cdp_specific_prompt += """
                    Lytics is a customer data platform that uses machine learning to help marketers
                    and digital teams automate personalized marketing experiences. It focuses on
                    creating unified customer profiles and predictive insights.
                    
                    Key features include:
                    - Behavioral scoring and segmentation
                    - Predictive marketing models
                    - Content affinity engine
                    - Unified customer profiles
                    - Cross-channel orchestration
                    """
                elif identified_cdp == "zeotap":
                    cdp_specific_prompt += """
                    Zeotap is a customer intelligence platform (CDP) that helps brands better understand
                    their customers and predict behaviors. It specializes in identity resolution and
                    enrichment with high-quality data.
                    
                    Key features include:
                    - Customer identity resolution
                    - Data enrichment
                    - Predictive audiences
                    - Privacy and consent management
                    - Cross-channel activation
                    """
            
            # Direct Groq query
            prompt = f"""You are a helpful CDP (Customer Data Platform) support agent that specializes in 
            answering questions about Segment, mParticle, Lytics, and Zeotap.
            
            {cdp_specific_prompt}
            
            Please answer this question as accurately as possible based on your knowledge:
            
            User Question: {query}
            
            For "how-to" questions, provide clear step-by-step instructions.
            If you don't know the specific details, provide general guidance based on common patterns in CDPs.
            Make it clear that you're providing general information that may need to be adapted to the specific platform.
            
            Add "[Note: This response is based on general knowledge rather than specific documentation]" at the end of your answer.
            """
            
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.2
                )
                
                answer = response.choices[0].message.content.strip()
                logger.info(f"Generated direct Groq response for query about {identified_cdp or 'unknown CDP'}")
                return {
                    "text": answer,
                    "source": "direct-groq",
                    "cdp": identified_cdp
                }
            except Exception as e:
                logger.error(f"Error generating direct answer with Groq API: {str(e)}")
                return {
                    "text": f"I encountered an error while trying to answer your question. Please try again later. Error: {str(e)}",
                    "source": "error"
                }
        
        # Normal flow with document context
        logger.info(f"Using documentation-based response for {identified_cdp or 'unknown CDP'}")
        # Prepare context from retrieved documents
        context = "\n\n---\n\n".join([
            f"CDP: {doc['cdp'].upper()}\nTitle: {doc['title']}\nURL: {doc['url']}\n\n{doc['content'][:1500]}" 
            for doc in relevant_docs
        ])
        
        # Generate answer using Groq
        try:
            prompt = f"""You are a helpful CDP (Customer Data Platform) support agent that specializes in 
            answering questions about Segment, mParticle, Lytics, and Zeotap. Use the following documentation 
            excerpts to answer the user's question. If the information isn't in the documentation, say so.
            
            For "how-to" questions, try to provide clear step-by-step instructions.
            If the question compares different CDPs, highlight the key differences.
            Always cite the source URL at the end of your answer.
            
            User Question: {query}
            
            Documentation Excerpts:
            {context}
            """
            
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.2
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated documentation-based response for {identified_cdp or 'unknown CDP'}")
            return {
                "text": answer,
                "source": "doc-based",
                "cdp": identified_cdp,
                "urls": [doc["url"] for doc in relevant_docs]
            }
            
        except Exception as e:
            logger.error(f"Error generating answer with Groq API: {str(e)}")
            return {
                "text": f"I encountered an error while trying to answer your question. Please try again later. Error: {str(e)}",
                "source": "error"
            }

    def load_documentation(self) -> Dict[str, List[Dict]]:
        """Load documentation from cache or scrape if not available"""
        all_docs = {}
        
        for cdp, url in self.cdps.items():
            cache_file = os.path.join(self.cache_dir, f"{cdp}_docs.json")
            
            if os.path.exists(cache_file):
                logger.info(f"Loading {cdp} documentation from cache")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    all_docs[cdp] = json.load(f)
            else:
                logger.info(f"Scraping {cdp} documentation from {url}")
                all_docs[cdp] = self.scrape_documentation(cdp, url)
                # Save to cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(all_docs[cdp], f, ensure_ascii=False, indent=2)
        
        return all_docs
    
    def scrape_documentation(self, cdp: str, base_url: str) -> List[Dict]:
        """Scrape the documentation from the given URL"""
        documents = []
        visited_urls = set()
        urls_to_visit = [base_url]
        
        # Limit scraping to avoid overloading the server
        max_pages = 50  # Reducing from 100 to 50 to make it faster
        count = 0
        
        base_domain = urlparse(base_url).netloc
        
        while urls_to_visit and count < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            visited_urls.add(current_url)
            count += 1
            
            try:
                logger.info(f"Scraping {current_url}")
                response = requests.get(current_url, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch {current_url}: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract content
                title = soup.title.string if soup.title else "Untitled"
                
                # Extract main content based on common documentation layouts
                content_divs = soup.find_all(['div', 'article', 'main', 'section'], 
                                           class_=re.compile(r'(content|main|article|docs)'))
                
                if not content_divs:
                    # Fallback to body if no specific content div is found
                    content_divs = [soup.body] if soup.body else []
                
                # Extract text from the most relevant div or fallback to body
                if content_divs:
                    content = max(content_divs, key=lambda x: len(x.get_text(strip=True).split())).get_text(strip=True)
                else:
                    content = ""
                
                # Skip pages with little or no content
                if len(content.split()) < 20:
                    continue
                
                # Create document
                document = {
                    "title": title,
                    "url": current_url,
                    "content": content[:8000],  # Limit content length
                    "cdp": cdp
                }
                
                documents.append(document)
                
                # Find more links to scrape
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Normalize URL
                    if href.startswith('/'):
                        # Convert relative URL to absolute
                        parsed_base = urlparse(base_url)
                        next_url = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
                    elif href.startswith('http'):
                        next_url = href
                    else:
                        # Skip fragment links, javascript, etc.
                        continue
                    
                    # Only follow links to the same domain
                    if urlparse(next_url).netloc == base_domain:
                        if next_url not in visited_urls and next_url not in urls_to_visit:
                            urls_to_visit.append(next_url)
            
            except Exception as e:
                logger.error(f"Error scraping {current_url}: {str(e)}")
        
        logger.info(f"Scraped {len(documents)} pages for {cdp}")
        return documents
    
    def find_relevant_documents(self, query: str, cdp: Optional[str] = None) -> List[Dict]:
        """Find relevant documentation based on the query using Groq API"""
        # Identify the CDP if not provided
        if not cdp:
            for cdp_name in self.cdps.keys():
                if cdp_name.lower() in query.lower():
                    cdp = cdp_name
                    break
        
        # If still no CDP identified, search across all
        docs_to_search = []
        if cdp:
            docs_to_search = self.documentation.get(cdp, [])
        else:
            # Search a limited number from each CDP
            for cdp_docs in self.documentation.values():
                docs_to_search.extend(cdp_docs[:5])  # Top 5 docs from each CDP
        
        # Check if we have any documents to search through
        if not docs_to_search:
            # Create a synthetic document if we lack real ones
            if cdp:
                logger.warning(f"No documents found for {cdp}. Creating a synthetic document.")
                synthetic_doc = {
                    "title": f"{cdp.capitalize()} Documentation",
                    "url": self.cdps.get(cdp, ""),
                    "content": f"This is a placeholder for {cdp} documentation. The actual content could not be retrieved.",
                    "cdp": cdp,
                    "synthetic": True  # Mark as synthetic
                }
                return [synthetic_doc]
            return []
        
        # Use Groq to rank documents by relevance
        try:
            # Limit content length and number of docs to avoid token limits
            docs_limited = docs_to_search[:20]  # Limit to max 20 docs
            doc_texts = [f"Title: {doc['title']}\nURL: {doc['url']}\nContent: {doc['content'][:250]}" 
                        for doc in docs_limited]
            
            if not doc_texts:
                return []
            
            # Combine into a single prompt to avoid multiple API calls
            documents_text = "\n\n---\n\n".join(doc_texts)
            prompt = f"""You are a search engine designed to find the most relevant documentation 
            for CDP (Customer Data Platform) questions. Given the following user query and document 
            excerpts, return the indices of the 3 most relevant documents, separated by commas.
            
            Query: {query}
            
            Documents:
            {documents_text}
            
            Return only the indices (0-based) of the 3 most relevant documents, separated by commas.
            """
            
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0
            )
            
            # Parse the response to get document indices
            indices_text = response.choices[0].message.content.strip()
            try:
                indices = [int(idx.strip()) for idx in indices_text.split(',')]
                return [docs_limited[idx] for idx in indices if 0 <= idx < len(docs_limited)]
            except (ValueError, IndexError):
                logger.warning(f"Failed to parse document indices from: {indices_text}")
                # Fall back to first 3 documents
                return docs_limited[:min(3, len(docs_limited))]
                
        except Exception as e:
            logger.error(f"Error using Groq API for document retrieval: {str(e)}")
            # Fall back to first 3 documents
            return docs_to_search[:min(3, len(docs_to_search))]
    

# Initialize Flask app
app = Flask(__name__)

# Create templates directory and templates
os.makedirs('templates', exist_ok=True)

# Create the HTML template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CDP Support Agent</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            margin-top: 50px;
            margin-bottom: 50px;
            min-height: 500px;
            display: flex;
            flex-direction: column;
        }
        .chat-header {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e9ecef;
        }
        .messages-container {
            flex-grow: 1;
            overflow-y: auto;
            max-height: 500px;
            padding: 10px;
            margin-bottom: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 18px;
            max-width: 75%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            align-self: flex-end;
            margin-left: auto;
        }
        .bot-message {
            background-color: #e9ecef;
            color: #212529;
            align-self: flex-start;
            margin-right: auto;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 10px 0;
        }
        .spinner-border {
            width: 1.5rem;
            height: 1.5rem;
        }
        .loader-text {
            margin-left: 10px;
        }
        .cdp-logos {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .cdp-logo {
            max-height: 40px;
            opacity: 0.7;
            transition: opacity 0.3s;
        }
        .cdp-logo:hover {
            opacity: 1;
        }
        .status-indicator {
            display: none;
            padding: 5px;
            border-radius: 5px;
            font-size: 0.8rem;
            margin-bottom: 10px;
        }
        .source-link {
            display: block;
            font-size: 0.8rem;
            margin-top: 5px;
            color: #6c757d;
        }
        .info-box {
            background-color: #e7f3fe;
            border-left: 5px solid #2196F3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        pre {
            white-space: pre-wrap;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        code {
            font-family: Consolas, Monaco, 'Andale Mono', monospace;
            font-size: 0.9rem;
        }
        /* Style for the step-by-step instructions */
        ol {
            margin-left: 20px;
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="chat-container">
            <div class="chat-header">
                <h2>CDP Support Agent</h2>
                <p class="text-muted">Ask me questions about Segment, mParticle, Lytics, and Zeotap</p>
                <div class="cdp-logos">
                    <img src="https://seeklogo.com/images/S/segment-logo-2C88F63929-seeklogo.com.png" alt="Segment" class="cdp-logo">
                    <img src="https://mma.prnewswire.com/media/1252388/mParticle_Logo.jpg?p=twitter" alt="mParticle" class="cdp-logo">
                    <img src="https://assets-global.website-files.com/6009ec8cda7f305645c9d91b/6341e90d4c0038187e5ca657_Lytics_Logo_2022.png" alt="Lytics" class="cdp-logo">
                    <img src="https://shipyard.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fzeotap.cd752fbe.png&w=3840&q=75" alt="Zeotap" class="cdp-logo">
                </div>
            </div>
            <div class="status-indicator alert alert-success">Documentation loaded successfully</div>
            <div class="messages-container" id="messages">
                <div class="message bot-message">
                    Hello! I'm your CDP Support Agent. How can I help you with Segment, mParticle, Lytics, or Zeotap today?
                </div>
            </div>
            <div class="loading" id="loading">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <span class="loader-text">Thinking...</span>
            </div>
            <div class="input-group mb-3">
                <input type="text" id="message-input" class="form-control" placeholder="Ask about CDP platforms...">
                <button class="btn btn-primary" id="send-btn">Send</button>
            </div>
            <div class="info-box">
                <strong>Example questions:</strong>
                <ul>
                    <li id="example1">How do I set up a new source in Segment?</li>
                    <li id="example2">How can I create a user profile in mParticle?</li>
                    <li id="example3">How do I build an audience segment in Lytics?</li>
                    <li id="example4">How can I integrate my data with Zeotap?</li>
                    <li id="example5">How does Segment's audience creation compare to Lytics?</li>
                </ul>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        const messageInput = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');
        const messagesContainer = document.getElementById('messages');
        const loadingIndicator = document.getElementById('loading');
        const statusIndicator = document.querySelector('.status-indicator');
        
        // Check if documentation is loaded
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ready') {
                    statusIndicator.textContent = 'Documentation loaded successfully!';
                    statusIndicator.classList.remove('alert-danger');
                    statusIndicator.classList.add('alert-success');
                } else {
                    statusIndicator.textContent = 'Loading documentation... This might take a few minutes.';
                    statusIndicator.classList.remove('alert-success');
                    statusIndicator.classList.add('alert-danger');
                }
                statusIndicator.style.display = 'block';
                setTimeout(() => {
                    statusIndicator.style.display = 'none';
                }, 5000);
            })
            .catch(error => {
                console.error('Error:', error);
                statusIndicator.textContent = 'Error loading documentation. Please refresh the page.';
                statusIndicator.classList.remove('alert-success');
                statusIndicator.classList.add('alert-danger');
                statusIndicator.style.display = 'block';
            });
        
        // Function to add a message to the chat
        function addMessage(message, isUser, source) {
            console.log("Adding message:", { message, isUser, source });
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            
            // Add source indicator for bot messages
            if (!isUser && source) {
                const sourceIndicator = document.createElement('span');
                sourceIndicator.style.fontSize = '0.7rem';
                sourceIndicator.style.display = 'inline-block';
                sourceIndicator.style.padding = '2px 5px';
                sourceIndicator.style.borderRadius = '3px';
                sourceIndicator.style.marginLeft = '5px';
                sourceIndicator.style.color = 'white';
                
                if (source === 'direct-groq') {
                    sourceIndicator.textContent = 'AI Generated';
                    sourceIndicator.style.backgroundColor = '#fd7e14';
                } else if (source === 'doc-based') {
                    sourceIndicator.textContent = 'Doc Based';
                    sourceIndicator.style.backgroundColor = '#20c997';
                }
                
                messageDiv.appendChild(sourceIndicator);
            }
            
            // Format the message with markdown-like syntax
            let formattedMessage = message;
            
            if (!isUser) {
                // Convert markdown-like code blocks
                formattedMessage = formattedMessage.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');
                
                // Convert URLs to links
                formattedMessage = formattedMessage.replace(
                    /(https?:\/\/[^\s]+)/g, 
                    '<a href="$1" target="_blank" class="text-light">$1</a>'
                );
                
                // Look for citation links at the end
                const sourceMatch = formattedMessage.match(/Source: (https?:\/\/[^\s]+)/);
                if (sourceMatch) {
                    formattedMessage = formattedMessage.replace(
                        /Source: (https?:\/\/[^\s]+)/, 
                        '<span class="source-link">Source: <a href="$1" target="_blank">$1</a></span>'
                    );
                }
                
                // Format step lists
                formattedMessage = formattedMessage.replace(
                    /(\d+\.\s+.*?)(?=\d+\.|$)/gs, 
                    '<li>$1</li>'
                );
                if (formattedMessage.includes('<li>')) {
                    formattedMessage = '<ol>' + formattedMessage + '</ol>';
                }
                
                // Format bold text
                formattedMessage = formattedMessage.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            }
            
            try {
                const contentDiv = document.createElement('div');
                contentDiv.innerHTML = formattedMessage;
                messageDiv.appendChild(contentDiv);
                
                messagesContainer.appendChild(messageDiv);
                
                // Scroll to the bottom
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            } catch (error) {
                console.error("Error rendering message:", error);
            }
        }
        
        // Function to send a message
        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, true);
            
            // Clear input
            messageInput.value = '';
            
            // Show loading indicator
            loadingIndicator.style.display = 'block';
            
            // Send message to server
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: message }),
            })
            .then(response => {
                console.log("Response status:", response.status);
                return response.json();
            })
            .then(data => {
                // Hide loading indicator
                loadingIndicator.style.display = 'none';
                
                console.log("Response data:", data);
                
                // Add bot response to chat
                if (data && typeof data.answer === 'object' && data.answer.text) {
                    console.log("Using structured response with source:", data.answer.source);
                    addMessage(data.answer.text, false, data.answer.source);
                } else if (data && data.answer) {
                    console.log("Using simple string response");
                    addMessage(data.answer, false);
                } else {
                    console.error("Invalid response format:", data);
                    addMessage('Sorry, received an invalid response format from the server.', false);
                }
            })
            .catch(error => {
                console.error("Fetch error:", error);
                loadingIndicator.style.display = 'none';
                addMessage('Sorry, there was an error processing your request. Please try again.', false);
            });
        }
        
        // Event listeners
        sendBtn.addEventListener('click', sendMessage);
        
        messageInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Example question click handlers
        document.querySelectorAll('#example1, #example2, #example3, #example4, #example5').forEach(el => {
            el.style.cursor = 'pointer';
            el.style.color = '#007bff';
            el.addEventListener('click', function() {
                messageInput.value = this.textContent;
                sendMessage();
            });
        });
    });
</script>
</body>
</html>
''')

# Initialize the CDP Support Agent
agent = None

@app.before_request
def initialize_agent():
    global agent
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY environment variable not set")
            # Will be initialized on demand
        else:
            agent = CDPSupportAgent(api_key=api_key)
            logger.info("CDP Support Agent initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing CDP Support Agent: {str(e)}")

def get_agent():
    global agent
    if agent is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        agent = CDPSupportAgent(api_key=api_key)
        logger.info("CDP Support Agent initialized on demand")
    return agent

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    try:
        support_agent = get_agent()
        return jsonify({"status": "ready"})
    except Exception as e:
        return jsonify({"status": "loading", "error": str(e)})

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({"answer": {"text": "Please provide a question.", "source": "error"}})
        
        support_agent = get_agent()
        answer = support_agent.answer_question(question)
        
        # Log the response source and structure
        logger.info(f"Response data: {json.dumps(answer)}")
        
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return jsonify({"answer": {"text": f"Sorry, an error occurred: {str(e)}", "source": "error"}})

if __name__ == '__main__':
    # Check if the API key is set
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY environment variable not set.")
        print("Please set it with: export GROQ_API_KEY='your-api-key'")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
