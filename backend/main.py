from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import re
from typing import List, Dict
import numpy as np
from urllib.parse import urlparse
import hashlib
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Website Content Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('all-MiniLM-L6-v2')

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "url_data"
VECTOR_DIM = 384  
CHUNK_SIZE = 200  
MIN_CHUNK_LENGTH = 50  

CONTENT_TAGS = [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',  
    'article', 'section', 'main',  
    'p', 'div', 'li', 'blockquote', 'pre',  
    'td', 'th', 'span', 'a', 'strong', 'em'  
]

class SearchRequest(BaseModel):
    url: HttpUrl
    query: str

class SearchResult(BaseModel):
    title: str
    path: str
    content: str
    html_chunk: str
    similarity_score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

def init_milvus():
    """Initialize Milvus connection and collection"""
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="html_chunk", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),  
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        ]

        schema = CollectionSchema(fields=fields, description="Website content chunks")
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        return collection
    except Exception as e:
        print(f"Error initializing Milvus: {e}")
        raise

def extract_meaningful_content(soup: BeautifulSoup) -> List[Dict]:
    """
    Extract only meaningful content-bearing elements with their HTML structure
    Returns list of content blocks with text and HTML
    """
    content_blocks = []

    for element in soup(['script', 'style', 'head', 'link', 'meta', 'noscript', 
                        'iframe', 'nav', 'footer', 'header', 'aside', 'button', 
                        'form', 'input', 'select', 'textarea', 'svg', 'path']):
        element.decompose()

    for element in soup.find_all():
        if len(element.get_text(strip=True)) == 0:
            element.decompose()

    for tag in soup.find_all(CONTENT_TAGS):

        text = tag.get_text(separator=' ', strip=True)

        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) >= 25 and not is_boilerplate_text(text):

            html_str = str(tag)

            cleaned_html = clean_html_attributes(tag)

            content_blocks.append({
                'text': text,
                'html': cleaned_html,
                'tag': tag.name,
                'text_length': len(text)
            })

    content_blocks.sort(key=lambda x: (
        0 if x['tag'] in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] else 1,
        -x['text_length']  
    ))

    return content_blocks

def is_boilerplate_text(text: str) -> bool:
    """Check if text is boilerplate/navigation content"""
    boilerplate_indicators = [
        'cookie', 'privacy', 'terms', 'conditions', 'policy',
        'login', 'sign up', 'register', 'subscribe', 'menu',
        'navigation', 'sidebar', 'footer', 'header', 'copyright'
    ]

    text_lower = text.lower()
    return any(indicator in text_lower for indicator in boilerplate_indicators)

def clean_html_attributes(tag: Tag) -> str:
    """
    Clean HTML by removing unnecessary attributes but keeping structure
    Keep only essential attributes like href for links
    """
    if isinstance(tag, NavigableString):
        return str(tag)

    tag_copy = BeautifulSoup(str(tag), 'html.parser').find()
    if not tag_copy:
        return str(tag)

    keep_attrs = ['href', 'src', 'alt', 'title']

    def clean_tag(element):
        if isinstance(element, NavigableString):
            return

        attrs_to_remove = [attr for attr in element.attrs if attr not in keep_attrs]
        for attr in attrs_to_remove:
            del element[attr]

        for child in element.children:
            if isinstance(child, Tag):
                clean_tag(child)

    clean_tag(tag_copy)
    return str(tag_copy)

def scrape_website(url: str) -> Dict:
    """Scrape website and extract clean, meaningful content"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string if soup.title else urlparse(url).netloc
        title = title.strip() if title else "Untitled"

        parsed_url = urlparse(url)
        path = parsed_url.path if parsed_url.path else "/home"

        content_blocks = extract_meaningful_content(soup)

        if not content_blocks:
            raise HTTPException(status_code=400, detail="No meaningful content found on website")

        print(f"Extracted {len(content_blocks)} content blocks")
        return {
            'url': url,
            'title': title,
            'path': path,
            'content_blocks': content_blocks
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to scrape website: {str(e)}")

def chunk_content_blocks(content_blocks: List[Dict], chunk_size: int = CHUNK_SIZE) -> List[Dict]:
    """
    Intelligently chunk content blocks while preserving HTML structure
    and avoiding duplicates
    """
    chunks = []
    seen_content = set()

    for block in content_blocks:
        block_text = block['text']
        block_html = block['html']
        block_tag = block['tag']

        content_hash = hashlib.md5(block_text.encode()).hexdigest()
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)

        if block_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if len(block_text) >= MIN_CHUNK_LENGTH:
                chunks.append({
                    'content': block_text,
                    'html_chunk': block_html,
                    'tags': [block_tag],
                    'content_hash': content_hash
                })
            continue

        if len(block_text) > chunk_size:

            sentences = re.split(r'[.!?]+', block_text)
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                    if len(current_chunk) >= MIN_CHUNK_LENGTH:
                        chunk_hash = hashlib.md5(current_chunk.encode()).hexdigest()
                        if chunk_hash not in seen_content:
                            chunks.append({
                                'content': current_chunk.strip(),
                                'html_chunk': f"<{block_tag}>{current_chunk.strip()}</{block_tag}>",
                                'tags': [block_tag],
                                'content_hash': chunk_hash
                            })
                            seen_content.add(chunk_hash)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence

            if current_chunk and len(current_chunk) >= MIN_CHUNK_LENGTH:
                chunk_hash = hashlib.md5(current_chunk.encode()).hexdigest()
                if chunk_hash not in seen_content:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'html_chunk': f"<{block_tag}>{current_chunk.strip()}</{block_tag}>",
                        'tags': [block_tag],
                        'content_hash': chunk_hash
                    })
                    seen_content.add(chunk_hash)
        else:

            if len(block_text) >= MIN_CHUNK_LENGTH:
                chunks.append({
                    'content': block_text,
                    'html_chunk': block_html,
                    'tags': [block_tag],
                    'content_hash': content_hash
                })

    print(f"Created {len(chunks)} unique chunks after deduplication")
    return chunks

def remove_similar_chunks(chunks: List[Dict], similarity_threshold: float = 0.9) -> List[Dict]:
    """Remove chunks that are too similar using embedding similarity"""
    if len(chunks) <= 1:
        return chunks

    texts = [chunk['content'] for chunk in chunks]
    embeddings = model.encode(texts)

    similarity_matrix = cosine_similarity(embeddings)

    unique_chunks = []
    used_indices = set()

    for i in range(len(chunks)):
        if i in used_indices:
            continue

        unique_chunks.append(chunks[i])
        used_indices.add(i)

        for j in range(i + 1, len(chunks)):
            if j not in used_indices and similarity_matrix[i][j] > similarity_threshold:
                used_indices.add(j)
                print(f"Removed similar chunk: {chunks[j]['content'][:100]}...")

    print(f"Reduced from {len(chunks)} to {len(unique_chunks)} chunks after similarity filtering")
    return unique_chunks

def store_in_milvus(collection: Collection, website_data: Dict, chunks: List[Dict]):
    """Store chunks in Milvus with embeddings"""
    try:
        if not chunks:
            raise ValueError("No chunks to store")

        unique_chunks = remove_similar_chunks(chunks)

        texts = [chunk['content'] for chunk in unique_chunks]
        embeddings = model.encode(texts)

        urls = [website_data['url']] * len(unique_chunks)
        titles = [website_data['title']] * len(unique_chunks)
        paths = [website_data['path']] * len(unique_chunks)
        contents = [chunk['content'][:4999] for chunk in unique_chunks]
        html_chunks = [chunk['html_chunk'][:4999] for chunk in unique_chunks]
        content_hashes = [chunk['content_hash'] for chunk in unique_chunks]

        data = [
            urls,
            titles,
            paths,
            contents,
            html_chunks,
            content_hashes,
            embeddings.tolist()
        ]

        collection.insert(data)
        collection.flush()

        return len(unique_chunks)
    except Exception as e:
        print(f"Error storing in Milvus: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store in Milvus: {str(e)}")

def search_similar(collection: Collection, query: str, top_k: int = 10) -> List[Dict]:
    """Search for similar content in Milvus"""
    try:

        query_embedding = model.encode([query])[0].tolist()

        collection.load()

        search_params = {"metric_type": "L2", "params": {"nprobe": 15}}

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k * 2,  
            output_fields=["url", "title", "path", "content", "html_chunk", "content_hash"]
        )

        search_results = []
        seen_hashes = set()

        for hits in results:
            for hit in hits:
                content_hash = hit.entity.get("content_hash", "")

                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                distance = hit.distance
                similarity_score = max(0, min(100, 100 * (1 - distance / 2)))  
                similarity_score = round(similarity_score, 1)

                if similarity_score >= 10.0:  
                    search_results.append({
                        "title": hit.entity.get("title", ""),
                        "path": hit.entity.get("path", ""),
                        "content": hit.entity.get("content", ""),
                        "html_chunk": hit.entity.get("html_chunk", ""),
                        "similarity_score": similarity_score
                    })

                if len(search_results) >= top_k:
                    break

        search_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return search_results[:top_k]
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize Milvus on startup"""
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        print("Milvus connection established successfully")
    except Exception as e:
        print(f"Warning: Could not connect to Milvus: {e}")

@app.get("/")
def read_root():
    return {
        "message": "Website Content Search API", 
        "status": "running",
        "version": "2.1",
        "features": ["semantic_search", "html_preservation", "intelligent_chunking", "deduplication"]
    }

@app.post("/search", response_model=SearchResponse)
async def search_website(request: SearchRequest):
    """
    Search website content based on URL and query
    """
    try:

        collection = init_milvus()

        print(f"Scraping website: {request.url}")

        website_data = scrape_website(str(request.url))

        print(f"Found {len(website_data['content_blocks'])} content blocks")

        chunks = chunk_content_blocks(website_data['content_blocks'], CHUNK_SIZE)

        print(f"Created {len(chunks)} chunks")
        if not chunks:
            raise HTTPException(status_code=400, detail="No meaningful content chunks created")

        num_stored = store_in_milvus(collection, website_data, chunks)
        print(f"Stored {num_stored} unique chunks in Milvus")

        results = search_similar(collection, request.query, top_k=10)
        print(f"Returning {len(results)} unique results")

        return SearchResponse(results=results)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        return {"status": "healthy", "milvus": "connected", "model": "all-MiniLM-L6-v2"}
    except Exception as e:
        return {"status": "unhealthy", "milvus": "disconnected", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)