import React, { useState } from 'react';
import { Search, Globe, AlertCircle, CheckCircle, Loader } from 'lucide-react';

export default function WebsiteContentSearch() {
  const [url, setUrl] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [expandedIndex, setExpandedIndex] = useState(null);
console.log(results)
  const API_BASE_URL = 'http://localhost:8000';

  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!url || !searchQuery) {
      setError('Please provide both URL and search query');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');
    setResults([]);

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url,
          query: searchQuery
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Search failed');
      }

      setResults(data.results);
      setSuccess(`Found ${data.results.length} matches`);
    } catch (err) {
      setError(err.message || 'An error occurred during search');
    } finally {
      setLoading(false);
    }
  };
const toggleExpand = (index) => {
  setExpandedIndex(expandedIndex === index ? null : index);
};

  const getMatchColor = (score) => {
    if (score >= 90) return 'success';
    if (score >= 70) return 'info';
    if (score >= 50) return 'warning';
    return 'secondary';
  };

  return (
    <div className="min-vh-100" style={{ backgroundColor: '#f8f9fa' }}>
      <div className="container py-5">
        <div className="text-center mb-5">
          <h1 className="display-4 fw-bold mb-3">Likhith's website content  Search tool </h1>
          <p className="lead text-muted">Assignment Submission</p>
        </div>

        <div className="card shadow-sm mb-4">
          <div className="card-body p-4">
            <form onSubmit={handleSearch}>
              <div className="mb-3">
                <label className="form-label d-flex align-items-center">
                  <Globe size={20} className="me-2" />
                  Website URL
                </label>
                <input
                  type="url"
                  className="form-control form-control-lg"
                  placeholder="https://example.com"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  disabled={loading}
                />
              </div>

              <div className="mb-4">
                <label className="form-label d-flex align-items-center">
                  <Search size={20} className="me-2" />
                  Search Query
                </label>
                <input
                  type="text"
                  className="form-control form-control-lg"
                  placeholder="Enter your search query"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  disabled={loading}
                />
              </div>

              <button 
                type="submit" 
                className="btn btn-primary btn-lg w-100"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader className="me-2" size={20} style={{ animation: 'spin 1s linear infinite' }} />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="me-2" size={20} />
                    Search
                  </>
                )}
              </button>
            </form>

            {error && (
              <div className="alert alert-danger d-flex align-items-center mt-3 mb-0" role="alert">
                <AlertCircle className="me-2" size={20} />
                {error}
              </div>
            )}

            {success && (
              <div className="alert alert-success d-flex align-items-center mt-3 mb-0" role="alert">
                <CheckCircle className="me-2" size={20} />
                {success}
              </div>
            )}
          </div>
        </div>

        {results.length > 0 && (
          
          <div>
            <h3 className="mb-4">Search Results</h3>
            {results.map((result, index) => (
              <div key={index} className="card shadow-sm mb-3">
                <div className="card-body">
                  <div className="d-flex justify-content-between align-items-start mb-3">
                    <div className="flex-grow-1">
                      <h5 className="card-title mb-2">{result.title}</h5>
                      <p className="text-muted small mb-0">Path: {result.path}</p>
                    </div>
                    <span className={`badge bg-${getMatchColor(result.similarity_score)} ms-3`}>
                      {result.similarity_score}% match
                    </span>
                  </div>

                  <p className="card-text">{result.content}</p>

                  <button
                    className="btn btn-sm btn-outline-primary mt-2"
                    onClick={() => toggleExpand(index)}
                  >
                    {expandedIndex === index ? '▲ Hide HTML' : '▼ View HTML'}
                  </button>

                  {expandedIndex === index && (
                    <div className="mt-3">
                      <pre className="bg-light p-3 rounded" style={{ 
                        fontSize: '0.85rem', 
                        maxHeight: '300px', 
                        overflow: 'auto',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word'
                      }}>
                        <code>{result.html_chunk}</code>
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        .card {
          border: none;
          border-radius: 12px;
        }
        
        .form-control:focus {
          border-color: #0d6efd;
          box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
        }
        
        .btn-primary {
          border-radius: 8px;
        }
        
        pre {
          margin-bottom: 0;
        }
      `}</style>
    </div>
  );
}