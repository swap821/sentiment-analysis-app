import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

/**
 * BatchAnalyzer.jsx — CSV Upload for Batch Sentiment Analysis
 * 
 * Users can drag-and-drop a CSV file to analyze multiple texts at once.
 */

const BatchAnalyzer = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file || !file.name.endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/analyze/batch', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Batch analysis failed');
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
  });

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`bg-[#10121a] border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
          isDragActive
            ? 'border-blue-500 bg-blue-500/5'
            : 'border-white/20 hover:border-white/40'
        }`}
      >
        <input {...getInputProps()} />
        <p className="text-4xl mb-4">📄</p>
        <p className="text-lg font-medium">
          {isDragActive ? 'Drop CSV here' : 'Drag & drop a CSV file, or click to select'}
        </p>
        <p className="text-sm text-gray-500 mt-2">
          CSV must have a 'text', 'review', 'content', or 'message' column
        </p>
      </div>

      {/* Loading */}
      {loading && (
        <div className="text-center py-8">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p className="text-gray-400">Analyzing texts...</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400">
          {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="bg-[#10121a] border border-white/10 rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-bold">Batch Results</h3>
            <div className="flex gap-4 text-sm">
              <span className="text-green-400">✅ {results.positive} Positive</span>
              <span className="text-red-400">❌ {results.negative} Negative</span>
              <span className="text-gray-400">Total: {results.total}</span>
            </div>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {results.results.map((r, i) => (
              <div
                key={i}
                className={`flex justify-between items-center p-3 rounded-lg border ${
                  r.sentiment === 'positive'
                    ? 'bg-green-500/5 border-green-500/20'
                    : 'bg-red-500/5 border-red-500/20'
                }`}
              >
                <p className="text-sm text-gray-300 flex-1 truncate mr-4">{r.text}</p>
                <div className="flex items-center gap-3">
                  <span className={`text-sm font-medium capitalize ${
                    r.sentiment === 'positive' ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {r.sentiment}
                  </span>
                  <span className="text-xs text-gray-500">
                    {(r.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchAnalyzer;