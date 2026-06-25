import { useState } from 'react';
import axios from 'axios';

/**
 * TextAnalyzer.jsx — Single Text Sentiment Analysis
 * 
 * The core feature: users enter text, select a model, and get
 * sentiment prediction with confidence score and emoji.
 */

const EXAMPLE_TEXTS = [
  "This movie was absolutely fantastic and thrilling!",
  "Terrible waste of time, completely boring and predictable.",
  "An okay film, nothing special but not bad either.",
];

const TextAnalyzer = () => {
  const [text, setText] = useState('');
  const [model, setModel] = useState('lstm');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('http://localhost:5000/analyze', {
        text,
        model,
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const emoji = result?.sentiment === 'positive' ? '😊' : '😞';
  const colorClass = result?.sentiment === 'positive'
    ? 'text-green-400 bg-green-500/10 border-green-500/30'
    : 'text-red-400 bg-red-500/10 border-red-500/30';

  return (
    <div className="space-y-6">
      {/* Input */}
      <div className="bg-[#10121a] border border-white/10 rounded-2xl p-6">
        <label className="text-sm text-gray-400 block mb-2">
          Enter text to analyze
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste text here..."
          rows={5}
          maxLength={5000}
          className="w-full bg-[#0a0b10] border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
        />
        <div className="flex justify-between items-center mt-3">
          <div className="flex gap-2">
            {EXAMPLE_TEXTS.map((t, i) => (
              <button
                key={i}
                onClick={() => setText(t)}
                className="text-xs px-3 py-1 bg-white/5 hover:bg-white/10 rounded-lg text-gray-400 transition-colors"
              >
                Example {i + 1}
              </button>
            ))}
          </div>
          <span className="text-xs text-gray-500">{text.length}/5000</span>
        </div>

        {/* Model Selection */}
        <div className="mt-4 flex items-center gap-4">
          <label className="text-sm text-gray-400">Model:</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="bg-[#0a0b10] border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            <option value="lstm">LSTM Neural Network (Deep Learning)</option>
            <option value="tfidf">TF-IDF + Logistic Regression (Traditional)</option>
          </select>
        </div>

        <button
          onClick={analyze}
          disabled={loading || !text.trim()}
          className="mt-4 w-full bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white font-bold py-3 rounded-xl transition-all disabled:opacity-50"
        >
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className={`${colorClass} border rounded-2xl p-6 text-center`}>
          <p className="text-6xl mb-2">{emoji}</p>
          <p className="text-2xl font-bold capitalize">{result.sentiment}</p>
          <p className="text-lg mt-1">
            Confidence: {result.confidence_percentage}
          </p>
          <p className="text-sm text-gray-400 mt-2">
            Model: {result.model_used}
          </p>
        </div>
      )}
    </div>
  );
};

export default TextAnalyzer;