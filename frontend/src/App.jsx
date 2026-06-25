import { useState } from 'react';
import TextAnalyzer from './components/TextAnalyzer';
import BatchAnalyzer from './components/BatchAnalyzer';
import ModelComparison from './components/ModelComparison';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

function App() {
  const [activeTab, setActiveTab] = useState('single');

  return (
    <div className="min-h-screen bg-[#0a0b10] text-white">
      <Navbar />

      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Hero */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
            AI{' '}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
              Sentiment Analyzer
            </span>
          </h1>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Compare traditional ML (TF-IDF + Logistic Regression) vs Deep Learning (LSTM Neural Network)
            for text sentiment analysis.
          </p>
        </div>

        {/* Tabs */}
        <div className="flex justify-center gap-4 mb-8">
          {[
            { key: 'single', label: 'Single Text' },
            { key: 'batch', label: 'Batch CSV' },
            { key: 'models', label: 'Model Comparison' },
          ].map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                activeTab === tab.key
                  ? 'bg-blue-600 text-white'
                  : 'bg-[#10121a] text-gray-400 hover:text-white border border-white/10'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        {activeTab === 'single' && <TextAnalyzer />}
        {activeTab === 'batch' && <BatchAnalyzer />}
        {activeTab === 'models' && <ModelComparison />}
      </main>

      <Footer />
    </div>
  );
}

export default App;