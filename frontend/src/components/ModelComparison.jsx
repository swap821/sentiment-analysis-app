/**
 * ModelComparison.jsx — Educational Model Comparison
 * 
 * Displays side-by-side comparison of TF-IDF + Logistic Regression vs LSTM,
 * explaining how each approach works.
 */

const ModelComparison = () => {
  const models = [
    {
      name: 'TF-IDF + Logistic Regression',
      type: 'Traditional ML',
      color: 'blue',
      pros: [
        'Fast training and prediction',
        'Highly interpretable',
        'Works well with limited data',
        'Low memory footprint',
      ],
      cons: [
        'Ignores word order and context',
        'Cannot capture semantic meaning',
        'Struggles with sarcasm and nuance',
      ],
      howItWorks: 'Converts text to numerical vectors using word frequency, then uses a linear classifier to separate positive and negative reviews.',
    },
    {
      name: 'LSTM Neural Network',
      type: 'Deep Learning',
      color: 'purple',
      pros: [
        'Understands word order and context',
        'Captures semantic relationships',
        'Better with sarcasm and nuance',
        'Can learn complex patterns',
      ],
      cons: [
        'Slower to train',
        'Requires more data',
        'Less interpretable (black box)',
        'Higher memory usage',
      ],
      howItWorks: 'Processes words as a sequence using memory cells that remember important information from earlier in the text.',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {models.map((m) => (
          <div
            key={m.name}
            className="bg-[#10121a] border border-white/10 rounded-2xl p-6"
          >
            <div className={`inline-block px-3 py-1 rounded-full text-xs font-medium mb-4 ${
              m.color === 'blue'
                ? 'bg-blue-500/20 text-blue-400'
                : 'bg-purple-500/20 text-purple-400'
            }`}>
              {m.type}
            </div>

            <h3 className="text-xl font-bold mb-3">{m.name}</h3>
            <p className="text-gray-400 text-sm mb-6">{m.howItWorks}</p>

            <div className="space-y-4">
              <div>
                <p className="text-sm font-medium text-green-400 mb-2">Pros</p>
                <ul className="space-y-1">
                  {m.pros.map((p, i) => (
                    <li key={i} className="text-sm text-gray-400 flex items-start gap-2">
                      <span className="text-green-500 mt-0.5">✓</span>
                      {p}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <p className="text-sm font-medium text-red-400 mb-2">Cons</p>
                <ul className="space-y-1">
                  {m.cons.map((c, i) => (
                    <li key={i} className="text-sm text-gray-400 flex items-start gap-2">
                      <span className="text-red-500 mt-0.5">✗</span>
                      {c}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Key Insight */}
      <div className="bg-blue-500/5 border border-blue-500/20 rounded-2xl p-6">
        <h4 className="text-lg font-bold text-blue-400 mb-2">Key Insight</h4>
        <p className="text-gray-300 text-sm leading-relaxed">
          TF-IDF + Logistic Regression treats each word independently (like a bag of words),
          while LSTM processes words in sequence, understanding that "not good" is negative
          even though "good" is a positive word. This is why LSTM typically performs better
          on sentiment analysis tasks — context matters.
        </p>
      </div>
    </div>
  );
};

export default ModelComparison;