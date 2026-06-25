const Navbar = () => (
  <nav className="border-b border-white/10 bg-[#10121a]/80 backdrop-blur-md">
    <div className="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
      <h1 className="text-xl font-bold">
        Sentiment <span className="text-blue-400">Analyzer</span>
      </h1>
      <a
        href="https://github.com/swap821/sentiment-analysis-app"
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm text-gray-400 hover:text-white transition-colors"
      >
        GitHub
      </a>
    </div>
  </nav>
);

export default Navbar;