"""
nltk_setup.py — Download Required NLTK Data

NLTK (Natural Language Toolkit) requires downloading additional data files
before it can be used. This script downloads all required data.

Run this BEFORE training the models:
    python nltk_setup.py
"""

import nltk


def download_nltk_data():
    """
    Download all NLTK data packages needed for the project.
    
    Required packages:
    - punkt: Tokenizer for splitting text into words/sentences
    - stopwords: Common words to remove (the, is, at, etc.)
    - wordnet: Lexical database for word relationships
    - averaged_perceptron_tagger: Part-of-speech tagger
    """
    print("Downloading NLTK data packages...")
    print("(This is a one-time setup)")
    
    packages = [
        'punkt',           # Tokenization
        'stopwords',       # Common words removal
        'wordnet',         # Word relationships
        'averaged_perceptron_tagger'  # POS tagging
    ]
    
    for package in packages:
        print(f"  Downloading {package}...")
        nltk.download(package, quiet=True)
    
    print("\nAll NLTK data downloaded successfully!")
    print("You can now run: python train_models.py")


if __name__ == "__main__":
    download_nltk_data()