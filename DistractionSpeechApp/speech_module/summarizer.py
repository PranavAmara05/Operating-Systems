import re
import logging
import nltk
from nltk.corpus import stopwords
from collections import Counter
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('summarizer')

class TextSummarizer:
    """
    Summarizes spoken text using NLP techniques
    """
    
    def __init__(self, language: str = 'english'):
        """Initialize the text summarizer
        
        Args:
            language: Language for stopwords
        """
        self.language = language
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words(language))
        self.collected_text = ""
        self.sentence_count = 0
        
        logger.info(f"TextSummarizer initialized for {language}")
    
    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are downloaded"""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords")
            nltk.download('stopwords', quiet=True)
    
    def add_text(self, text: str):
        """Add text to the collection for summarization
        
        Args:
            text: Text to add
        """
        if not text:
            return
            
        self.collected_text += " " + text
        self.sentence_count += len(re.findall(r'[.!?]', text))
        
        logger.debug(f"Added text. Total sentences: {self.sentence_count}")
    
    def clear_text(self):
        """Clear the collected text"""
        self.collected_text = ""
        self.sentence_count = 0
    
    def extract_keywords(self, text: Optional[str] = None, top_n: int = 5) -> List[str]:
        """Extract top keywords from text
        
        Args:
            text: Text to analyze (if None, uses collected text)
            top_n: Number of top keywords to return
        
        Returns:
            List[str]: Top keywords
        """
        text_to_analyze = text if text is not None else self.collected_text
        
        if not text_to_analyze:
            return []
        
        # Extract words, remove punctuation, convert to lowercase
        words = re.findall(r'\b\w+\b', text_to_analyze.lower())
        
        # Filter out stopwords
        keywords = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        # Count frequencies
        freq = Counter(keywords)
        
        # Get top keywords
        top_keywords = [word for word, _ in freq.most_common(top_n)]
        
        logger.debug(f"Extracted keywords: {', '.join(top_keywords)}")
        return top_keywords
    
    def summarize(self, text: Optional[str] = None, max_sentences: int = 3) -> str:
        """Summarize text using extractive summarization
        
        Args:
            text: Text to summarize (if None, uses collected text)
            max_sentences: Maximum number of sentences in summary
        
        Returns:
            str: Summarized text
        """
        text_to_summarize = text if text is not None else self.collected_text
        
        if not text_to_summarize:
            return ""
        
        # Extract keywords
        top_keywords = self.extract_keywords(text_to_summarize)
        
        if not top_keywords:
            # If no keywords found, return first sentence
            sentences = re.split(r'(?<=[.!?])\s+', text_to_summarize)
            return sentences[0] if sentences else ""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text_to_summarize)
        
        # Score sentences based on keyword presence
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for keyword in top_keywords if keyword.lower() in sentence.lower())
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sentence for sentence, _ in scored_sentences[:max_sentences]]
        
        # Re-order sentences based on original order
        original_order = []
        for sentence in sentences:
            if sentence in [s for s, _ in scored_sentences[:max_sentences]]:
                original_order.append(sentence)
            if len(original_order) >= max_sentences:
                break
        
        summary = ' '.join(original_order)
        
        # Ensure summary ends with punctuation
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        logger.info(f"Generated summary: {len(summary)} chars")
        return summary
    
    def should_summarize(self, min_sentences: int = 3) -> bool:
        """Check if there's enough text to summarize
        
        Args:
            min_sentences: Minimum number of sentences needed
        
        Returns:
            bool: True if there's enough text to summarize
        """
        return self.sentence_count >= min_sentences