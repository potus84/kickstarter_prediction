import re
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize
from pycorenlp import StanfordCoreNLP


class TextPreprocess:
    raw_text = ''

    def __init__(self, raw_text):
        self.raw_text = raw_text

    def remove_stopwords_and_tokenize(self, text):
        """
        Remove stopwords and tokenize the tweet
        :param tweet: raw tweet
        :return: the tokens of tweet with no stop word
        """
        stop_words = set(stopwords.words('english'))
        lower_text = text.lower()
        # Tokenize the raw text (either use the Tweet Tokenizer or normal tokenizer)

        # Temporarily disabled now)
        # tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        # tweet_tokens = tweet_tokenizer.tokenize(raw_text)


        # At first, try NLTK tokenizer
        tweet_tokens = word_tokenize(lower_text)
        # Create a set of none stop words
        no_stop_words = []
        for token in tweet_tokens:
            if token not in stop_words:
                no_stop_words.append(token)

        # Remove words have less than 2 letters (usually have no meaning)
        no_stop_words = [re.sub(r'^\w\w?$', '', i) for i in no_stop_words]

        return no_stop_words

    def preprocess_text(self):
        """
        Preprocess tweet, remove url, emoji, mentions, hastags, stopwords
        :return: tweet
        """
        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
        cleaned_text = p.clean(self.raw_text)
        return cleaned_text

    def get_sentiment_value(self):
        """
        Return the sentiment value of the tweet
        :param tweet: a non-preprocess tweet
        :return: integer sentiment value in range [0-4]
        """
        nlp = StanfordCoreNLP('http://localhost:9000')
        process_text = self.preprocess_text()
        res = nlp.annotate(process_text, properties={'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 10000,})
        sentiment = int(res["sentences"][0]["sentimentValue"])
        return sentiment

    def num_occurrences(self, pattern):
        """
        Extract number of occurences of a pattern in text
        :param text: a string represet text
        :param pattern: a regex pattern
        :return: Integer indicates number of occurrences
        """
        return len(re.findall(pattern, self.raw_text))
