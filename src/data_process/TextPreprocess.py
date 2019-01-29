import re
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize

class TextPreprocess:
    raw_text = ""

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

    def preprocess_and_tokenize_tweet(self):
        """
        Preprocess tweet, remove url, emoji, mentions, hastags, stopwords
        :return: tweet
        """
        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
        cleaned_tweet = p.clean(self.raw_text)
        cleaned_tokens = self.remove_stopwords_and_tokenize(cleaned_tweet)  # remove stopwords
        return cleaned_tokens
