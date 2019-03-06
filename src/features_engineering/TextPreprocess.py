import re
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize
from pycorenlp import StanfordCoreNLP
from textblob import TextBlob


class TextPreprocess:
    raw_text = ''

    def __init__(self, raw_text):
        if isinstance(raw_text, str):
            self.raw_text = raw_text
        else:
            print('Failed: ', raw_text)

    def remove_stopwords_and_tokenize(self, text):

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
        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
        raw_text = self.raw_text
        cleaned_text = p.clean(raw_text)
        return cleaned_text

    # def get_sentiment_value(self):
    #     """
    #     Return the sentiment value of the tweet
    #     :param tweet: a non-preprocess tweet
    #     :return: integer sentiment value in range [0-4]
    #     """
    #     nlp = StanfordCoreNLP('http://localhost:9000')
    #     process_text = self.preprocess_text()
    #     res = nlp.annotate(process_text, properties={'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 10000,})
    #     sentiment = int(res["sentences"][0]["sentimentValue"])
    #     return sentiment

    def get_subjectivity_value(self):
        txt_blob = TextBlob(self.raw_text)
        subjectivity = txt_blob.sentiment.subjectivity
        return subjectivity

    def get_polarity_value(self):
        txt_blob = TextBlob(self.raw_text)
        polarity = txt_blob.sentiment.polarity
        return polarity

    def get_sentiment_value(self):
        txt_blob = TextBlob(self.raw_text)
        subjectivity = txt_blob.sentiment.subjectivity
        polarity = txt_blob.sentiment.polarity
        return [subjectivity, polarity]

    def num_occurrences(self, pattern):

        return len(re.findall(pattern, self.raw_text))

    def check_existence_of_words(self, wordlist):
        raw_text = self.preprocess_text()
        raw_text = raw_text.lower()
        found_word = 0
        for word in wordlist:
            if raw_text.find(word) != -1:
                found_word = 1
                break

        return found_word

    def contain_bad_words(self, google_bad_words_list, noswearing_bad_words_list):
        return max(self.check_existence_of_words(google_bad_words_list),
                   self.check_existence_of_words(noswearing_bad_words_list))
