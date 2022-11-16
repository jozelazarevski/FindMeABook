import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import pandas as pd
import textblob as tb
from textblob import TextBlob

book_file='books/Don-Quijote.epub'
book = epub.read_epub('books/An-Elementary-Spanish-Reader.epub')

for doc in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
    print (doc)
    

blacklist = [
	'[document]',
	'noscript',
	'header',
	'html',
	'meta',
	'head', 
	'input',
	'script',
	# there may be more elements you don't want, such as "style", etc.
]

def epub2thtml(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapters.append(item.get_content())
    return chapters

def chap2text(chap):
    output = ''
    soup = BeautifulSoup(chap, 'html.parser')
    text = soup.find_all(text=True)
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output

def thtml2ttext(thtml):
    Output = []
    for html in thtml:
        text =  chap2text(html)
        Output.append(text)
    return Output

def epub2text(epub_path):
    chapters = epub2thtml(epub_path)
    ttext = thtml2ttext(chapters)
    return ttext


out=epub2text(book_file)

concatenated_lists=[]
concatenated_text=''

for text in out:
    concatenated_text = concatenated_text+text

# to remove '\xa0'
    
    
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
stop_words=set(stopwords.words("spanish"))

  
from collections import Counter
from nltk import ngrams
import re,string

#This removes words of up to 3 characters entirely:
cleaned_text = re.sub(r'\b\w{,0}\b', '',concatenated_text)
cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))
tokenized_word=word_tokenize(cleaned_text)

ngram_counts = Counter(ngrams(tokenized_word, 5))
ngram_counts.most_common(10)

ngram_counts_df = pd.DataFrame.from_dict(ngram_counts, orient='index').reset_index()
    


tokenized_sentence =sent_tokenize(concatenated_text)
ngram_counts = Counter(ngrams(tokenized_sentence, 1))
ngram_counts.most_common(100)





#Removing Stopwords
filtered_sent=[]
for w in tokenized_sent:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_sent)
print("Filterd Sentence:",filtered_sent)

 

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,ngram_range = (1,1), # to use bigrams ngram_range=(2,2)
                     tokenizer = token.tokenize)
text_counts= cv.fit_transform(tokenized_word)
 
 
words_df = pd.DataFrame(tokenized_word,columns=['words'])
#filter out 1 character words
words_df = words_df[words_df['words'].str.len()>2]

words_df['count']=1

words_df.count()

words_df= words_df.groupby('words').sum()
words_df.sort_values(by='count',ascending=False,inplace=True)



 