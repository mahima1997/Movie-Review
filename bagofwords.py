import pandas as pd
import re
from bs4 import BeautifulSoup                                                        #Removing HTML Markup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("labeledTrainData.tsv", header=0,delimiter="\t", quoting=3)     # "header=0" indicates that the first line of the file contains
                                                                                    # column names, "delimiter=\t" indicates that the fields are
                                                                                    # separated by tabs, and quoting=3 tells Python to ignore
                                                                                    # doubled quotes

#For many problems, it makes sense to remove punctuation. On the other hand, in this case, we are tackling a sentiment analysis problem,
# and it is possible that "!!!" or ":-(" could carry sentiment, and should be treated as words.Here we are removing the punctuation altogether,


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)

    # 1. Remove HTML tags i.e library is used for pulling data out of HTML and XML files
    review_text = BeautifulSoup(raw_review).get_text()    #Calling get_text(),Beautifulsoup function gives the text of the review, without tags or markup.
    # 2. Regex used here says Find anything that is NOT a lowercase letter (a-z) or an upper case letter (A-Z), and replace it with a space."
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    # 4. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    #as this function has to be called many times the search time needs to be removed
    stops = set(stopwords.words("english"))
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    # 6. Join the words back into one string separated by space, and return the result.
    return( " ".join( meaningful_words ))

#There are many other things we could do to the data - For example, Porter Stemming and Lemmatizing (both available in NLTK) would allow us to
#treat "messages", "message", and "messaging" as the same word, which could certainly be useful.

# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

print "Cleaning and parsing the training set movie reviews...\n"
# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
for i in xrange(0, num_reviews):
    # If the index is evenly divisible by 1000, print a message
    if((i+1)%1000 == 0):
        print "Review %d of %d\n" % (i+1, num_reviews)
        # Call our function for each one, and add the result to the list of clean reviews
    clean_train_reviews.append( review_to_words(train["review"][i]))

print "Creating the bag of words...\n"

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None, stop_words = None,max_features = 5000)

# fit_transform() fits the model and learns the vocabulary; second, it transforms our training data into feature vectors.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an array The input to fit_transform should be a list of strings.
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()

print "Training the random forest..."

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"])

# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t",quoting=3 )

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...\n")
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

output = pd.DataFrame(data={"id":test["id"], "sentiment":result} )

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )


