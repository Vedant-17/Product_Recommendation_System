import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Accept user input for the words to search
search_words = input("Enter the words to search: ")
words = search_words.split()

# Load the product data
products_data = pd.read_csv('flipkart.csv', encoding='latin-1')

# Preprocess the product names using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(products_data['product_name'])

# Compute the cosine similarity between the search query and each product name
query = ' '.join(words)
query_vector = vectorizer.transform([query])
similarity = cosine_similarity(X, query_vector)

# Get the top matching products and their scores
n_matches = 10
indices = similarity.ravel().argsort()[-n_matches:][::-1]
matches = products_data.iloc[indices][['product_name','retail_price']]
    
# Print the list of matching products
if len(matches) > 0:
    print("Matching products:")
    print(matches.to_string(index=True))
else:
    print("No matching products found.")
