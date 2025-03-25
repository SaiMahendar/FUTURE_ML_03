import sys
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets with error handling
try:
    books = pd.read_csv('books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv('ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
except FileNotFoundError as e:
    print(f"Error: File not found - {e}. Please ensure 'books.csv' and 'ratings.csv' are in the correct directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

# Merge datasets on ISBN
data = pd.merge(ratings, books, on='ISBN', how='inner')

# Debug: Inspect the dataset
print("Merged dataset columns:", data.columns.tolist())
print("Merged dataset shape:", data.shape)
print("Merged dataset head:\n", data.head(10))

# Debug: Check if "Harry Potter" exists in the dataset
harry_books = data[data['Book-Title'].str.contains("Harry Potter", case=False, na=False)]
print(f"Number of Harry Potter books in dataset: {len(harry_books)}")
if not harry_books.empty:
    print("Sample Harry Potter books found:")
    print(harry_books[['Book-Title', 'Book-Author']].head())

# Sample data with key examples
key_books = data[data['Book-Title'].str.contains("Clara Callan|Harry Potter", case=False, na=False)]
data_sample = pd.concat([data.sample(n=min(5000, len(data)-len(key_books)), random_state=42), 
                         key_books]).drop_duplicates(subset=['ISBN'])
data_sample = data_sample.dropna(subset=['Book-Title', 'Book-Author']).reset_index(drop=True)

# Debug: Check if "Harry Potter" made it into the sample
harry_sample = data_sample[data_sample['Book-Title'].str.contains("Harry Potter", case=False, na=False)]
print(f"Number of Harry Potter books in sample: {len(harry_sample)}")
if not harry_sample.empty:
    print("Sample Harry Potter books in sample:")
    print(harry_sample[['Book-Title', 'Book-Author']].head())

# TF-IDF for content-based recommendations
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(data_sample['Book-Title'] + ' ' + data_sample['Book-Author'])

# Content-based recommendation function
def get_content_recommendations(title):
    # Normalize input
    title = title.strip().lower()
    
    # Search by title with regex for better matching
    mask = data_sample['Book-Title'].apply(lambda x: bool(re.search(title, str(x).lower())) if pd.notna(x) else False)
    if not mask.any():
        # Fallback: search by author
        mask = data_sample['Book-Author'].apply(lambda x: bool(re.search(title, str(x).lower())) if pd.notna(x) else False)
        if not mask.any():
            # Fallback: use TF-IDF to find closest matches
            query_vec = tfidf.transform([title])
            sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            sim_indices = sim_scores.argsort()[-6:-1][::-1]
            if sim_scores[sim_indices[0]] < 0.1:  # Threshold for relevance
                return None
            return data_sample.iloc[sim_indices][['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 
                                                  'Publisher', 'Image-URL-M', 'Book-Rating']]
    idx = data_sample[mask].index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-6:-1][::-1]
    return data_sample.iloc[sim_indices][['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 
                                          'Publisher', 'Image-URL-M', 'Book-Rating']]

# Main function to search and print results to terminal
def main():
    while True:
        user_input = input("Enter a book title or author to search (e.g., 'Harry Potter' or 'J.K. Rowling'), or 'exit' to quit: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting program.")
            break
        
        if not user_input:
            print("Error: Please enter a book title or author.")
            continue

        try:
            recs = get_content_recommendations(user_input)
        except Exception as e:
            print(f"An error occurred while fetching recommendations: {e}")
            continue

        if not recs is None and not recs.empty:
            print("\nRecommendations Found:")
            print("-" * 50)
            for idx, book in recs.iterrows():
                print(f"Title: {book['Book-Title']}")
                print(f"Author: {book['Book-Author']}")
                print(f"Year: {book['Year-Of-Publication']}")
                print(f"Publisher: {book['Publisher']}")
                print(f"Rating: {book['Book-Rating']}")
                print(f"ISBN: {book['ISBN']}")
                print(f"Image URL: {book['Image-URL-M']}")
                print("-" * 50)
        else:
            print("Sorry, no recommendations found for your input. Try a different title or author.")

if __name__ == "__main__":
    main()