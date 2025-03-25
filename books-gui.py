import sys
import pandas as pd
import numpy as np
import re
import base64
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QScrollArea
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO

# Load datasets
try:
    books = pd.read_csv('books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv('ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
except FileNotFoundError as e:
    print(f"Error: File not found - {e}. Please ensure 'books.csv' and 'ratings.csv' are in the correct directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

# Merge datasets
data = pd.merge(ratings, books, on='ISBN', how='inner')
data_sample = data.dropna(subset=['Book-Title', 'Book-Author']).reset_index(drop=True)

# TF-IDF for content-based recommendations
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(data_sample['Book-Title'] + ' ' + data_sample['Book-Author'])

def get_content_recommendations(title):
    title = title.strip().lower()
    mask = data_sample['Book-Title'].str.lower().str.contains(title, na=False)
    if not mask.any():
        query_vec = tfidf.transform([title])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        sim_indices = sim_scores.argsort()[-6:-1][::-1]
        if sim_scores[sim_indices[0]] < 0.1:
            return None
        return data_sample.iloc[sim_indices]
    idx = data_sample[mask].index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-6:-1][::-1]
    return data_sample.iloc[sim_indices]

def fetch_image_base64(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Referer': 'https://www.amazon.com/'  # Adding a referer might help
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None

def get_star_rating(rating):
    try:
        rating = int(rating)
        return "⭐" * rating
    except:
        return "No Rating"

class BookRecommenderGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel("Enter a book title or author:")
        self.inputField = QLineEdit()
        self.searchButton = QPushButton("Search")
        self.resultArea = QVBoxLayout()
        self.scrollArea = QScrollArea()
        self.resultWidget = QWidget()
        self.resultWidget.setLayout(self.resultArea)
        self.scrollArea.setWidget(self.resultWidget)
        self.scrollArea.setWidgetResizable(True)

        layout.addWidget(self.label)
        layout.addWidget(self.inputField)
        layout.addWidget(self.searchButton)
        layout.addWidget(self.scrollArea)
        
        self.searchButton.clicked.connect(self.searchBook)
        
        self.setLayout(layout)
        self.setWindowTitle("Book Recommendation System")
        self.resize(600, 500)

    def searchBook(self):
        title = self.inputField.text().strip()
        if not title:
            self.clearResults()
            error_label = QLabel("Error: Please enter a book title or author.")
            self.resultArea.addWidget(error_label)
            return
        
        recs = get_content_recommendations(title)
        self.clearResults()
        
        if recs is not None and not recs.empty:
            for _, book in recs.iterrows():
                book_layout = QHBoxLayout()
                book_info = QLabel(f"<b>Title:</b> {book['Book-Title']}<br>"
                                 f"<b>Author:</b> {book['Book-Author']}<br>"
                                 f"<b>Year:</b> {book['Year-Of-Publication']}<br>"
                                 f"<b>Publisher:</b> {book['Publisher']}<br>"
                                 f"<b>Rating:</b> {get_star_rating(book['Book-Rating'])}<br>"
                                 f"<b>ISBN:</b> {book['ISBN']}")
                book_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                
                image_label = QLabel()
                image_url = book.get('Image-URL-M', '')  # Safely get image URL
                if image_url:
                    image_data = fetch_image_base64(image_url)
                    if image_data:
                        pixmap = QPixmap()
                        if pixmap.loadFromData(image_data):
                            scaled_pixmap = pixmap.scaled(100, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                            image_label.setPixmap(scaled_pixmap)
                        else:
                            print(f"Failed to load image data for {book['Book-Title']}")
                    else:
                        # Fallback placeholder
                        placeholder_pixmap = QPixmap(100, 150)
                        placeholder_pixmap.fill(Qt.GlobalColor.gray)
                        image_label.setPixmap(placeholder_pixmap)
                else:
                    # No URL available
                    placeholder_pixmap = QPixmap(100, 150)
                    placeholder_pixmap.fill(Qt.GlobalColor.gray)
                    image_label.setPixmap(placeholder_pixmap)
                
                book_layout.addWidget(image_label)
                book_layout.addWidget(book_info)
                self.resultArea.addLayout(book_layout)
        else:
            error_label = QLabel("Sorry, no recommendations found.")
            self.resultArea.addWidget(error_label)
    
    def clearResults(self):
        while self.resultArea.count():
            item = self.resultArea.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clearLayout(item.layout())
    
    def clearLayout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clearLayout(item.layout())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BookRecommenderGUI()
    window.show()
    sys.exit(app.exec())
