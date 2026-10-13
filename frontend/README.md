# MovieMind - Magical Movie Discovery & Sentiment Analysis

MovieMind is an educational frontend project demonstrating web development and a baseline Natural Language Processing (NLP) sentiment classifier. It requires no backend, database, or API keys.

## Features
- **Magical UI:** A responsive, dreamy fantasy-library design built with plain HTML/CSS.
- **Movie Browsing:** Search, filter, and sort a dynamic movie collection.
- **Persistent Reviews:** Write and save reviews using the browser's `localStorage`.
- **Baseline Sentiment Analysis:** Analyzes review text (Positive/Neutral/Negative) using a vanilla JS word-lexicon classifier.
- **Dynamic Dashboards:** Calculates real-time sentiment statistics and displays charts via Chart.js.

## Setup & Testing Instructions
1. **Run Locally:** No build step is required. Simply extract all files into a folder and double-click `index.html` to open it in your browser. (For the best experience, use a local server like VS Code Live Server).
2. **Test Persistence:** Submit a review on `write-review.html`, then refresh the page or go to the movie's details page. You will see your review remains saved in `localStorage`.
3. **Test NLP Logic:** Try writing a review with words like "amazing", "beautiful", "not good", or "terrible" to see how the baseline classifier categorizes your sentiment on the dashboard.