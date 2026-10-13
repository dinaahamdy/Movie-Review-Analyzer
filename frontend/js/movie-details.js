document.addEventListener('DOMContentLoaded', () => {
    // Populate standard nav manually for snippet brevity, normally loaded globally
    document.querySelector('nav').innerHTML = `
        <a href="index.html" class="logo"><span>✨</span> MovieMind</a>
        <ul>
            <li><a href="index.html">Home</a></li>
            <li><a href="movies.html">Explore Movies</a></li>
            <li><a href="write-review.html">Write a Review</a></li>
            <li><a href="sentiment.html">Sentiment Insights</a></li>
            <li><a href="about-nlp.html">About NLP</a></li>
        </ul>
    `;

    const mainContent = document.getElementById('mainContent');
    const urlParams = new URLSearchParams(window.location.search);
    const movieId = urlParams.get('id');
    const movie = movies.find(m => m.id === movieId);

    if (!movie) {
        mainContent.innerHTML = `
            <div style="text-align: center; padding: 5rem 0;">
                <h2>Movie Not Found</h2>
                <p>The cinematic scroll you are looking for does not exist in this library.</p>
                <br><a href="movies.html" class="btn">Return to Collection</a>
            </div>`;
        return;
    }

    const reviews = getReviewsByMovieId(movieId);
    const stats = getMovieStats(movieId);
    
    // Calculate Sentiment Summary
    let pos = 0, neu = 0, neg = 0;
    reviews.forEach(r => {
        if(r.sentimentLabel === 'Positive') pos++;
        else if(r.sentimentLabel === 'Negative') neg++;
        else neu++;
    });
    const total = reviews.length;
    const posPct = total ? Math.round((pos/total)*100) : 0;
    const neuPct = total ? Math.round((neu/total)*100) : 0;
    const negPct = total ? Math.round((neg/total)*100) : 0;

    // Render Page
    mainContent.innerHTML = `
        <div class="movie-header">
            <img src="${movie.poster}" alt="${movie.title}" class="movie-poster">
            <div class="movie-info">
                <h1>${movie.title} (${movie.year})</h1>
                <div class="meta-tags">
                    ${movie.genres.map(g => `<span class="genre-tag">\${g}</span>`).join('')}
                    <span>• Dir: ${movie.director}</span>
                    <span>• ${movie.duration}</span>
                </div>
                <div class="stats" style="border:none; padding:0; font-size:1.2rem;">
                    <span>${generateStarHTML(Math.round(stats.averageRating))} ${stats.averageRating} (${stats.count} reviews)</span>
                </div>
                <p class="synopsis">"${movie.synopsis}"</p>
                
                <div class="sentiment-summary">
                    <h3>Sentiment Insights</h3>
                    ${total === 0 ? '<p>No reviews yet to analyze.</p>' : `
                        <p>Positive: \${posPct}% | Neutral: ${neuPct}\% \vert{} Negative:${negPct}%</p>
                        <div class="sentiment-bar">
                            <div style="width:\${posPct}%; background:#28a745;" title="Positive"></div>
                            <div style="width:\${neuPct}%; background:#6c757d;" title="Neutral"></div>
                            <div style="width:\${negPct}%; background:#dc3545;" title="Negative"></div>
                        </div>
                    `}
                </div>
                <br>
                <a href="write-review.html?movieId=${movie.id}" class="btn">Write a Review</a>
            </div>
        </div>

        <div class="reviews-section">
            <h2>Archive of Perspectives</h2>
            <div class="controls-bar" style="justify-content: flex-start; background: transparent; border: none; padding:0; margin-bottom:1rem;">
                <select id="reviewSort">
                    <option value="newest">Newest First</option>
                    <option value="oldest">Oldest First</option>
                    <option value="highest">Highest Rating</option>
                </select>
            </div>
            <div id="reviewsList"></div>
        </div>
    `;

    const reviewsList = document.getElementById('reviewsList');
    const reviewSort = document.getElementById('reviewSort');

    function renderReviews() {
        let sortedReviews = [...reviews];
        const sortVal = reviewSort.value;
        sortedReviews.sort((a,b) => {
            if(sortVal === 'newest') return new Date(b.timestamp) - new Date(a.timestamp);
            if(sortVal === 'oldest') return new Date(a.timestamp) - new Date(b.timestamp);
            if(sortVal === 'highest') return (b.rating || 0) - (a.rating || 0);
            return 0;
        });

        reviewsList.innerHTML = '';
        if(sortedReviews.length === 0) {
            reviewsList.innerHTML = `
                <div class="empty-state">
                    <h3>Be the first to share your thoughts!</h3>
                    <p>This movie's pages are blank. <a href="write-review.html?movieId=${movie.id}" style="color:var(--muted-gold);">Write a review</a>.</p>
                </div>`;
            return;
        }

        sortedReviews.forEach(r => {
            const dateStr = new Date(r.timestamp).toLocaleDateString();
            reviewsList.innerHTML += `
                <div class="review-card">
                    <div class="review-header">
                        <strong>${r.reviewerName || 'Anonymous'}</strong>
                        <span>${dateStr}</span>
                    </div>
                    <div style="margin-bottom:0.5rem;">${r.rating ? generateStarHTML(r.rating) : ''}</div>
                    <p style="margin-bottom:1rem;">"${r.reviewText}"</p>
                    <div>
                        <span class="sentiment-badge sentiment-${r.sentimentLabel}">${r.sentimentLabel}</span>
                        <span style="font-size:0.8rem; color:var(--soft-brown); margin-left:10px;">Method: ${r.sentimentMethod}</span>
                    </div>
                </div>
            `;
        });
    }

    if(reviewSort) {
        reviewSort.addEventListener('change', renderReviews);
    }
    renderReviews();
});