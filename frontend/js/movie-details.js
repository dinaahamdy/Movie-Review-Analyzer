
document.addEventListener('DOMContentLoaded', () => {
    // Populate Navigation
    const nav = document.querySelector('nav');

    if (nav) {
        nav.innerHTML = `
            <a href="index.html" class="logo">
                <span>✨</span> MovieMind
            </a>
            <ul>
                <li><a href="index.html">Home</a></li>
                <li><a href="movies.html">Explore Movies</a></li>
                <li><a href="write-review.html">Write a Review</a></li>
                <li><a href="sentiment.html">Sentiment Insights</a></li>
                <li><a href="about-nlp.html">About NLP</a></li>
            </ul>
        `;
    }

    const mainContent = document.getElementById('mainContent');

    if (!mainContent) {
        console.error('Element #mainContent was not found.');
        return;
    }

    // Get Movie ID from URL
    const urlParams = new URLSearchParams(window.location.search);
    const movieId = urlParams.get('id');

    // Find the selected movie
    const movie = movies.find(m => String(m.id) === String(movieId));

    // Handle missing movie
    if (!movie) {
        mainContent.innerHTML = `
            <div style="text-align: center; padding: 5rem 0;">
                <h2>Movie Not Found</h2>
                <p>
                    The cinematic scroll you are looking for
                    does not exist in this library.
                </p>
                <br>
                <a href="movies.html" class="btn">
                    Return to Collection
                </a>
            </div>
        `;
        return;
    }

    // Get reviews and movie statistics
    const reviews = getReviewsByMovieId(movieId) || [];
    const stats = getMovieStats(movieId);

    // Calculate Sentiment Summary
    let pos = 0;
    let neu = 0;
    let neg = 0;

    reviews.forEach(review => {
        if (review.sentimentLabel === 'Positive') {
            pos++;
        } else if (review.sentimentLabel === 'Negative') {
            neg++;
        } else {
            neu++;
        }
    });

    const total = reviews.length;

    const posPct = total
        ? Math.round((pos / total) * 100)
        : 0;

    const neuPct = total
        ? Math.round((neu / total) * 100)
        : 0;

    const negPct = total
        ? 100 - posPct - neuPct
        : 0;

    // Safely escape text before inserting it into HTML
    function escapeHTML(value) {
        return String(value ?? '').replace(/[&<>"']/g, char => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        })[char]);
    }

    // Render Movie Details
    mainContent.innerHTML = `
        <div class="movie-header">

            <img
                src="${escapeHTML(movie.poster)}"
                alt="${escapeHTML(movie.title)}"
                class="movie-poster"
            >

            <div class="movie-info">

                <h1>
                    ${escapeHTML(movie.title)}
                    (${escapeHTML(movie.year)})
                </h1>

                <div class="meta-tags">
                    ${(movie.genres || []).map(genre => `
                        <span class="genre-tag">
                            ${escapeHTML(genre)}
                        </span>
                    `).join('')}

                    <span>
                        • Dir: ${escapeHTML(movie.director)}
                    </span>

                    <span>
                        • ${escapeHTML(movie.duration)}
                    </span>
                </div>

                <div
                    class="stats"
                    style="border:none; padding:0; font-size:1.2rem;"
                >
                    <span>
                        ${generateStarHTML(
                            Math.round(Number(stats.averageRating) || 0)
                        )}
                        ${Number(stats.averageRating || 0).toFixed(1)}
                        (${stats.count || 0} reviews)
                    </span>
                </div>

                <p class="synopsis">
                    "${escapeHTML(movie.synopsis)}"
                </p>

                <div class="sentiment-summary">
                    <h3>Sentiment Insights</h3>

                    ${total === 0 ? `
                        <p>No reviews yet to analyze.</p>
                    ` : `
                        <p>
                            Positive: ${posPct}% |
                            Neutral: ${neuPct}% |
                            Negative: ${negPct}%
                        </p>

                        <div
                            class="sentiment-bar"
                            style="
                                display:flex;
                                width:100%;
                                height:12px;
                                overflow:hidden;
                                border-radius:6px;
                            "
                        >
                            <div
                                style="
                                    width:${posPct}%;
                                    background:#28a745;
                                "
                                title="Positive: ${posPct}%"
                            ></div>

                            <div
                                style="
                                    width:${neuPct}%;
                                    background:#6c757d;
                                "
                                title="Neutral: ${neuPct}%"
                            ></div>

                            <div
                                style="
                                    width:${negPct}%;
                                    background:#dc3545;
                                "
                                title="Negative: ${negPct}%"
                            ></div>
                        </div>
                    `}
                </div>

                <br>

                <a
                    href="write-review.html?movieId=${encodeURIComponent(movie.id)}"
                    class="btn"
                >
                    Write a Review
                </a>

            </div>
        </div>

        <div class="reviews-section">

            <h2>Archive of Perspectives</h2>

            <div
                class="controls-bar"
                style="
                    justify-content:flex-start;
                    background:transparent;
                    border:none;
                    padding:0;
                    margin-bottom:1rem;
                "
            >
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

    // Render and Sort Reviews
    function renderReviews() {
        const sortedReviews = [...reviews];
        const sortVal = reviewSort.value;

        sortedReviews.sort((a, b) => {
            if (sortVal === 'newest') {
                return new Date(b.timestamp || 0)
                    - new Date(a.timestamp || 0);
            }

            if (sortVal === 'oldest') {
                return new Date(a.timestamp || 0)
                    - new Date(b.timestamp || 0);
            }

            if (sortVal === 'highest') {
                return (Number(b.rating) || 0)
                    - (Number(a.rating) || 0);
            }

            return 0;
        });

        if (sortedReviews.length === 0) {
            reviewsList.innerHTML = `
                <div class="empty-state">
                    <h3>Be the first to share your thoughts!</h3>

                    <p>
                        This movie's pages are blank.
                        <a
                            href="write-review.html?movieId=${encodeURIComponent(movie.id)}"
                            style="color:var(--muted-gold);"
                        >
                            Write a review
                        </a>.
                    </p>
                </div>
            `;
            return;
        }

        reviewsList.innerHTML = sortedReviews.map(review => {
            const date = new Date(review.timestamp);
            const dateStr = Number.isNaN(date.getTime())
                ? 'Unknown date'
                : date.toLocaleDateString();

            const sentiment = [
                'Positive',
                'Neutral',
                'Negative'
            ].includes(review.sentimentLabel)
                ? review.sentimentLabel
                : 'Neutral';

            const rating = Number(review.rating) || 0;

            return `
                <div class="review-card">

                    <div class="review-header">
                        <strong>
                            ${escapeHTML(
                                review.reviewerName || 'Anonymous'
                            )}
                        </strong>

                        <span>${escapeHTML(dateStr)}</span>
                    </div>

                    <div style="margin-bottom:0.5rem;">
                        ${rating > 0
                            ? generateStarHTML(rating)
                            : ''
                        }
                    </div>

                    <p style="margin-bottom:1rem;">
                        "${escapeHTML(review.reviewText)}"
                    </p>

                    <div>
                        <span class="sentiment-badge sentiment-${sentiment}">
                            ${sentiment}
                        </span>

                        <span
                            style="
                                font-size:0.8rem;
                                color:var(--soft-brown);
                                margin-left:10px;
                            "
                        >
                            Method:
                            ${escapeHTML(
                                review.sentimentMethod || 'Unknown'
                            )}
                        </span>
                    </div>

                </div>
            `;
        }).join('');
    }

    // Update reviews when sort selection changes
    if (reviewSort) {
        reviewSort.addEventListener('change', renderReviews);
    }

    // Initial render
    renderReviews();
});
