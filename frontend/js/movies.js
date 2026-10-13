
document.addEventListener('DOMContentLoaded', () => {
    const grid = document.getElementById('moviesGrid');
    const searchInput = document.getElementById('searchInput');
    const genreFilter = document.getElementById('genreFilter');
    const sortOrder = document.getElementById('sortOrder');
    const clearBtn = document.getElementById('clearFiltersBtn');
    const emptyState = document.getElementById('emptyState');

    // Check that required HTML elements exist
    if (!grid || !searchInput || !genreFilter || !sortOrder || !clearBtn || !emptyState) {
        console.error('Movie page is missing one or more required HTML elements.');
        return;
    }

    // Check URL parameters for genre
    const urlParams = new URLSearchParams(window.location.search);
    const urlGenre = urlParams.get('genre');

    if (
        urlGenre &&
        Array.from(genreFilter.options).some(
            option => option.value === urlGenre
        )
    ) {
        genreFilter.value = urlGenre;
    }

    // Escape dynamic text before inserting it into HTML
    function escapeHTML(value) {
        return String(value ?? '').replace(/[&<>"']/g, char => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        })[char]);
    }

    // Render Movies
    function renderMovies() {
        const query = searchInput.value.trim().toLowerCase();
        const genre = genreFilter.value;
        const sort = sortOrder.value;

        // Filter movies by title and genre
        let filtered = (movies || []).filter(movie => {
            const title = String(movie.title || '').toLowerCase();
            const genres = movie.genres || [];

            const matchesSearch = title.includes(query);
            const matchesGenre =
                genre === 'All' || genres.includes(genre);

            return matchesSearch && matchesGenre;
        });

        // Add statistics for sorting and display
        filtered = filtered.map(movie => {
            const stats = getMovieStats(movie.id) || {};

            return {
                ...movie,
                stats: {
                    averageRating: Number(stats.averageRating) || 0,
                    count: Number(stats.count) || 0
                }
            };
        });

        // Sort movies
        filtered.sort((a, b) => {
            if (sort === 'titleAZ') {
                return a.title.localeCompare(b.title);
            }

            if (sort === 'titleZA') {
                return b.title.localeCompare(a.title);
            }

            if (sort === 'newest') {
                return (Number(b.year) || 0) - (Number(a.year) || 0);
            }

            if (sort === 'highestRated') {
                return b.stats.averageRating - a.stats.averageRating;
            }

            if (sort === 'mostReviewed') {
                return b.stats.count - a.stats.count;
            }

            return 0;
        });

        // Display empty state if no movies match
        if (filtered.length === 0) {
            grid.innerHTML = '';
            emptyState.style.display = 'block';
            return;
        }

        emptyState.style.display = 'none';

        // Render movie cards
        grid.innerHTML = filtered.map(movie => {
            const genresHTML = (movie.genres || [])
                .map(genreName => `
                    <span class="genre-tag">
                        ${escapeHTML(genreName)}
                    </span>
                `)
                .join('');

            return `
                <div class="movie-card">
                    <img
                        src="${escapeHTML(movie.poster || '')}"
                        alt="${escapeHTML(movie.title)}"
                        loading="lazy"
                    >

                    <div class="movie-card-info">
                        <h3>${escapeHTML(movie.title)}</h3>

                        <div class="meta-tags">
                            <span>${escapeHTML(movie.year)}</span>
                            ${genresHTML}
                        </div>

                        <div class="stats">
                            <span>
                                ★ ${movie.stats.averageRating.toFixed(1)}
                                (${movie.stats.count} reviews)
                            </span>

                            <a
                                href="movie-details.html?id=${encodeURIComponent(movie.id)}"
                                class="btn"
                                style="padding: 0.4rem 0.8rem; font-size: 0.9rem;"
                            >
                                View
                            </a>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    // Search movies while typing
    searchInput.addEventListener('input', renderMovies);

    // Filter by genre
    genreFilter.addEventListener('change', renderMovies);

    // Sort movies
    sortOrder.addEventListener('change', renderMovies);

    // Clear all filters
    clearBtn.addEventListener('click', () => {
        searchInput.value = '';
        genreFilter.value = 'All';
        sortOrder.value = 'newest';

        // Clear URL parameters without reloading
        window.history.replaceState({}, '', 'movies.html');

        renderMovies();
    });

    // Initial render
    renderMovies();
});
