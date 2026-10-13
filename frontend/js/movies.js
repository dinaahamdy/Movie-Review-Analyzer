document.addEventListener('DOMContentLoaded', () => {
    const grid = document.getElementById('moviesGrid');
    const searchInput = document.getElementById('searchInput');
    const genreFilter = document.getElementById('genreFilter');
    const sortOrder = document.getElementById('sortOrder');
    const clearBtn = document.getElementById('clearFiltersBtn');
    const emptyState = document.getElementById('emptyState');

    // Check URL parameters for genre
    const urlParams = new URLSearchParams(window.location.search);
    const urlGenre = urlParams.get('genre');
    if(urlGenre && Array.from(genreFilter.options).some(opt => opt.value === urlGenre)) {
        genreFilter.value = urlGenre;
    }

    function renderMovies() {
        const query = searchInput.value.toLowerCase();
        const genre = genreFilter.value;
        const sort = sortOrder.value;

        // Filter
        let filtered = movies.filter(m => {
            const matchesSearch = m.title.toLowerCase().includes(query);
            const matchesGenre = genre === 'All' || m.genres.includes(genre);
            return matchesSearch && matchesGenre;
        });

        // Add stats for sorting
        filtered = filtered.map(m => {
            const stats = getMovieStats(m.id);
            return { ...m, stats };
        });

        // Sort
        filtered.sort((a, b) => {
            if (sort === 'titleAZ') return a.title.localeCompare(b.title);
            if (sort === 'titleZA') return b.title.localeCompare(a.title);
            if (sort === 'newest') return b.year - a.year;
            if (sort === 'highestRated') return b.stats.averageRating - a.stats.averageRating;
            if (sort === 'mostReviewed') return b.stats.count - a.stats.count;
            return 0;
        });

        grid.innerHTML = '';
        if (filtered.length === 0) {
            emptyState.style.display = 'block';
        } else {
            emptyState.style.display = 'none';
            filtered.forEach(movie => {
                grid.innerHTML += `
                    <div class="movie-card">
                        <img src="${movie.poster}" alt="${movie.title}">
                        <div class="movie-card-info">
                            <h3>${movie.title}</h3>
                            <div class="meta-tags">
                                <span>${movie.year}</span> 
                                ${movie.genres.map(g => `<span class="genre-tag">\${g}</span>`).join('')}
                            </div>
                            <div class="stats">
                                <span>★ ${movie.stats.averageRating} (${movie.stats.count} reviews)</span>
                                <a href="movie-details.html?id=${movie.id}" class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.9rem;">View</a>
                            </div>
                        </div>
                    </div>
                `;
            });
        }
    }

    searchInput.addEventListener('input', renderMovies);
    genreFilter.addEventListener('change', renderMovies);
    sortOrder.addEventListener('change', renderMovies);
    clearBtn.addEventListener('click', () => {
        searchInput.value = '';
        genreFilter.value = 'All';
        sortOrder.value = 'newest';
        window.history.replaceState({}, '', 'movies.html'); // Clear URL param silently
        renderMovies();
    });

    renderMovies();
});