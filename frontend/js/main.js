// Global scripts (Navigation, Mobile Menu, Utility functions)
document.addEventListener('DOMContentLoaded', () => {
    // Mobile Menu Toggle
    const menuBtn = document.createElement('button');
    menuBtn.classList.add('mobile-menu-btn');
    menuBtn.innerHTML = '☰';
    const nav = document.querySelector('nav');
    if(nav) {
        nav.insertBefore(menuBtn, nav.querySelector('ul'));
        menuBtn.addEventListener('click', () => {
            nav.querySelector('ul').classList.toggle('show-mobile');
        });
    }

    // Highlight active nav
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('nav ul li a').forEach(link => {
        if (link.getAttribute('href') === currentPage) {
            link.classList.add('active');
        }
    });
});

// Render Star UI
function generateStarHTML(rating) {
    let stars = '';
    for(let i=1; i<=5; i++) {
        stars += i <= rating ? '<span class="star filled">★</span>' : '<span class="star">☆</span>';
    }
    return stars;
}