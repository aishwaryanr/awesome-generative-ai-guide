const slides = document.querySelectorAll('.slide');
let current = 0;

function showSlide(n) {
    slides[current].classList.remove('active');
    current = (n + slides.length) % slides.length;
    slides[current].classList.add('active');
    const currentEl = document.getElementById('current');
    const totalEl = document.getElementById('total');
    const progress = document.getElementById('progress');
    if (currentEl) currentEl.textContent = current + 1;
    if (totalEl) totalEl.textContent = slides.length;
    if (progress) progress.style.width = ((current + 1) / slides.length * 100) + '%';
}

function nextSlide() { showSlide(current + 1); }
function prevSlide() { showSlide(current - 1); }

// Expose for onclick handlers
window.nextSlide = nextSlide;
window.prevSlide = prevSlide;

// Keyboard navigation
document.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); nextSlide(); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); prevSlide(); }
});

// Initialize
if (slides.length > 0) {
    showSlide(0);
}
