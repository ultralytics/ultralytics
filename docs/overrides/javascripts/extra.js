document.addEventListener('DOMContentLoaded', function() {
    // Check if the user prefers dark mode
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        // Apply the dark theme
        document.body.classList.add('md-theme--slate');
    } else {
        // Apply the light theme
        document.body.classList.add('md-theme--default');
    }
});
console.log("extra.js is loaded!");  // in Chrome click 'inspect' and then view 'Console'
