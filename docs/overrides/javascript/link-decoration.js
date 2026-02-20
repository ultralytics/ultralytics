(function() {
    const tagLinks = () => {
        document.querySelectorAll('a[href*="www.ultralytics.com"]').forEach(link => {
            // Avoid double-tagging if the script runs multiple times
            if (link.dataset.utmTagged) return;

            let currentPath = window.location.pathname;
            if (currentPath === "/") {
                currentPath = "homepage";
            } else {
                currentPath = currentPath.replace(/^\/|\/$/g, '');
            }

            try {
                const url = new URL(link.href);
                url.searchParams.set('utm_source', 'docs.ultralytics.com');
                url.searchParams.set('utm_medium', 'referral');
                url.searchParams.set('utm_campaign', 'docs_to_web');
                url.searchParams.set('utm_content', currentPath);
                
                link.href = decodeURIComponent(url.toString());
                link.dataset.utmTagged = "true"; // Mark as tagged
            } catch (e) {
                console.error("UTM Tagging failed for link:", link.href);
            }
        });
    };

    // 1. Run immediately on initial load
    tagLinks();

    // 2. Watch for "Zensify" content swaps (MutationObserver)
    const observer = new MutationObserver(() => {
        tagLinks();
    });

    // We watch the 'body' for any structural changes (standard for SPA/PJAX)
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
})();