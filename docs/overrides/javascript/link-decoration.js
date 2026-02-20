(function() {
    const tagLinks = () => {
        document.querySelectorAll('a[href*="www.ultralytics.com"]').forEach(link => {
            // Avoid double-processing the same link
            if (link.dataset.utmTagged) return;

            let currentPath = window.location.pathname;
            if (currentPath === "/") {
                currentPath = "homepage";
            } else {
                currentPath = currentPath.replace(/^\/|\/$/g, '');
            }

            try {
                const url = new URL(link.href);
                const params = url.searchParams;

                // Only set defaults if the parameter doesn't already exist
                if (!params.has('utm_source')) params.set('utm_source', 'docs.ultralytics.com');
                if (!params.has('utm_medium')) params.set('utm_medium', 'referral');
                if (!params.has('utm_campaign')) params.set('utm_campaign', 'docs_to_web');
                if (!params.has('utm_content')) params.set('utm_content', currentPath);

                // Update the link href and mark it as tagged
                link.href = decodeURIComponent(url.toString());
                link.dataset.utmTagged = "true"; 
            } catch (e) {
                console.error("UTM Tagging failed for link:", link.href);
            }
        });
    };

    // 1. Run immediately on initial load
    tagLinks();

    // 2. Watch for dynamic content changes (MutationObserver)
    const observer = new MutationObserver(() => {
        tagLinks();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
})();