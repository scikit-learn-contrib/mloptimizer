document.addEventListener("DOMContentLoaded", function() {
    // Select the toc-item and its ul (if exists)
    const tocItem = document.querySelector('.bd-toc-item ul');

    // Check if the ul doesn't exist
    if (!tocItem) {
        // Hide the entire sidebar if the toc ul is not found
        const sidebar = document.querySelector('.bd-sidebar');
        if (sidebar) {
            sidebar.style.display = 'none';
        }
    }

    // Force plotly figures to recalculate size on page load
    // This fixes the initial rendering issue with responsive plots
    setTimeout(function() {
        if (typeof Plotly !== 'undefined') {
            const plotlyDivs = document.querySelectorAll('.plotly-graph-div');
            plotlyDivs.forEach(function(div) {
                Plotly.Plots.resize(div);
            });
        }
    }, 100); // Small delay to ensure DOM is fully rendered
});
