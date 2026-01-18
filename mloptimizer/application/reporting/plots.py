import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import matplotlib.ticker as ticker


def logbook_to_pandas(logbook):
    """
    Function to transform a deap logbook to a pandas dataframe.

    Parameters
    ----------
    logbook : deap.tools.Logbook
        The logbook to transform

    Returns
    -------
    df : pd.DataFrame
        The logbook as a pandas dataframe
    """
    data = logbook.chapters["parameters"]
    df = pd.DataFrame(data)
    return df



def plotly_logbook(logbook, population):
    """
    Generate a plot from the logbook with no additional points for "Max per Gen" and "Min per Gen".
    The population points grow based on the number of overlapping individuals.

    Parameters
    ----------
    logbook : deap.tools.Logbook
        The logbook to plot
    population : pd.DataFrame
        The population to plot

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure
    """
    # --- Prep ---------------------------------------------------------------
    #df = pd.DataFrame(logbook).rename(columns=str.lower).sort_values("gen")
    df = pd.DataFrame(logbook)
    xcat = df["gen"].astype(str)


    pop = population.copy()
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    pop["fitness"] = pd.to_numeric(pop["fitness"], errors="coerce")
    q = pop.groupby("population")["fitness"].quantile([0.25, 0.75]).unstack()
    q = q.rename(columns={0.25: "q1", 0.75: "q3"}).reindex(df["gen"]).interpolate()
    lower = q["q1"].clip(0, 1)
    upper = q["q3"].clip(0, 1)
    band_name = "Avg (IQR)"

    # y-range for readability
    data_min = float(min(df["min"].min(), lower.min()))
    data_max = float(max(df["max"].max(), upper.max()))
    y_min = data_min - (data_max - data_min)*0.1
    y_max = data_max + (data_max - data_min)*0.1

    # palette
    c_avg = "rgb(56,128,255)"  # blue
    c_best_gen = "rgb(147,112,219)"  # purple
    c_best_all = "rgb(255,140,0)"  # orange
    c_band = "rgba(56,128,255,0.10)"  # lighter so it doesn’t hide violins/boxes

    fig = go.Figure()

    # --- draw BAND FIRST (in the back) -------------------------------------
    fig.add_trace(go.Scatter(x=xcat, y=upper, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xcat, y=lower, mode="lines", name=band_name,
                             fill="tonexty", fillcolor=c_band, line=dict(width=0),
                             hoverinfo="skip"))

    # --- Individuals as DISTRIBUTIONS --------------------------------------
    tall = population.copy()
    tall["gen"] = pd.to_numeric(tall["population"], errors="coerce").astype("Int64")
    tall["fitness"] = pd.to_numeric(tall["fitness"], errors="coerce")
    tall = tall.dropna(subset=["gen", "fitness"])
    tall["gen"] = tall["gen"].astype(str)

    # Choose violin when n>=10, else box (violins look bad with tiny samples)
    counts = tall.groupby("gen")["fitness"].size()
    use_violin = (counts.min() >= 10)

    if use_violin:
        fig.add_trace(go.Violin(
            x=tall["gen"], y=tall["fitness"],
            name="Individuals",
            points=False,
            box_visible=True,
            meanline_visible=True,
            opacity=0.7,
            line=dict(width=1.2, color="rgba(30,30,30,0.7)"),
            fillcolor="rgba(90,160,255,0.45)",
            width=0.7,
            spanmode="hard",
            scalemode="width",
        ))
        fig.update_layout(violinmode="overlay")
    else:
        fig.add_trace(go.Box(
            x=tall["gen"], y=tall["fitness"],
            name="Individuals",
            boxpoints=False,  # keep it clean
            opacity=0.8,
            line=dict(width=1.2, color="rgba(30,30,30,0.7)"),
            fillcolor="rgba(90,160,255,0.35)",
        ))

    # --- Lines on top -------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=xcat, y=df["avg"], name="Average",
        mode="lines+markers",
        line=dict(color=c_avg, width=4),
        marker=dict(size=6, color=c_avg)
    ))

    fig.add_trace(go.Scatter(
        x=xcat, y=df["max"], name="Best per generation",
        mode="markers",
        line=dict(color=c_best_gen, width=2.5)
    ))

    # --- Layout -------------------------------------------------------------
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        xaxis=dict(type="category",
                   categoryorder="array",
                   categoryarray=[str(g) for g in df["gen"]]),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", y=1.02, yanchor="bottom",
                    x=0.5, xanchor="center", font=dict(size=14)),
        #margin=dict(l=60, r=40, t=50, b=60),
        width=950, height=560,
        font=dict(family="Arial", size=15)
    )

    return fig


def plot_logbook(logbook, font_size=12):
    """
    Generate sns figure from logbook. Evolution of fitness and population.

    Parameters
    ----------
    logbook : deap.tools.Logbook
        The logbook to plot
    font_size : int
        The font size of the labels, legends and ticks (-2). Default is 12.
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure
    """
    df = pd.DataFrame(logbook)
    g = sns.lineplot(data=df.drop(columns=['gen', 'nevals']), palette="colorblind")
    # Labels
    # 3 decimales en el eje Y
    g.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    g.set_xlabel("Generation", fontsize=font_size)
    g.set_ylabel("Fitness", fontsize=font_size)
    g.tick_params(axis='x', labelsize=font_size-2)
    g.tick_params(axis='y', labelsize=font_size-2)
    g.legend(fontsize=font_size)

    return g.get_figure()


def plotly_search_space(populations_df: pd.DataFrame, features: list = None, s=25,
                        marker_color: str = "blue",
                        colorscale: str = "Blues",
                        kde_line_color: str = "rgba(31, 119, 180, 1.0)",
                        kde_fillcolor: str = "rgba(31, 119, 180, 0.2)",
                        font_size: int = 10,
                        kde_resolution: int = 50,
                        max_scatter_points: int = 1000,
                        use_webgl: bool = True
                        ):
    """
    Generate plotly figure from populations dataframe and features. Search space.

    Parameters
    ----------
    populations_df : pd.DataFrame
        The dataframe with the population data
    features : list
        The features to plot (column names of the dataframe)
    s : int
        The size of the points
    marker_color : str
        The color of the scatter plot markers
    colorscale : str
        The colorscale for the KDE contour plots
    kde_line_color : str
        The line color for the KDE distribution plots
    kde_fillcolor : str
        The fill color for the KDE distribution plots
    font_size : int
        The font size for the axis labels and ticks
    kde_resolution : int
        Resolution of KDE grids (default 50). Lower = smaller file size.
        Use 25-35 for documentation, 50-100 for high-fidelity interactive use.
    max_scatter_points : int
        Maximum scatter points per subplot (default 1000). Data is sampled
        if exceeded. Set to None to disable sampling.
    use_webgl : bool
        If True (default), use Scattergl (WebGL) for scatter plots.
        Set to False to use SVG-based Scatter, which avoids WebGL crashes
        but may be slower with many points. Recommended False for docs.
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure
    """
    # Use specified features or default to all numeric columns
    if features is None:
        features = populations_df.select_dtypes(include="number").columns.tolist()
    else:
        missing_features = [f for f in features if f not in populations_df.columns]
        if missing_features:
            raise ValueError(f"Features not found in dataframe: {missing_features}")
        features = list(features)

    # Filter out features with zero variance (single unique value)
    # These produce empty subplots since KDE/scatter can't be computed
    features = [f for f in features if populations_df[f].nunique() > 1]

    # Sample data if too large to reduce file size and prevent WebGL crashes
    if max_scatter_points is not None and len(populations_df) > max_scatter_points:
        scatter_df = populations_df.sample(n=max_scatter_points, random_state=42)
    else:
        scatter_df = populations_df

    n_features = len(features)

    # Smart scaling based on number of features
    if n_features <= 3:
        fig_size = 700
        title_size = font_size + 4
        tick_size = font_size
        marker_size = max(3, int(s / 4 * (font_size / 11)))
        max_ticks = 6
    elif n_features <= 5:
        fig_size = 900
        title_size = font_size + 3
        tick_size = font_size - 1
        marker_size = max(2, int(s / 5 * (font_size / 11)))
        max_ticks = 5
    elif n_features <= 8:
        fig_size = 1000
        title_size = font_size + 2
        tick_size = font_size - 2
        marker_size = max(1, int(s / 6 * (font_size / 11)))
        max_ticks = 4
    else:
        fig_size = 1100
        title_size = font_size + 1
        tick_size = max(6, font_size - 3)
        marker_size = max(1, int(s / 8 * (font_size / 11)))
        max_ticks = 3

    # Create subplots with tighter spacing for many features
    spacing = max(0.01, 0.03 - (n_features * 0.002))

    fig = make_subplots(
        rows=n_features,
        cols=n_features,
        vertical_spacing=spacing,
        horizontal_spacing=spacing
    )

    for i, row_feature in enumerate(features):
        for j, col_feature in enumerate(features):
            row_idx = i + 1
            col_idx = j + 1

            if i < j:  # Upper triangle - scatter plots
                scatter_class = go.Scattergl if use_webgl else go.Scatter
                fig.add_trace(
                    scatter_class(
                        x=scatter_df[col_feature],
                        y=scatter_df[row_feature],
                        mode='markers',
                        marker=dict(
                            size=marker_size,
                            color=marker_color,
                            opacity=0.2
                        ),
                        showlegend=False,
                        hovertemplate=f'{col_feature}: %{{x}}<br>{row_feature}: %{{y}}<extra></extra>'
                    ),
                    row=row_idx,
                    col=col_idx
                )

            elif i > j:  # Lower triangle - KDE contour plots
                x_data = populations_df[col_feature].dropna()
                y_data = populations_df[row_feature].dropna()

                try:
                    if len(x_data) > 1 and len(y_data) > 1 and np.nanstd(x_data) > 0 and np.nanstd(y_data) > 0:
                        xmin, xmax = x_data.min(), x_data.max()
                        ymin, ymax = y_data.min(), y_data.max()

                        # Use configurable resolution for smaller file sizes
                        grid_res = complex(0, kde_resolution)
                        xx, yy = np.mgrid[xmin:xmax:grid_res, ymin:ymax:grid_res]
                        positions = np.vstack([xx.ravel(), yy.ravel()])
                        values = np.vstack([x_data, y_data])
                        kernel = stats.gaussian_kde(values)
                        f = np.reshape(kernel(positions).T, xx.shape)

                        fig.add_trace(
                            go.Contour(
                                x=xx[:, 0],
                                y=yy[0, :],
                                z=f.T,
                                colorscale=colorscale,
                                showscale=False,
                                contours=dict(
                                    coloring='fill',
                                    showlines=False
                                ),
                                opacity=0.6,
                                hovertemplate=f'{col_feature}: %{{x}}<br>{row_feature}: %{{y}}<extra></extra>'
                            ),
                            row=row_idx,
                            col=col_idx
                        )
                except:
                    pass

            else:  # Diagonal - KDE distribution plots
                data = populations_df[row_feature].dropna()

                try:
                    if len(data) > 1 and np.nanstd(data) > 0:
                        kde = stats.gaussian_kde(data)
                        # Use 2x kde_resolution for 1D plots (still much smaller than contours)
                        x_range = np.linspace(data.min(), data.max(), kde_resolution * 2)
                        y_kde = kde(x_range)

                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=y_kde,
                                fill='tozeroy',
                                mode='lines',
                                line=dict(color=kde_line_color, width=1),
                                fillcolor=kde_fillcolor,
                                showlegend=False,
                                hovertemplate=f'{row_feature}: %{{x}}<br>Density: %{{y}}<extra></extra>'
                            ),
                            row=row_idx,
                            col=col_idx
                        )
                except:
                    pass

            # Update axis labels only on edges
            if i == n_features - 1:  # Bottom row
                fig.update_xaxes(
                    title_text=col_feature,
                    title_font=dict(size=title_size),
                    tickfont=dict(size=tick_size),
                    nticks=max_ticks,
                    row=row_idx,
                    col=col_idx
                )
            if j == 0:  # Left column
                fig.update_yaxes(
                    title_text=row_feature,
                    title_font=dict(size=title_size),
                    tickfont=dict(size=tick_size),
                    nticks=max_ticks,
                    row=row_idx,
                    col=col_idx
                )

    # Fixed size with smart margins
    margin_size = max(40, 80 - n_features * 3)

    fig.update_layout(
        height=fig_size,
        width=fig_size,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=margin_size, r=20, t=20, b=margin_size)
    )

    # Update all axes with adaptive tick sizing
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=tick_size), nticks=max_ticks)
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(size=tick_size), nticks=max_ticks)

    return fig


def save_plotly_figure(fig, path, for_docs=False):
    """
    Save a Plotly figure with optimized settings.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to save
    path : str
        Output file path. Extension determines format:
        - .html: Interactive HTML (use for_docs=True for smaller files)
        - .png, .jpg, .svg, .pdf: Static image (requires kaleido)
    for_docs : bool
        If True, optimizes for documentation (smaller file, no MathJax).
        Recommended for Sphinx galleries.

    Returns
    -------
    str
        The path where the file was saved
    """
    import os
    ext = os.path.splitext(path)[1].lower()

    if ext == '.html':
        # HTML output with size optimizations
        fig.write_html(
            path,
            include_plotlyjs='cdn',  # Load from CDN instead of embedding (~3MB saved)
            full_html=not for_docs,  # Minimal HTML for docs
            include_mathjax=False,   # Skip MathJax if not needed
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'responsive': True,  # Resize plot to fit container width
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'] if for_docs else []
            }
        )
    elif ext in ('.png', '.jpg', '.jpeg', '.svg', '.pdf', '.webp'):
        # Static image - ideal for documentation, no WebGL issues
        fig.write_image(path, scale=2 if ext in ('.png', '.jpg', '.jpeg', '.webp') else 1)
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .html, .png, .jpg, .svg, .pdf, or .webp")

    return path


def plotly_search_space_for_docs(populations_df: pd.DataFrame, features: list = None,
                                  output_path: str = None, image_format: str = "png"):
    """
    Generate a documentation-optimized search space plot.

    This is a convenience function that creates smaller, faster-loading plots
    suitable for Sphinx documentation and web embedding. It uses aggressive
    size optimizations and can save directly to static image formats.

    Parameters
    ----------
    populations_df : pd.DataFrame
        The dataframe with the population data
    features : list, optional
        The features to plot (column names). If None, uses all numeric columns.
    output_path : str, optional
        If provided, saves the figure to this path. Format determined by extension
        (.png, .svg, .html). If None, returns the figure without saving.
    image_format : str
        Default format when output_path has no extension. One of: 'png', 'svg', 'html'.
        Recommended: 'png' or 'svg' for docs (no WebGL issues).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure object

    Examples
    --------
    >>> # For Sphinx documentation - saves as static PNG (recommended)
    >>> fig = plotly_search_space_for_docs(df, output_path="search_space.png")

    >>> # For interactive HTML with minimal size
    >>> fig = plotly_search_space_for_docs(df, output_path="search_space.html")

    >>> # Just get the figure without saving
    >>> fig = plotly_search_space_for_docs(df)
    """
    import os

    # Create figure with documentation-optimized settings
    fig = plotly_search_space(
        populations_df,
        features=features,
        kde_resolution=25,       # Low resolution for small file size
        max_scatter_points=300,  # Minimal scatter points
        font_size=9,             # Slightly smaller fonts
        s=15,                    # Smaller markers
        use_webgl=False          # SVG-based scatter avoids WebGL crashes
    )

    # Save if path provided
    if output_path:
        ext = os.path.splitext(output_path)[1].lower()
        if not ext:
            output_path = f"{output_path}.{image_format}"
        save_plotly_figure(fig, output_path, for_docs=True)

    return fig


def plot_search_space(populations_df: pd.DataFrame, height=2, s=25):
    """
    Generate sns figure from populations dataframe and features. Search space.

    Parameters
    ----------
    populations_df : pd.DataFrame
        The dataframe with the population
    height : int
        The height of the figure
    s : int
        The size of the points

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure
    """
    sns.set_style("white")
    g = sns.PairGrid(populations_df, diag_sharey=False, height=height)
    g = g.map_upper(sns.scatterplot, s=s, color="r", alpha=0.2)
    g = g.map_lower(
        sns.kdeplot,
        shade=True,
        cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    )
    g = g.map_diag(sns.kdeplot, shade=True, palette="crest", alpha=0.2, color="red")
    return g
