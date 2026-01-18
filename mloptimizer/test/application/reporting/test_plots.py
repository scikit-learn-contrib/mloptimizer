import pytest
from mloptimizer.application.reporting.plots import logbook_to_pandas, plot_logbook, plot_search_space, \
    plotly_logbook, plotly_search_space, plotly_search_space_for_docs
from mloptimizer.domain.optimization import Optimizer
from mloptimizer.domain.hyperspace import HyperparameterSpace
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


@pytest.fixture
def default_tree_optimizer():
    x, y = load_iris(return_X_y=True)
    default_hyperparameter_space = HyperparameterSpace.get_default_hyperparameter_space(DecisionTreeClassifier)
    genetic_params = {
        "generations": 10,
        "population_size": 100,
        'cxpb': 0.5, 'mutpb': 0.5,
        'n_elites': 2, 'tournsize': 3, 'indpb': 0.5
    }
    opt = Optimizer(features=x, labels=y, estimator_class=DecisionTreeClassifier,
                    genetic_params=genetic_params,
                    hyperparam_space=default_hyperparameter_space)
    opt.optimize_clf(**genetic_params)
    return opt


def test_logbook_to_pandas(default_tree_optimizer):
    logbook = default_tree_optimizer.genetic_algorithm.logbook
    df = logbook_to_pandas(logbook)
    assert df is not None


def test_plot_logbook(default_tree_optimizer):
    logbook = default_tree_optimizer.genetic_algorithm.logbook
    fig = plot_logbook(logbook)
    plt.show()
    assert fig is not None


def test_plot_search_space(default_tree_optimizer):
    populations_df = default_tree_optimizer.genetic_algorithm.population_2_df()
    fig = plot_search_space(populations_df)
    plt.show()
    assert fig is not None

def test_plotly_logbook(default_tree_optimizer):
    from pathlib import Path
    logbook = default_tree_optimizer.genetic_algorithm.logbook
    populations_df = default_tree_optimizer.genetic_algorithm.population_2_df()
    fig = plotly_logbook(logbook, populations_df)

    out = Path(__file__).resolve().parent / "logbook.html"
    print(f"Writing report to: {out}")  # muestra en consola la ruta exacta (donde está `test_plots.py`)
    fig.write_html(str(out))

    assert out.exists() and out.stat().st_size > 0

def test_plotly_search_space(default_tree_optimizer):
    from pathlib import Path
    populations_df = default_tree_optimizer.genetic_algorithm.population_2_df()
    fig = plotly_search_space(populations_df)
    out = Path(__file__).resolve().parent / "search_space.html"
    print(f"Writing report to: {out}")  # muestra en consola la ruta exacta (donde está `test_plots.py`)
    fig.write_html(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plotly_search_space_for_docs(default_tree_optimizer):
    from pathlib import Path
    populations_df = default_tree_optimizer.genetic_algorithm.population_2_df()
    out_dir = Path(__file__).resolve().parent

    # Test PNG output (recommended for docs)
    out_png = out_dir / "search_space_docs.png"
    fig = plotly_search_space_for_docs(populations_df, output_path=str(out_png))
    assert fig is not None
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"PNG size: {out_png.stat().st_size / 1024:.1f} KB")

    # Test HTML output with docs optimization
    out_html = out_dir / "search_space_docs.html"
    fig = plotly_search_space_for_docs(populations_df, output_path=str(out_html))
    assert out_html.exists() and out_html.stat().st_size > 0
    html_size_kb = out_html.stat().st_size / 1024
    print(f"HTML size: {html_size_kb:.1f} KB")
    # Docs-optimized HTML should be under 500KB (no embedded ~3MB plotly.js)
    assert html_size_kb < 500, f"HTML too large: {html_size_kb:.1f} KB"
