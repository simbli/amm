"""
Contains the show function for displaying PhD thesis figures.
"""

from thesis import figures


def create(article: str, figure_number: int, run: bool = False) -> None:
    """
    Create thesis figures from conference and journal article publications.

    Args:
        article: The article ID.
        figure_number: The desired figure to show.
        run: Set True to run trajectory optimization algorithms, or set False
        to load pre-generated trajectories.

    Returns:
        None
    """
    figure = figures.Figure(article, figure_number, run)
    figure.plot()
    figure.save()
    figure.show()
    figure.close()
