"""
Contains the Figure class for creating conference and journal article figures.
"""

from matplotlib import pyplot as plt

from thesis.figures import a2020mpc, a2021enc, a2022pso, a2023amm, a2023rsc
from thesis import structures as st

_FIGURES = {
    '2020mpc': {
        2: a2020mpc.fig2,
        3: a2020mpc.fig3,
        4: a2020mpc.fig4,
        5: a2020mpc.fig5,
        6: a2020mpc.fig6,
        7: a2020mpc.fig7,
        8: a2020mpc.fig8,
    },
    '2021enc': {
        1: a2021enc.fig1,
        2: a2021enc.fig2,
        3: a2021enc.fig3,
        4: a2021enc.fig4,
        5: a2021enc.fig5,
        6: a2021enc.fig6,
        7: a2021enc.fig7,
        8: a2021enc.fig8,
        9: a2021enc.fig9,
        10: a2021enc.fig10,
        11: a2021enc.fig11,
        12: a2021enc.fig12,
        14: a2021enc.fig14,
        15: a2021enc.fig15,
        16: a2021enc.fig16,
        17: a2021enc.fig17,
        18: a2021enc.fig18,
        19: a2021enc.fig19,
        20: a2021enc.fig20,
        21: a2021enc.fig21,
        22: a2021enc.fig22,
        23: a2021enc.fig23,
        25: a2021enc.fig25,
        26: a2021enc.fig26,
        27: a2021enc.fig27,
        28: a2021enc.fig28,
        29: a2021enc.fig29,
    },
    '2022pso': {
        3: a2022pso.fig3,
        4: a2022pso.fig4,
    },
    '2023amm': {
        8: a2023amm.fig8,
        9: a2023amm.fig9,
        10: a2023amm.fig10,
        11: a2023amm.fig11,
        12: a2023amm.fig12,
        13: a2023amm.fig13,
        14: a2023amm.fig14,
        15: a2023amm.fig15,
        16: a2023amm.fig16,
        17: a2023amm.fig17,
        18: a2023amm.fig18,
    },
    '2023rsc': {
        3: a2023rsc.fig3,
        4: a2023rsc.fig4,
        5: a2023rsc.fig5,
        6: a2023rsc.fig6,
        7: a2023rsc.fig7,
        8: a2023rsc.fig8,
        9: a2023rsc.fig9,
        10: a2023rsc.fig10,
        11: a2023rsc.fig11,
    },
}


class Figure:
    """
    Creates, plots, shows and saves article figures for my PhD thesis.
    """

    def __init__(self, article: str, figure_number: int, run: bool):
        self.article = article
        self.number = figure_number
        self.run = run
        self._validate_input()

    def plot(self) -> None:
        print(f'\nGenerating Figure {self.number} for {self.article}...\n')
        _FIGURES[self.article][self.number](self.run)

    def save(self) -> None:
        path = st.files.output_path / self.article
        path.mkdir(parents=True, exist_ok=True)
        for suffix in ['png']:  # ['png', 'svg']:
            plt.savefig(path / f'{self.article}_{self.number}.{suffix}')
        print(f'Saved Figure {self.number} for {self.article}.')

    @staticmethod
    def close() -> None:
        plt.close()

    @staticmethod
    def show() -> None:
        plt.show()

    def _validate_input(self) -> None:
        if self.article not in _FIGURES:
            raise ValueError(f"'article' must be one of {list(_FIGURES)}")

        article_figures = list(_FIGURES[self.article])
        if self.number not in article_figures:
            raise ValueError(
                f"'figure_number' must be one of {article_figures}"
            )

        if not isinstance(self.run, bool):
            raise ValueError(f"'run' must be 'True' or 'False'")
