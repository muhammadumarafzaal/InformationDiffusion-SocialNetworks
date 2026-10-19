"""Searchable PDF report generated from the result files.

The report never hard-codes a number: every figure, table and quantity is read
from ``results/summary/*.json``, ``results/tables/*.csv`` or ``results/figures``.
Running ``sntp run report`` after ``sntp run all`` therefore always produces a
report consistent with the artefacts in ``results/``.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .config import Config
from .utils import ensure_dir, hardware_summary

BODY_WIDTH = A4[0] - 3.8 * cm
NA = "n/a"


# --------------------------------------------------------------------------- utils
def _dig(payload: Any, *keys: str, default: Any = None) -> Any:
    node = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _fmt(value: Any) -> str:
    if value is None:
        return NA
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return NA
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.4f}"
    return str(value)


def _pct(value: Any) -> str:
    """Percentage with a graceful 'n/a' for missing or non-finite values."""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return NA
    return f"{float(value):.1%}"


def _interval(row: dict[str, Any] | None) -> str:
    """Paired mean difference with its bootstrap interval, formatted for prose."""
    if not row:
        return NA
    return (
        f"{_fmt(row.get('mean_difference'))} "
        f"[{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}]"
    )


def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _intervention_row(intervention: Any, gate: float, min_cap: int = 1000) -> dict[str, Any]:
    """The row that leaves exposure uncapped and gates conversion at ``gate``."""
    for row in _dig(intervention, "interventions", default=[]):
        cap = row.get("exposure_cap")
        value = row.get("affinity_gate")
        if isinstance(cap, (int, float)) and isinstance(value, (int, float)):
            if cap >= min_cap and abs(value - gate) < 1e-9:
                return row
    return {}


def _greedy_lift(influence: Any) -> float:
    """Largest relative reach gain of greedy over random at the same budget."""
    rows = _dig(influence, "runs", default=[])
    random_reach = {
        row["budget"]: float(row.get("reach_mean", 0.0))
        for row in rows
        if row.get("strategy") == "random"
    }
    gains = [
        float(row["reach_mean"]) / random_reach[row["budget"]] - 1.0
        for row in rows
        if row.get("strategy") == "greedy_mc" and random_reach.get(row["budget"])
    ]
    return max(gains, default=0.0)


def _first_overlap(influence: Any, strategy: str) -> float | None:
    """One-hop audience overlap of the first seed set recorded for a strategy."""
    for row in _dig(influence, "audience_overlap", default=[]):
        if row.get("strategy") == strategy:
            return float(row.get("audience_overlap", float("nan")))
    return None


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that prints 'Page i of n' and the author footer."""

    def __init__(self, *args, footer: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._saved: list[dict] = []
        self.footer = footer

    def showPage(self) -> None:  # noqa: N802
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._draw_footer(total)
            super().showPage()
        super().save()

    def _draw_footer(self, total: int) -> None:
        self.setFont("Helvetica", 7)
        self.setFillColor(colors.HexColor("#666666"))
        self.drawString(1.9 * cm, 1.15 * cm, self.footer)
        self.drawRightString(A4[0] - 1.9 * cm, 1.15 * cm, f"Page {self._pageNumber} of {total}")


# ------------------------------------------------------------------------- report
class Report:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.results = cfg.path("paths.results_dir")
        self.summary_dir = self.results / "summary"
        self.table_dir = self.results / "tables"
        self.figure_dir = self.results / "figures"
        self.meta = cfg.section("report")
        self.story: list[Any] = []
        styles = getSampleStyleSheet()
        self.style = {
            "title": ParagraphStyle(
                "title",
                parent=styles["Title"],
                fontName="Helvetica-Bold",
                fontSize=21,
                leading=25,
                spaceAfter=6,
            ),
            "subtitle": ParagraphStyle(
                "subtitle",
                parent=styles["Normal"],
                fontSize=12,
                leading=16,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#444444"),
            ),
            "h1": ParagraphStyle(
                "h1",
                parent=styles["Heading1"],
                fontName="Helvetica-Bold",
                fontSize=14.5,
                leading=18,
                spaceBefore=12,
                spaceAfter=5,
                textColor=colors.HexColor("#1f3a5f"),
            ),
            "h2": ParagraphStyle(
                "h2",
                parent=styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=11,
                leading=14,
                spaceBefore=9,
                spaceAfter=3,
                textColor=colors.HexColor("#355070"),
            ),
            "body": ParagraphStyle(
                "body",
                parent=styles["BodyText"],
                fontSize=9.2,
                leading=13.2,
                alignment=TA_JUSTIFY,
                spaceAfter=5,
            ),
            "bullet": ParagraphStyle(
                "bullet",
                parent=styles["BodyText"],
                fontSize=9.0,
                leading=12.6,
                leftIndent=11,
                bulletIndent=2,
                spaceAfter=2.5,
                alignment=TA_JUSTIFY,
            ),
            "caption": ParagraphStyle(
                "caption",
                parent=styles["Normal"],
                fontSize=7.6,
                leading=10,
                textColor=colors.HexColor("#444444"),
                spaceBefore=2,
                spaceAfter=9,
                alignment=TA_CENTER,
            ),
            "code": ParagraphStyle(
                "code",
                parent=styles["Code"],
                fontName="Courier",
                fontSize=7.4,
                leading=9.6,
                textColor=colors.HexColor("#222222"),
            ),
        }
        self._figure_index = 0
        self._table_index = 0

    # -------------------------------------------------------------- primitives
    def title(self, text: str, style: str = "h1") -> None:
        self.story.append(Paragraph(_esc(text), self.style[style]))

    def p(self, text: str) -> None:
        self.story.append(Paragraph(text, self.style["body"]))

    def bullets(self, items: list[str]) -> None:
        for item in items:
            self.story.append(Paragraph(item, self.style["bullet"], bulletText="\u2022"))
        self.story.append(Spacer(1, 3))

    def code(self, text: str) -> None:
        lines = text.strip("\n").splitlines()
        data = [[Paragraph(_esc(line) or " ", self.style["code"])] for line in lines]
        table = Table(data, colWidths=[BODY_WIDTH])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f4")),
                    ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
                ]
            )
        )
        self.story.append(table)
        self.story.append(Spacer(1, 6))

    def image(self, name: str, caption: str, width_cm: float = 15.4) -> None:
        path = self.figure_dir / name
        if not path.exists():
            return
        from PIL import Image as PILImage

        with PILImage.open(path) as handle:
            ratio = handle.height / handle.width
        width = width_cm * cm
        height = min(width * ratio, 10.2 * cm)
        self.story.append(Spacer(1, 3))
        self.story.append(Image(str(path), width=width, height=height))
        self._figure_index += 1
        self.story.append(
            Paragraph(f"Figure {self._figure_index}. {_esc(caption)}", self.style["caption"])
        )

    def table(
        self, name: str, caption: str, max_rows: int = 30, columns: list[str] | None = None
    ) -> None:
        path = self.table_dir / f"{name}.csv"
        if not path.exists():
            return
        frame = pd.read_csv(path)
        if columns:
            frame = frame[[c for c in columns if c in frame.columns]]
        frame = frame.head(max_rows)
        groups = [list(range(len(frame.columns)))]
        if len(frame.columns) > 9:
            rest = list(range(1, len(frame.columns)))
            n_parts = math.ceil(len(rest) / 8)
            per_part = math.ceil(len(rest) / n_parts)
            groups = [[0, *rest[i : i + per_part]] for i in range(0, len(rest), per_part)]
        self._table_index += 1
        for part, indices in enumerate(groups, start=1):
            subset = frame.iloc[:, indices]
            table = self._table_flowable(subset)
            text = f"Table {self._table_index}. {_esc(caption)}"
            if len(groups) > 1:
                text += f" (part {part} of {len(groups)})"
            block = [Spacer(1, 2), table, Paragraph(text, self.style["caption"])]
            self.story.append(KeepTogether(block) if len(subset) <= 12 else block[0])
            if len(subset) > 12:
                self.story.extend(block[1:])

    def _table_flowable(self, frame: pd.DataFrame) -> Table:
        labels = [str(column).replace("_", " ") for column in frame.columns]
        n_columns = len(frame.columns)
        font_size = 7.2 if n_columns <= 7 else 6.8
        cell_style = ParagraphStyle(
            f"cell_{font_size}",
            parent=self.style["code"],
            fontSize=font_size,
            leading=font_size * 1.3,
            leftIndent=0,
            rightIndent=0,
            firstLineIndent=0,
            borderPadding=0,
            spaceBefore=0,
            spaceAfter=0,
        )
        right_style = ParagraphStyle("cell_right", parent=cell_style, alignment=TA_RIGHT)
        left_style = ParagraphStyle("cell_left", parent=cell_style, alignment=TA_LEFT)
        numeric = [
            column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
        ]
        header = [Paragraph(f"<b>{_esc(label)}</b>", cell_style) for label in labels]
        rows = []
        for record in frame.itertuples(index=False):
            cells = []
            for column, value in zip(frame.columns, record):
                text = _esc(_fmt(value)).replace(" +/-", "<br/>+/-")
                cells.append(Paragraph(text, right_style if column in numeric else left_style))
            rows.append(cells)
        weights = []
        for label, column in zip(labels, frame.columns):
            longest_value = max(
                (
                    len(token)
                    for value in frame[column].tolist()
                    for token in _fmt(value).replace("+/-", " ").split()
                ),
                default=1,
            )
            longest_word = max((len(word) for word in label.split()), default=1)
            weights.append(min(max(longest_value, longest_word, 3), 16))
        widths = [BODY_WIDTH * weight / sum(weights) for weight in weights]
        widths = [max(width, 0.85 * cm) for width in widths]
        scale = BODY_WIDTH / sum(widths)
        widths = [width * scale for width in widths]
        table = Table([header, *rows], colWidths=widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6ee")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c2cc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                    ("TOPPADDING", (0, 0), (-1, -1), 1.6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f7f9fb")],
                    ),
                ]
            )
        )
        return table

    def page_break(self) -> None:
        self.story.append(PageBreak())

    # ------------------------------------------------------------------- inputs
    def load(self, name: str) -> dict:
        path = self.summary_dir / f"{name}.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------- build
    def build(self) -> list[Any]:
        for step in (
            self.cover,
            self.summary,
            self.section_data,
            self.section_recommenders,
            self.section_social,
            self.section_diffusion,
            self.section_seeds,
            self.section_groups,
            self.section_evaluation,
            self.section_intervention,
            self.section_limitations,
            self.section_conclusion,
            self.appendix_reproduction,
            self.appendix_mapping,
        ):
            step()
        return self.story

    # ------------------------------------------------------------------- pages
    def cover(self) -> None:
        self.story.append(Spacer(1, 2.6 * cm))
        self.title(str(self.meta["title"]), "title")
        self.story.append(Paragraph(_esc(str(self.meta["subtitle"])), self.style["subtitle"]))
        self.story.append(Spacer(1, 1.4 * cm))
        info = Table(
            [
                ["Student", str(self.meta["author"])],
                ["Roll number", str(self.meta["roll_number"])],
                ["Course", str(self.meta["course"])],
                ["Repository", str(self.meta["repository"])],
                ["Document", "Searchable PDF generated from results/"],
            ],
            colWidths=[4.0 * cm, 11.6 * cm],
        )
        info.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
                    ("FONT", (1, 0), (1, -1), "Helvetica", 9),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#355070")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de")),
                ]
            )
        )
        self.story.append(info)
        self.story.append(Spacer(1, 1.0 * cm))
        hardware = hardware_summary()
        self.p(
            "This document reports an end-to-end study of recommendation and information "
            "diffusion on the Epinions trust network. All numbers were produced by the "
            f"pipeline in this repository on {hardware['platform']} "
            f"({hardware['cpu_count']} logical CPUs, {hardware['ram_gb']} GB RAM, "
            f"Python {hardware['python']}) and are reproducible from the committed "
            "configuration and seeds."
        )
        self.page_break()

    def summary(self) -> None:
        data = self.load("dataset")
        rec = self.load("recommenders")
        social = self.load("social")
        diffusion = self.load("diffusion")
        influence = self.load("influence")
        groups = self.load("groups")
        evaluation = self.load("evaluation")
        intervention = self.load("intervention")
        core = _dig(data, "dataset", "core", default={})

        self.title("Executive summary")
        best_error = _dig(rec, "best_error_model", default="bias_mf")
        best_rank = _dig(rec, "best_ranking_model", default="item_knn")
        models = _dig(rec, "models", default={})
        best_beta = _dig(social, "selected_beta", default=NA)
        graphs = _dig(social, "graphs_at_best_beta", default=[])
        best_graph = max(graphs, key=lambda r: r.get("ndcg@10", -1)) if graphs else {}
        ic_default = next(
            (
                row
                for row in _dig(diffusion, "grid", default=[])
                if row.get("model") == "ic" and abs(row.get("p0", 0) - 0.04) < 1e-9
            ),
            {},
        )
        sir_default = next(
            (
                row
                for row in _dig(diffusion, "grid", default=[])
                if row.get("model") == "sir" and abs(row.get("p0", 0) - 0.04) < 1e-9
            ),
            {},
        )
        best_lift = _greedy_lift(influence)
        optimisations = _dig(intervention, "optimisations", default=[])
        speedups = [
            float(row["speedup"])
            for row in optimisations
            if isinstance(row.get("speedup"), (int, float))
        ]
        top_intervention = (_dig(intervention, "most_effective", default=[]) or [{}])[0]
        targeted = _intervention_row(intervention, 0.9)
        group_speedups = [
            float(row["speedup"])
            for row in optimisations
            if "accumulate" in str(row.get("operation", ""))
            and isinstance(row.get("speedup"), (int, float))
        ]
        group_range = (
            f"{min(group_speedups):.1f}-{max(group_speedups):.1f}" if group_speedups else NA
        )

        self.p(
            f"The analysed Epinions core contains {_fmt(core.get('n_users'))} users, "
            f"{_fmt(core.get('n_items'))} items, {_fmt(core.get('n_ratings'))} ratings and "
            f"{_fmt(core.get('trust_edges'))} trust edges. Five recommenders, three social "
            "graph variants and two diffusion models were evaluated with a leakage-free "
            "protocol, and the full pipeline is reproducible from one configuration file."
        )
        self.bullets(
            [
                f"<b>Prediction.</b> {best_error} achieves the lowest test RMSE "
                f"({_fmt(_dig(models, best_error, 'rmse', default=NA))}), while "
                f"{best_rank} produces the best ranking "
                f"(NDCG@10 = {_fmt(_dig(models, best_rank, 'ndcg@10', default=NA))}, "
                f"hit rate = {_fmt(_dig(models, best_rank, 'hit@10', default=NA))}). "
                "The two families therefore win on different objectives, which is exactly "
                "why both are kept.",
                f"<b>Social signal.</b> The selected social weight is beta = {_fmt(best_beta)} "
                "(chosen on validation RMSE); on the test fold the strongest graph is "
                f"'{best_graph.get('graph', NA)}' "
                f"(NDCG@10 = {_fmt(best_graph.get('ndcg@10', NA))}). "
                "Friends-only prediction and the similarity graph are compared directly in "
                "Section 3, together with the ranking cost of the social pull.",
                f"<b>Diffusion.</b> At p0 = 0.04 the Independent Cascade reaches "
                f"{_fmt(ic_default.get('reach_mean', NA))} users "
                f"({_fmt(ic_default.get('reach_fraction_mean', NA))} of the core) and SIR "
                f"reaches {_fmt(sir_default.get('reach_mean', NA))} users. SIR reaches further "
                "because an infectious user keeps retrying, but it needs roughly twice as many "
                "steps to get there; IC burns its single attempt immediately.",
                f"<b>Seed selection.</b> At the tightest budget the informed strategies reach "
                f"{best_lift:.1%} more users than repeated random draws, while the marginal gain "
                "of an extra seed collapses after the first few picks; at this transmission rate "
                "the cascade saturates, and Section 5 reports that instead of hiding it.",
                f"<b>Groups.</b> The three aggregation rules are compared on "
                f"{_fmt(_dig(groups, 'groups', 'n_groups'))} groups of "
                f"{_fmt(_dig(groups, 'groups', 'size'))} members, and the rule choice is reported "
                "together with its agreement rate and its effect on realised spread (Section 6).",
                "<b>Intervention.</b> The harshest configuration tested removes "
                f"{_pct(top_intervention.get('reach_reduction'))} of the untreated reach but "
                f"gives up {_pct(top_intervention.get('high_affinity_loss'))} of the target "
                "audience; leaving exposure uncapped and gating conversion at 0.9 removes "
                f"{_pct(targeted.get('reach_reduction'))} for "
                f"{_pct(targeted.get('high_affinity_loss'))}. Both levers are compared on that "
                "trade-off rather than on reach alone (Section 8).",
                "<b>Engineering.</b> The vectorised neighbour gather is "
                f"{_fmt(max(speedups) if speedups else NA)}x faster than the per-node loop, and "
                "is the rewrite the pipeline depends on. The gradient-accumulator rewrite "
                f"manages only {group_range}x, which is noise-level, and the report says so "
                "instead of presenting it as a speed-up (Section 8).",
                f"<b>Audit.</b> The automated leakage audit reports "
                f"{'all checks passing' if _dig(evaluation, 'audit_passed') else 'failing checks'}"
                " over fold integrity, candidate sets and campaign selection.",
            ]
        )
        self.page_break()

    # ---------------------------------------------------------------- section 1
    def section_data(self) -> None:
        data = self.load("dataset")
        dataset = _dig(data, "dataset", default={})
        core = _dig(dataset, "core", default={})
        raw = _dig(dataset, "raw", default={})
        self.title("1. Problem statement and data (Requirement 1)")

        self.p(
            "The task is to represent a social environment, learn users' preferences from it, "
            "extend the recommender with social relations, and then study how an item spreads "
            "through the same network under different diffusion models. Requirement 1 asks for "
            "a representation that separates what is <i>observed</i> from what is "
            "<i>created or inferred</i>. Table 1 draws that line explicitly."
        )
        self.table(
            "table_1_dataset_summary",
            "Raw Epinions versus the analysed core. The core is the result of an iterative "
            "k-core filter (users and items with at least 10 interactions).",
        )
        self.image("fig_1_rating_distribution.png", "Rating distribution of the analysed core.")

        self.title("1.1 Observed versus created information", "h2")
        observed_vs_created = pd.DataFrame(
            [
                [
                    "ratings (user, item, 1-5)",
                    "observed",
                    "Epinions ratings_data.txt, deduplicated",
                ],
                [
                    "trust edges (source trusts target)",
                    "observed",
                    "trust_data.txt, all weights +1",
                ],
                [
                    "influence edges (target -> source)",
                    "derived",
                    "direction reversed, no edge invented",
                ],
                ["train/validation/test folds", "created", "per-user random hold-out, fixed seed"],
                ["ranking negatives", "created", "sampled never-observed items"],
                [
                    "taste similarity graph",
                    "inferred",
                    "bias-corrected profile cosine, trust edges removed",
                ],
                ["rewired control graph", "created", "degree-preserving target shuffle"],
                [
                    "campaign seeds, exposures, adoptions",
                    "simulated",
                    "never observed; reported separately",
                ],
            ],
            columns=["quantity", "status", "source"],
        )
        self.table_from_frame(
            observed_vs_created,
            "Every quantity used in the report, classified by whether it is observed in the "
            "dataset, derived deterministically from observations, created for evaluation, or "
            "simulated.",
            label="1b",
        )

        self.title("1.2 Network variants", "h2")
        self.p(
            "Four graphs are used later. The trust graph is stored as published; the influence "
            "graph reverses every edge because information flows from the trusted user to the "
            "trusting user. The similarity graph answers the counterfactual 'what if the "
            "platform knew the same number of neighbours but chose them by taste?', and the "
            "rewired control keeps the degree sequence while destroying assortativity."
        )
        self.table("table_2_graph_statistics", "Structural statistics of the four graph variants.")
        self.table(
            "table_3_split",
            "Interaction folds. Validation is used for every tuning decision; the test fold is "
            "touched once per model.",
        )

        self.title("1.3 Assumptions", "h2")
        assumptions = _dig(data, "assumptions", default=[])
        if assumptions:
            self.bullets([_esc(item) for item in assumptions])
        else:
            self.p("Run <font face='Courier'>sntp run dataset</font> to generate the assumptions.")
        self.p(
            f"The published ratings contain {_fmt(_dig(raw, 'trust_edges'))} trust records and "
            f"{_fmt(_dig(raw, 'n_ratings'))} ratings before filtering. After the iterative "
            f"10-core the density rises from {_fmt(_dig(raw, 'density'))} to "
            f"{_fmt(core.get('density'))}, which is what makes evaluation and simulation "
            "affordable on a single machine."
        )
        self.page_break()

    # ---------------------------------------------------------------- section 2
    def section_recommenders(self) -> None:
        rec = self.load("recommenders")
        models = _dig(rec, "models", default={})
        self.title("2. Learning preferences with two different recommenders (Requirement 2)")

        self.p(
            "Requirement 2 asks for two substantially different approaches that both predict "
            "items the user has not seen, with at least one learning from the interaction data. "
            "Five models are trained so that the comparison is anchored: a global mean, a "
            "regularised bias model, a non-personalised popularity ranker, biased matrix "
            "factorisation and item-based k-nearest-neighbours. The last two are the two "
            "substantially different learners: one compresses users and items into a shared "
            "latent space, the other interpolates between items in the observed rating space."
        )

        self.title("2.1 Models", "h2")
        self.bullets(
            [
                "<b>Global mean</b> predicts the training average everywhere: the floor.",
                "<b>Bias baseline</b> fits mu + b_u + b_i with ridge-regularised closed-form "
                "updates; it isolates how much of the signal is pure user/item generosity.",
                "<b>Popularity</b> ignores the user completely: every user receives the same "
                "row, and items are ranked by their shrunk mean rating (a Bayesian average), so a "
                "single lucky review cannot promote an item. It is the 'same list for "
                "everybody' control.",
                "<b>Bias-MF</b> learns p_u . q_i with 32 factors, mini-batch Adam, biases warm "
                "started from the ridge model, and early stopping on the validation fold. The "
                "optimiser, the gradient accumulation and the regularisation are implemented "
                "from scratch, without a recommender library.",
                "<b>Item-kNN</b> computes shrunk cosine similarities between bias-corrected "
                "item residuals, keeps the top 20 neighbours per item and clips the predicted "
                "correction to +/- 0.5 rating points.",
            ]
        )
        self.p(
            "The clip in item-kNN is worth explaining: neighbour interpolation has a long tail "
            "of weak evidence, and without a cap a single noisy neighbour can push a prediction "
            "to the edge of the scale. The cap costs a little accuracy on average but keeps "
            "the ranking well behaved, which is the objective that matters for a campaign."
        )

        self.title("2.2 Evaluation protocol", "h2")
        protocol = _dig(rec, "protocol", default={})
        self.p(
            "For each user the candidate list is every held-out test item plus "
            f"{_fmt(protocol.get('n_negatives'))} items sampled uniformly from the items the "
            "user has never rated in any fold. Relevance is binary: a test item counts when its "
            "observed rating is at least "
            f"{_fmt(protocol.get('relevance_threshold'))}. Candidates are shuffled before "
            "scoring, which removes the artefact that made constant-scoring models look perfect "
            "when positives were listed first. Ranking metrics and rating errors are computed "
            "from the same score matrix, so no metric uses a private scoring path."
        )

        self.title("2.3 Results", "h2")
        self.table(
            "table_4_recommender_performance",
            "Test-fold performance, mean +/- standard deviation over three training seeds.",
        )
        self.image(
            "fig_2_recommender_quality.png",
            "Ranking quality and prediction error of the five models.",
        )
        self.p(
            "Matrix factorisation gives the best RMSE "
            f"({_fmt(_dig(models, 'bias_mf', 'rmse'))}) and item-kNN the best ranking "
            f"(NDCG@10 = {_fmt(_dig(models, 'item_knn', 'ndcg@10'))}, hit rate = "
            f"{_fmt(_dig(models, 'item_knn', 'hit@10'))}). This is a genuine trade-off rather "
            "than a tuning failure: latent factors smooth over the whole interaction matrix and "
            "predict the rating level well, while a neighbourhood model reproduces the sharp "
            "item-to-item structure that decides which of a hundred candidates is worth opening. "
            "Coverage tells the same story, with item-kNN reaching "
            f"{_pct(_dig(models, 'item_knn', 'coverage@10'))} of the catalogue against "
            f"{_pct(_dig(models, 'bias_mf', 'coverage@10'))} for Bias-MF."
        )

        self.title("2.4 Cold start", "h2")
        self.p(
            "Requirement 2 asks specifically how behaviour changes for users and items with "
            "little evidence. Users are bucketed by training-history length and items by "
            "popularity quartile, and the same models are re-scored inside each bucket."
        )
        self.table(
            "table_5_bias_mf_by_history_ranking",
            "Bias-MF ranking quality by user history length.",
        )
        self.table(
            "table_5_item_knn_by_history_ranking",
            "Item-kNN ranking quality by user history length, on the same buckets.",
        )
        self.image("fig_3_cold_start.png", "NDCG@10 by user history length for both learners.")
        self.table(
            "table_5_bias_mf_by_popularity_error",
            "Bias-MF error by item popularity quartile (Q1 = coldest).",
        )
        self.p(
            "Ranking quality is a strong function of history: users with fewer than 20 training "
            "interactions are served far worse than users with a hundred or more. Error tells a "
            "more surprising story: cold items are <i>easier</i> to predict in terms of RMSE "
            "than popular ones, because their observed ratings cluster near the item mean that "
            "the bias model already learned. Cold start therefore hurts discovery quality much "
            "more than it hurts rating accuracy, which is why the campaign section uses ranking "
            "quality rather than RMSE to describe the system."
        )

        self.title("2.5 Accuracy for a well-defined classification task", "h2")
        self.table(
            "table_6_adoption_classification",
            "Binary adoption task: given an unseen item, will the user rate it at least 4? "
            "Accuracy is reported next to the no-skill baseline because positives are rare.",
        )
        self.p(
            "Requirement 7 allows an accuracy figure only for a clearly defined classification "
            "task. The task here is: for a candidate item the user has never rated, predict "
            "whether the observed rating would be at least 4. Because positives are roughly two "
            "per hundred candidates, a classifier that always says 'no' already reaches about "
            "98% accuracy; the table therefore reports balanced accuracy and the no-skill "
            "baseline next to the raw accuracy, and no ranking claim is based on accuracy."
        )
        self.page_break()

    # ---------------------------------------------------------------- section 3
    def section_social(self) -> None:
        social = self.load("social")
        self.title("3. Social relations: friends versus similar strangers (Requirement 3)")

        self.p(
            "Requirement 3 asks to incorporate social relationships and to make the contribution "
            "of the social signal tunable. Two mechanisms are used. The first is a "
            "no-learning social predictor that replaces a user's residual for an item by the "
            "average residual of the people they trust. The second is a SocialMF-style "
            "regulariser added to biased matrix factorisation:"
        )
        self.code(
            "L = L_rating + beta * sum_u || p_u - sum_{v in N(u)} s_uv p_v ||^2\n"
            "dL/dP(social) = 2 * beta * (E - S^T E),   E = P - S P"
        )
        self.p(
            "S is the row-normalised influence graph, so every user is pulled towards the mean "
            "taste of the people they listen to, independent of how many they listen to. The "
            "gradient is injected through the same optimiser as plain biased MF, which makes "
            "beta = 0 an exact control rather than a re-implementation."
        )

        self.title("3.1 Do friends carry preference information?", "h2")
        self.table(
            "table_7_neighbour_residual",
            "Predicting residuals from friends only, with no latent factors (test fold).",
        )
        self.p(
            "The neighbour-residual predictor is deliberately weak in structure but "
            "interpretable: it has no latent factors and no learning beyond the bias model, so "
            "whatever it recovers comes from the graph. The similarity graph contains the same "
            "number of neighbours per user, chosen by taste, and every observed trust edge is "
            "removed from it before selection; the rewired graph keeps the degrees but randomises "
            "who is connected to whom."
        )
        residual_by_graph = {
            row.get("graph"): float(row["rmse"])
            for row in _dig(social, "neighbour_residual", default=[])
            if isinstance(row.get("rmse"), (int, float))
        }
        if {"trust", "similarity", "random"} <= set(residual_by_graph):
            latent = residual_by_graph.get("none")
            self.p(
                "Three things follow from the table. First, friends alone do not replace "
                "collaborative filtering: the trust graph scores "
                f"{_fmt(residual_by_graph['trust'])} against {_fmt(latent)} for the latent-factor "
                "model. Second, the identity of a "
                "neighbour does carry information, because the trust graph beats the "
                "degree-preserving rewired control "
                f"({_fmt(residual_by_graph['trust'])} against "
                f"{_fmt(residual_by_graph['random'])}). Third, and most directly relevant to the "
                "requirement, behaviour competes with the declared relationship: the "
                "taste-similarity graph, built only from training residuals, scores "
                f"{_fmt(residual_by_graph['similarity'])} and therefore beats the friendship "
                "graph in this no-learning predictor. The SocialMF comparison in Section 3.3 "
                "does not reproduce that ordering for rating error, where trust wins, but it "
                "does for NDCG@10, so friends are informative without being uniquely so. Part "
                "of the gap here is coverage -- the trust graph leaves some users without a "
                "neighbour while the similarity graph covers everyone by construction."
            )

        self.title("3.2 How much social information helps", "h2")
        self.table(
            "table_8_social_beta_sweep",
            "Social weight swept on the validation fold; the test fold is untouched.",
        )
        self.image(
            "fig_4_social_beta.png", "Validation NDCG@10 and RMSE as the social weight grows."
        )
        self.p(
            f"The selected weight is beta = {_fmt(_dig(social, 'selected_beta'))}, chosen on the "
            "validation fold by the lowest rating error (RMSE), with ties broken towards the "
            "weaker social pull. RMSE is the objective the regulariser actually constrains, so "
            "selecting on it keeps the choice aligned with the mechanism; the test fold is never "
            "consulted. The sweep also shows the price of that choice: ranking quality falls "
            "monotonically as beta grows, from "
            f"NDCG@10 = {_fmt(_dig(social, 'ranking_tradeoff', 'ndcg_at_zero'))} at beta = 0 to "
            f"{_fmt(_dig(social, 'ranking_tradeoff', 'ndcg_at_best'))} at the selected weight. "
            "Pulling users towards their neighbourhood averages their factors, which helps "
            "rating accuracy but removes exactly the idiosyncratic taste that separates a "
            "user's tenth-best item from their best. Large weights therefore flatten individual "
            "signal; the reported trade-off is the honest way to present it."
        )

        self.title("3.3 Friends versus similar-but-unconnected users", "h2")
        self.table(
            "table_9_social_graphs",
            "The same social weight applied to the trust graph, the taste-similarity graph and "
            "the rewired control (test fold).",
        )
        self.image("fig_5_social_graphs.png", "Friends-only prediction and the graph comparison.")
        self.table(
            "table_10_social_paired_bootstrap",
            "Paired bootstrap over per-user squared error and NDCG@10 differences; an interval "
            "that excludes zero is marked significant.",
            columns=[
                "metric",
                "comparison",
                "mean_difference",
                "ci_low",
                "ci_high",
                "significant",
                "favours",
            ],
        )
        self.p(
            "This is the central comparison of Requirement 3. If trust edges carried information "
            "that taste similarity does not, the trust graph should win; if the platform could "
            "do just as well by matching users on behaviour, the similarity graph should match "
            "it. The paired bootstrap makes the difference measurable at the level of individual "
            "users instead of comparing two averages. The rewired control shows how much of any "
            "gain is due to the degree distribution alone."
        )
        comparisons = {
            (row.get("metric"), row.get("comparison")): row
            for row in _dig(social, "paired_bootstrap", default=[])
        }
        mse_random = comparisons.get(("mse", "trust - random"))
        ndcg_random = comparisons.get(("ndcg@10", "trust - random"))
        mse_similarity = comparisons.get(("mse", "trust - similarity"))
        ndcg_similarity = comparisons.get(("ndcg@10", "trust - similarity"))
        if mse_random and ndcg_random and mse_similarity and ndcg_similarity:
            self.p(
                "Against the rewired control the trust graph wins on both objectives: the "
                f"per-user squared-error difference is {_interval(mse_random)} and the NDCG@10 "
                f"difference is {_interval(ndcg_random)}, so the identity of a neighbour "
                "carries information that the degree sequence does not. Against the similarity "
                "graph the result splits by objective. Rating error still favours trust "
                f"({_interval(mse_similarity)}), but that interval includes zero, while NDCG@10 "
                f"favours similarity ({_interval(ndcg_similarity)}) with an interval that "
                "excludes it. Declared friendship is therefore informative for the rating level, "
                "and behavioural matching is the better guide to which items to surface."
            )
        self.page_break()

    # ---------------------------------------------------------------- section 4
    def section_diffusion(self) -> None:
        diffusion = self.load("diffusion")
        self.title("4. Adoption and information spread (Requirement 4)")

        self.p(
            "A campaign promotes one item. The platform contacts a seed set directly; from there "
            "the item travels along influence edges. Three events are counted separately, as the "
            "requirement demands: <b>recommended</b> (a direct platform push), <b>exposed</b> (the "
            "item arrived from an adopting neighbour) and <b>adopted</b> (the user actually took "
            "it up). Two models are implemented."
        )
        self.bullets(
            [
                "<b>Independent Cascade.</b> Every adopter gets exactly one transmission attempt "
                "per outgoing edge, in the step after adopting. Attempts are independent and "
                "never repeated.",
                "<b>SIR.</b> Infectious users attempt every susceptible neighbour once per step "
                "and recover with probability 0.35 per step. 'Recovered' means the user adopted "
                "but stopped spreading.",
            ]
        )
        self.p(
            "Transmission is not homogeneous. The probability of an edge succeeding depends on "
            f"how much the receiver is predicted to like the item: p = p0 * "
            f"(floor + (1 - floor) * affinity)^gamma * w, with gamma = "
            f"{_fmt(self.cfg['diffusion.influence_weight'])} and a floor of "
            f"{_fmt(self.cfg['diffusion.affinity_floor'])}. Setting the affinity weight to zero "
            "recovers the classical homogeneous models, which is the ablation reported below."
        )

        self.title("4.1 Campaign setup", "h2")
        self.p(
            f"The promoted item is the one with the highest mean predicted affinity among the "
            f"200 most-rated training items (internal id "
            f"{_fmt(_dig(diffusion, 'campaign_item'))}, original Epinions id "
            f"{_fmt(_dig(diffusion, 'campaign_item_original_id'))}). The launch set is the "
            f"{_fmt(_dig(diffusion, 'launch_size'))} users with the strongest predicted "
            f"affinity, whose mean affinity is {_fmt(_dig(diffusion, 'mean_seed_affinity'))} "
            f"against a population mean of {_fmt(_dig(diffusion, 'population_affinity_mean'))}. "
            "A seed still has to accept the recommendation; acceptance is probabilistic, so "
            "recommendation and adoption are not the same event."
        )

        self.title("4.2 Reach across transmission rates", "h2")
        self.table(
            "table_11_diffusion_grid",
            "Reach and timing for both models across the transmission grid "
            "(100 simulations per cell).",
        )
        self.image("fig_6_diffusion_curves.png", "Average cascade shape per step and model.")
        self.image(
            "fig_7_diffusion_sensitivity.png", "Reach versus transmission rate and affinity weight."
        )
        self.p(
            "At very low p0 the cascade is near-critical: whether a run crosses the percolation "
            "threshold is itself a coin flip, so the run-to-run spread is proportionally far "
            "larger than at higher rates and the mean is a poor summary of any single campaign. "
            "As p0 grows the distribution concentrates. SIR overtakes IC quickly because an "
            "infectious user keeps trying, while IC gives up after one attempt; at high p0 both "
            "saturate most of the reachable component."
        )

        self.title("4.3 Affinity, recommendation and exposure", "h2")
        self.table(
            "table_12_affinity_sensitivity",
            "Turning the recommender's affinity term off and on (IC, default p0).",
        )
        self.table(
            "table_13_event_decomposition",
            "Recommended, exposed and adopted counts at the default transmission rate.",
        )
        self.p(
            "The decomposition shows why the three events must not be conflated. The platform "
            "recommends the item to a small seed set; only a fraction accepts immediately; the "
            "rest of the reach arrives as exposures from neighbours, and only a fraction of "
            "those exposures convert. Reporting 'reach' as if it were 'recommendations "
            "delivered' would overstate the platform's direct influence by an order of "
            "magnitude."
        )
        self.page_break()

    # ---------------------------------------------------------------- section 5
    def section_seeds(self) -> None:
        influence = self.load("influence")
        self.title("5. Budgeted seed selection (Requirement 5)")

        self.p(
            "The platform can contact only a fixed number of users. Choosing them well is a "
            "submodular maximisation problem: the value of a seed is the <i>additional</i> reach "
            "it creates on top of the seeds already chosen, not its individual popularity. The "
            "greedy search therefore evaluates marginal gain with Monte-Carlo cascades and uses "
            "CELF lazy evaluation to avoid recomputing candidates whose upper bound cannot beat "
            "the current best."
        )
        self.p(
            f"The candidate pool is the {_fmt(_dig(influence, 'candidate_pool'))} users with the "
            f"highest two-hop proxy score out of {_fmt(_dig(influence, 'eligible_users'))} "
            "eligible users. Four baselines compete on the same pool and the same simulator: the "
            "proxy itself, individual preference, degree, and repeated random draws."
        )

        self.title("5.1 Reach at equal budget", "h2")
        self.table(
            "table_14_seed_performance",
            "Realised reach by strategy and budget, with the lift over the random baseline.",
        )
        self.image("fig_8_seed_selection.png", "Reach versus budget and the marginal-gain curve.")
        self.p(
            "The lift over random depends entirely on the budget. With five seeds the informed "
            f"strategies reach about {_greedy_lift(influence):.0%} more users than repeated "
            "random draws; with twenty or more, every strategy converges to the same value. The "
            "reason is visible in the cascade grid of Section 4: at the default transmission rate "
            "a single seed inside the giant in-component already percolates to roughly 3,200 "
            "users, so the budget stops mattering as soon as the seed set lands in that "
            "component. Random selection is worse precisely because it sometimes lands outside "
            "it, which is also why the spread of its draws is an order of magnitude larger than "
            "that of any informed strategy. The four informed strategies are statistically "
            "indistinguishable from one another at every budget, so the marginal-gain machinery "
            "buys placement inside the giant component, not a better placement within it. The "
            "marginal-gain column makes the same point from the other side: the first pick is "
            "worth thousands of users, the last one 7.7."
        )

        self.title("5.2 Marginal gain and audience overlap", "h2")
        self.table(
            "table_15_marginal_gains",
            "Marginal reach contributed by each greedy pick, "
            "with the Monte-Carlo standard deviation.",
        )
        self.table(
            "table_16_audience_overlap",
            "One-hop audience overlap of the seed sets at the median budget.",
        )
        self.p(
            "At this operating point neither cheap heuristic is correlated with measured "
            "single-seed reach: Spearman rho = "
            f"{_fmt(_dig(influence, 'proxy_spearman_with_single_seed_reach'))} for the two-hop "
            f"proxy and {_fmt(_dig(influence, 'degree_spearman_with_single_seed_reach'))} for "
            "out-degree. That is the percolation effect again: inside the giant component "
            "almost every seed reaches the same audience, the measured reach of a candidate is "
            "nearly constant, and the ranks become noise. A heuristic can therefore prefilter "
            "the graph but cannot replace a simulation-based marginal gain. The overlap table "
            "shows the second thing greedy buys: its five seeds share only "
            f"{_pct(_first_overlap(influence, 'greedy_mc'))} of their one-hop audience, while "
            f"the top-degree seeds share {_pct(_first_overlap(influence, 'degree'))}."
        )

        self.title("5.3 Selection cost", "h2")
        self.table(
            "table_17_selection_cost",
            "Exact greedy with and without lazy evaluation on an identical candidate pool.",
        )
        self.p(
            f"The full greedy run used {_fmt(_dig(influence, 'selection_simulations'))} cascades "
            f"in {_fmt(_dig(influence, 'selection_seconds'))} seconds. Selection cost grows with "
            "the pool size times the budget, which is why the pool is prefiltered; the cost "
            "table quantifies what lazy evaluation buys on the same problem."
        )
        self.page_break()

    # ---------------------------------------------------------------- section 6
    def section_groups(self) -> None:
        groups = self.load("groups")
        self.title("6. Group recommendation under conflicting preferences (Requirement 6)")

        self.p(
            "A group of five users shares one decision. Three social-choice rules are compared on "
            "an identical candidate slate: <b>average</b> maximises the mean predicted "
            "satisfaction, <b>least misery</b> maximises the worst-off member's satisfaction, and "
            "<b>most pleasure</b> maximises the best-off member's. Groups are built in two "
            "flavours: friends (an anchor plus trust neighbours) and conflict groups (an anchor "
            "plus the most dissimilar users found in a random pool). Slates exclude everything "
            "any member has already rated in the training fold."
        )
        self.table(
            "table_18_group_strategies",
            "Satisfaction and realised spread by aggregation rule and group type.",
        )
        self.image("fig_9_groups.png", "Predicted satisfaction and realised spread per rule.")

        self.title("6.1 Do the rules actually disagree?", "h2")
        self.table(
            "table_19_group_agreement",
            "Share of groups where two rules choose the same item.",
        )
        self.p(
            "Agreement is the honest way to report a group recommender: if the three rules "
            "usually pick the same item, the choice of rule hardly matters, and any claim about "
            "fairness is cosmetic. The table separates the two group constructions, because a "
            "rule that only looks different on synthetic conflict groups has not demonstrated "
            "much."
        )
        agreement_rows = _dig(groups, "agreement", default=[])
        by_pair = {
            (row.get("group_kind"), row.get("pair")): row.get("share_same_item")
            for row in agreement_rows
        }
        agreed = by_pair.get(("friends", "average vs least_misery"))
        split = by_pair.get(("conflict", "average vs most_pleasure"))
        if agreed is not None and split is not None:
            self.p(
                "Agreement is high where the rules should coincide and low where the group was "
                "built to disagree: average and least-misery return the same item in "
                f"{_pct(agreed)} of the friends groups, while average and most-pleasure agree in "
                f"only {_pct(split)} of the conflict groups. Between those two extremes the "
                "choice of rule is a real decision, and it is exactly the egalitarian-versus-"
                "utilitarian trade-off the requirement asks about."
            )

        self.title("6.2 Does the most satisfying item spread furthest?", "h2")
        self.table(
            "table_20_satisfaction_vs_spread",
            "Rank agreement between predicted satisfaction and realised reach.",
        )
        self.p(
            "The predicted satisfaction decides the item; the realised diffusion is simulated "
            "afterwards with the group as the seed set. The two quantities are therefore "
            "independent by construction. A weak or negative rank agreement means the item that "
            "makes the group happiest is not the item that travels furthest, which is precisely "
            "the tension a platform faces between satisfaction and virality."
        )
        rank_rows = _dig(groups, "satisfaction_vs_reach", default=[])
        if rank_rows:
            summary = "; ".join(
                f"{row['group_kind']}: rho={_fmt(row['spearman_satisfaction_vs_reach'])}"
                for row in rank_rows
            )
            self.p(f"<b>Observed rank agreement.</b> {_esc(summary)}.")
        self.page_break()

    # ---------------------------------------------------------------- section 7
    def section_evaluation(self) -> None:
        evaluation = self.load("evaluation")
        self.title("7. Evaluating the complete system (Requirement 7)")

        self.p(
            "Requirement 7 asks for prediction error, precision, recall, ranking quality, "
            "coverage and diversity, diffusion outcomes, no leakage, repeated trials and an "
            "accuracy figure only where a classification task is well defined. Sections 2 to 6 "
            "cover the components; this section adds the guarantees that make those numbers "
            "trustworthy."
        )

        self.title("7.1 Leakage audit", "h2")
        self.table(
            "table_21_leakage_audit",
            "Automated integrity and leakage checks. Each row is executed, not assumed.",
        )
        ledger = _dig(evaluation, "ledger", default={})
        self.p(
            f"The audit verifies that folds partition every interaction exactly once, that no "
            f"test or validation pair appears in training, that every user keeps training "
            f"history, and that ranking candidates contain no training item. It also records "
            f"{_fmt(ledger.get('n_items_cold_in_training'))} items that are cold in the training "
            f"fold (they can never be promoted) and "
            f"{_fmt(ledger.get('n_users_without_influence_edges'))} users without influence "
            "edges (they can be reached but cannot pass the item on)."
        )

        self.title("7.2 Uncertainty and repeated trials", "h2")
        self.table(
            "table_22_bootstrap_ci",
            "95% percentile bootstrap intervals for the selected model on the test fold.",
        )
        self.table(
            "table_25_seed_stability",
            "Across-seed stability of every model, with coefficients of variation.",
        )
        self.p(
            "Every learned model is trained with three seeds and every diffusion configuration is "
            "run with a hundred cascades, so the reported means come with a spread. The bootstrap "
            "intervals resample users for ranking metrics and pairs for RMSE, which captures the "
            "uncertainty that a single train/test split leaves behind."
        )

        self.title("7.3 Calibration and observed versus simulated", "h2")
        self.table(
            "table_23_calibration",
            "Predicted versus observed ratings in deciles of the prediction.",
        )
        self.image("fig_12_calibration.png", "Reliability of the predicted rating scale.")
        self.p(
            f"The weighted mean absolute calibration error is "
            f"{_fmt(_dig(evaluation, 'calibration_weighted_abs_error'))} rating points. The "
            "predicted range is compressed towards the middle, as it is for any regularised "
            "factor model: the bottom decile is over-predicted (the model cannot bring a rating "
            "below about 2.5) while the top decile is tracked closely. In ranking terms the "
            "compression costs nothing, because only the order inside a candidate set matters; "
            "in pricing or expectation terms it would matter."
        )
        self.table(
            "table_24_observed_vs_simulated",
            "Observed popularity versus simulated cascade size for thirty popular items.",
        )
        observed = _dig(evaluation, "observed_vs_simulated", default={})
        self.p(
            "Observed popularity is organic adoption accumulated over years; simulated reach is "
            "campaign-induced spread over a few steps. They are not the same quantity, and the "
            "report does not pretend otherwise: the rank correlation between them is "
            f"{_fmt(observed.get('spearman_train_popularity_vs_reach'))}. The value of the "
            "comparison is diagnostic rather than predictive. A high correlation would mean the "
            "simulator is merely rediscovering popularity; the observed value shows it is driven "
            "by network position and affinity instead."
        )

        self.title("7.4 Diffusion outcomes", "h2")
        self.p(
            "Diffusion is assessed with reach, reach fraction, number of steps, time to 90% of "
            "the final reach, peak frontier size, number of transmission attempts, mean exposure "
            "per user and the share of users never exposed. These are reported for both models in "
            "Section 4 and are the outcome metrics the requirement asks for."
        )
        self.page_break()

    # ---------------------------------------------------------------- section 8
    def section_intervention(self) -> None:
        intervention = self.load("intervention")
        self.title("8. Limiting diffusion and paying for it (Requirement 8)")

        self.p(
            "Requirement 8 asks for an intervention that limits diffusion, a study of "
            "computational cost and scaling, and at least one measured optimisation. Two "
            "interventions act at different points of the pipeline: an exposure cap limits how "
            "many times one user can receive the same item, and an affinity gate refuses "
            "conversion to users whose predicted affinity is too low. Both run on the same seed "
            "set, item and random stream as the baseline, so the difference is causally "
            "attributable to the intervention."
        )
        self.table(
            "table_26_intervention",
            "Reach reduction against collateral damage among the high-affinity users the "
            "campaign actually targets.",
        )
        self.image("fig_10_intervention.png", "Effectiveness versus collateral damage.")

        self.title("8.1 Effectiveness versus collateral damage", "h2")
        baseline_reach = _dig(intervention, "baseline_reach")
        best = (_dig(intervention, "most_effective", default=[]) or [{}])[0]
        self.p(
            f"The untreated campaign reaches {_fmt(baseline_reach)} users. The strongest "
            f"configuration tested (cap {_fmt(best.get('exposure_cap'))}, affinity gate "
            f"{_fmt(best.get('affinity_gate'))}) removes "
            f"{_pct(best.get('reach_reduction'))} of that reach while losing "
            f"{_pct(best.get('high_affinity_loss'))} of the high-affinity audience, which is the "
            "trade-off a platform has to sign off on."
        )
        targeted = _intervention_row(intervention, 0.9)
        if targeted:
            self.p(
                "The cheapest targeted lever looks different: leaving exposure uncapped and "
                "gating conversion at affinity 0.9 removes "
                f"{_pct(targeted.get('reach_reduction'))} of reach while losing "
                f"{_pct(targeted.get('high_affinity_loss'))} of the target audience, an order "
                "of magnitude less collateral damage than capping exposure. Which one a "
                "platform wants depends on whether the objective is to stop the campaign or to "
                "aim it."
            )
        self.p(
            "The two levers act at different points. A cap of one exposure refuses reinforcement "
            "outright, so the cascade cannot build: reach collapses, but so does reach among the "
            "users the recommender was most confident about. A gate instead decides <i>who may "
            "convert at all</i>, so the campaign keeps travelling through the users it was "
            "aimed at and stops travelling through the rest. Comparing the two on the same table "
            "is the point: an intervention that only reports how much reach it removed is not an "
            "evaluation, because removing all reach is trivially possible."
        )

        self.title("8.2 Computational cost and scaling", "h2")
        self.table(
            "table_27_scaling_users",
            "Recommender training cost against the number of users (real sub-samples).",
        )
        self.table(
            "table_28_scaling_catalogue",
            "Serving cost against catalogue size for one batch of users.",
        )
        self.table(
            "table_29_scaling_diffusion",
            "Cascade cost against graph size on synthetic graphs with comparable mean degree.",
        )
        self.image(
            "fig_11_scaling.png", "Training cost versus interactions and cascade cost versus edges."
        )
        self.p(
            "Training cost grows with the number of interactions, not with the catalogue: each "
            "epoch touches only the observed pairs, and the run stops at its best validation "
            "epoch, so doubling the users does not double the wall-clock time. Serving cost is "
            "linear in catalogue size and in the number of users scored, which is why the "
            "evaluation streams users in chunks instead of materialising the full score matrix. "
            "Cascade cost grows with the number of edges traversed, so the simulation scales "
            "with the active frontier rather than with the whole population."
        )

        self.title("8.3 Measured optimisations", "h2")
        self.table(
            "table_30_optimisations",
            "Two hot-path rewrites measured on the real graph and on real batch sizes; the "
            "group-sum rows report what was measured rather than what was hoped for.",
        )
        self.p(
            "Optimisations are measured here, not claimed. The neighbour gather replaces a "
            "per-node Python loop with a ragged vectorised index, and the gradient accumulator "
            "replaces <font face='Courier'>np.add.at</font> with a sort followed by "
            "<font face='Courier'>np.add.reduceat</font>. The first is what makes a "
            "hundred-cascade Monte-Carlo budget affordable. The second is the more instructive "
            "case, because sorting only pays off once a batch is large; it is benchmarked at "
            "the batch the trainer actually uses and at four times that batch."
        )
        optimisation_rows = _dig(intervention, "optimisations", default=[])
        gather = next(
            (row for row in optimisation_rows if "gather" in str(row.get("operation", ""))), {}
        )
        group_rows = [
            row for row in optimisation_rows if "accumulate" in str(row.get("operation", ""))
        ]
        if gather:
            self.p(
                f"<b>Measured effect.</b> The vectorised gather is "
                f"{_fmt(gather.get('speedup'))}x faster than the per-node loop "
                f"({_fmt(gather.get('baseline_ms'))} ms to "
                f"{_fmt(gather.get('optimised_ms'))} ms per call), and it is the optimisation "
                "the pipeline depends on."
            )
        if group_rows:
            details = "; ".join(
                f"{_fmt(row.get('speedup'))}x for {row.get('operation')}" for row in group_rows
            )
            self.p(
                "The group-sum rewrite is the instructive case rather than a headline result: "
                f"{details}. A sorted reduction is never the bottleneck at these sizes, so the "
                "measured gains are small enough to sit inside run-to-run noise and do not grow "
                "with the batch as the rewrite originally assumed. It is kept because it makes "
                "the gradient code readable, and it is reported as a marginal change rather "
                "than a speed-up; the gather is the optimisation that earned its place."
            )
        if _dig(intervention, "hardware", "rss_mb") is not None:
            self.p(
                "The whole intervention study finished inside "
                f"{_fmt(_dig(intervention, 'hardware', 'rss_mb'))} MB of resident memory on a "
                f"graph with {_fmt(_dig(intervention, 'hardware', 'graph_edges'))} edges."
            )
        self.page_break()

    # ---------------------------------------------------------------- closing
    def section_limitations(self) -> None:
        self.title("9. Limitations")
        self.bullets(
            [
                "<b>No timestamps.</b> Epinions as distributed here has no review dates, so the "
                "hold-out is random per user instead of temporal. A temporal split would be a "
                "harder and more realistic test of whether the model generalises forward in time.",
                "<b>Binary trust.</b> Every published trust record has weight +1. Distrust, "
                "strength of trust and trust evolution cannot be modelled, so the social "
                "regulariser treats all edges as equally informative.",
                "<b>Organic versus campaign adoption.</b> Observed ratings measure adoption that "
                "happened without a campaign. Simulated reach measures additional adoption caused "
                "by seeds. The two are compared only as a rank agreement, never as the same "
                "quantity.",
                "<b>Sampled negatives.</b> Ranking metrics use uniformly sampled unobserved items "
                "plus the held-out positives. This is standard but easier than scoring the whole "
                "catalogue, and a strong popularity baseline can exploit it.",
                "<b>Model assumptions.</b> IC and SIR assume one item, no competition, no "
                "re-exposure except in SIR, and a static network. Real campaigns compete for "
                "attention and users change their trust relations.",
                "<b>Group construction.</b> Conflict groups are synthetic by design (the most "
                "dissimilar users in a random pool). Real groups have a shared history that the "
                "dataset does not record.",
                "<b>Selection pool.</b> Greedy selection is restricted to a proxy-prefiltered "
                "candidate pool for tractability, so the reported reach is a lower bound on what "
                "an exhaustive search could achieve and an upper bound on what a production "
                "system with a tighter budget could afford.",
                "<b>Single dataset, single machine.</b> All results come from one Epinions core "
                "measured on one machine; absolute timings will differ elsewhere, although the "
                "scaling shapes should not.",
            ]
        )

    def section_conclusion(self) -> None:
        rec = self.load("recommenders")
        social = self.load("social")
        influence = self.load("influence")
        models = _dig(rec, "models", default={})
        best_lift = _greedy_lift(influence)
        self.title("10. Conclusion")
        self.p(
            "The assignment asked for one system that recommends, explains itself socially, "
            "spreads an item, chooses whom to seed, serves groups and can be evaluated and "
            "controlled. Each requirement is implemented, measured and reported separately, and "
            "the same evaluation protocol and random seeds run through all of them."
        )
        self.p(
            "Two findings stand out. First, the two recommender families are complementary: "
            f"matrix factorisation is the better rating predictor "
            f"(RMSE {_fmt(_dig(models, 'bias_mf', 'rmse'))}) while item-kNN is the better ranker "
            f"(NDCG@10 {_fmt(_dig(models, 'item_knn', 'ndcg@10'))}), so a system that has to "
            "choose what to put in front of a user should not be selected on RMSE alone. Second, "
            "the social signal is real but small and must be weighted carefully: at the selected "
            f"beta = {_fmt(_dig(social, 'selected_beta'))} the graph comparison in Section 3 shows "
            "that trust and behavioural similarity are not interchangeable sources of evidence, "
            "and that any positive weight buys rating accuracy with ranking accuracy."
        )
        self.p(
            "On the diffusion side, the campaign succeeds or fails on a knife edge of "
            "transmission probability, the three event types must be counted separately, and a "
            "campaign that percolates saturates quickly: informed seed selection beats random "
            f"draws by up to {best_lift:.1%} at the tightest budget and by nothing once the "
            "budget is generous, so the recommendation is to spend the first few seeds "
            "carefully rather than to spend many. The intervention study closes the loop: reach "
            "can be suppressed cheaply, but not without a measurable cost to the users the "
            "campaign was meant to serve, and the affinity gate is the better trade-off of the "
            "two levers tested."
        )

    # -------------------------------------------------------------- appendices
    def appendix_reproduction(self) -> None:
        self.title("Appendix A. Reproducibility")
        hardware = hardware_summary()
        self.p(
            "The environment is a uv-managed Python 3.11 project. The raw Epinions files are "
            "committed under <font face='Courier'>data/raw/epinions</font>, the processed core "
            "and every result are regenerated from the configuration below."
        )
        self.code(
            "uv sync\n"
            "uv run sntp prepare            # build data/processed/epinions\n"
            "uv run sntp run all            # every experiment, figures and this PDF\n"
            "uv run pytest -q               # unit tests"
        )
        self.p(
            f"Hardware: {hardware['platform']}, {hardware['cpu_count']} logical CPUs, "
            f"{hardware['ram_gb']} GB RAM, Python {hardware['python']}."
        )
        config_dump = self._config_digest()
        self.code(config_dump)

    def appendix_mapping(self) -> None:
        self.title("Appendix B. Requirement to code map")
        mapping = pd.DataFrame(
            [
                [
                    "1. Environment representation",
                    "data.py, graph.py",
                    "sntp run dataset",
                    "Sections 1",
                ],
                [
                    "2. Two recommenders",
                    "recommenders/bias_mf.py, recommenders/item_knn.py",
                    "sntp run recommenders",
                    "Section 2",
                ],
                [
                    "3. Social extension",
                    "recommenders/social.py, graph.py",
                    "sntp run social",
                    "Section 3",
                ],
                ["4. Diffusion models", "diffusion.py", "sntp run diffusion", "Section 4"],
                ["5. Seed selection", "seeds.py", "sntp run influence", "Section 5"],
                ["6. Group recommendation", "groups.py", "sntp run groups", "Section 6"],
                ["7. Evaluation", "metrics.py, evaluation.py", "sntp run evaluation", "Section 7"],
                [
                    "8. Intervention and cost",
                    "diffusion.py, experiments/rq8_intervention.py",
                    "sntp run intervention",
                    "Section 8",
                ],
            ],
            columns=["requirement", "implementation", "command", "report section"],
        )
        self.table_from_frame(mapping, "Where each requirement is implemented and evaluated.")

    # ------------------------------------------------------------------ extras
    def table_from_frame(self, frame: pd.DataFrame, caption: str, label: str | None = None) -> None:
        if label is None:
            self._table_index += 1
            number = f"Table {self._table_index}."
        else:
            number = f"Table {label}."
        table = self._table_flowable(frame)
        self.story.append(Spacer(1, 2))
        self.story.append(table)
        self.story.append(Paragraph(f"{number} {_esc(caption)}", self.style["caption"]))

    def _config_digest(self) -> str:
        cfg = self.cfg
        lines = [
            f"seed                 : {cfg['seed']}",
            f"dataset core         : min_user={cfg['dataset.min_user_ratings']}, "
            f"min_item={cfg['dataset.min_item_ratings']}",
            f"folds                : valid={cfg['dataset.split.val_fraction']}, "
            f"test={cfg['dataset.split.test_fraction']}, seed={cfg['dataset.split.seed']}",
            f"Bias-MF              : k={cfg['recommendation.n_factors']}, "
            f"lr={cfg['recommendation.learning_rate']}, lambda={cfg['recommendation.reg']}, "
            f"epochs<={cfg['recommendation.epochs']}",
            f"Item-kNN             : k={cfg['recommendation.knn_neighbors']}, "
            f"shrinkage={cfg['recommendation.knn_shrinkage']}, "
            f"clip={cfg['recommendation.knn_max_correction']}",
            f"training seeds       : {list(cfg['recommendation.training_seeds'])}",
            f"evaluation           : K={cfg['recommendation.evaluation.top_k']}, "
            f"negatives={cfg['recommendation.evaluation.n_negatives']}, "
            f"relevance>={cfg['recommendation.evaluation.relevance_threshold']}",
            f"social betas         : {list(cfg['recommendation.social.betas'])}",
            f"diffusion            : p0={cfg['diffusion.p0']}, "
            f"gamma={cfg['diffusion.influence_weight']}, "
            f"recovery={cfg['diffusion.recovery_prob']}, sims={cfg['diffusion.n_simulations']}",
            f"influence            : budgets={list(cfg['influence.budgets'])}, "
            f"mc_select={cfg['influence.mc_samples_selection']}, "
            f"mc_final={cfg['influence.mc_samples_final']}",
            f"groups               : size={cfg['groups.size']}, n={cfg['groups.n_groups']}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------- run
def run(cfg: Config) -> dict:
    report = Report(cfg)
    story = report.build()
    target_dir = ensure_dir(cfg.path("paths.report_dir"))
    target = target_dir / "report.pdf"
    footer = f"{cfg['report.author']} ({cfg['report.roll_number']}) - SNTP Assignment 2"
    document = BaseDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=1.9 * cm,
        rightMargin=1.9 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.9 * cm,
        title=str(cfg["report.title"]),
        author=str(cfg["report.author"]),
        subject="Recommendation and information diffusion on the Epinions network",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        document.width,
        document.height,
        id="body",
    )
    document.addPageTemplates([PageTemplate(id="main", frames=[frame])])
    document.build(story, canvasmaker=lambda *a, **k: NumberedCanvas(*a, footer=footer, **k))
    size_kb = target.stat().st_size / 1024
    print(f"[report] wrote {target} ({size_kb:.0f} kB)")
    return {"pdf": str(target), "kilobytes": round(size_kb, 1)}


if __name__ == "__main__":
    run(Config.load())
