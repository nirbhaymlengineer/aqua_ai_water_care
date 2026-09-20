from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AquaSkin AI",
    page_icon="💧",
    layout="wide",
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "water_model.pth"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ============================================================
# PARAMETER CONFIGURATION
# ============================================================

PARAMETER_NAMES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

DISPLAY_NAMES = {
    "ph": "pH",
    "Hardness": "Hardness",
    "Solids": "TDS",
    "Chloramines": "Chloramines",
    "Sulfate": "Sulfate",
    "Conductivity": "Conductivity",
    "Organic_carbon": "Organic carbon",
    "Trihalomethanes": "TTHMs",
    "Turbidity": "Turbidity",
}

UNITS = {
    "ph": "pH units",
    "Hardness": "mg/L as CaCO3",
    "Solids": "mg/L",
    "Chloramines": "mg/L",
    "Sulfate": "mg/L",
    "Conductivity": "uS/cm",
    "Organic_carbon": "mg/L",
    "Trihalomethanes": "ug/L",
    "Turbidity": "NTU",
}

REFERENCES = {
    "ph": {
        "low": 6.5,
        "high": 8.5,
        "label": "6.5 to 8.5",
    },
    "Hardness": {
        "high": 180.0,
        "label": "Up to 180 mg/L",
    },
    "Solids": {
        "high": 500.0,
        "label": "Up to 500 mg/L",
    },
    "Chloramines": {
        "high": 4.0,
        "label": "Up to 4 mg/L",
    },
    "Sulfate": {
        "high": 250.0,
        "label": "Up to 250 mg/L",
    },
    "Conductivity": {
        "label": "Indicator only",
    },
    "Organic_carbon": {
        "label": "Indicator only",
    },
    "Trihalomethanes": {
        "high": 80.0,
        "label": "Up to 80 ug/L",
    },
    "Turbidity": {
        "high": 5.0,
        "label": "Below 5 NTU",
    },
}

COLORS = {
    "navy": "#12355B",
    "blue": "#2878B5",
    "green": "#2E8B57",
    "amber": "#E6A700",
    "red": "#C83E4D",
    "gray": "#6B7280",
    "light_gray": "#F3F4F6",
    "dark": "#1F2937",
    "white": "#FFFFFF",
}


# ============================================================
# MODEL ARCHITECTURE
# Must match your training architecture exactly.
# ============================================================

class WaterNet(nn.Module):

    def __init__(self, input_size, hidden_layers=None):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [32, 16, 1]

        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# ============================================================
# LOAD MODEL AND SCALER
# Streamlit caches these objects and does not reload them on
# every button click.
# ============================================================

@st.cache_resource
def load_prediction_assets():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH.name}."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {SCALER_PATH.name}."
        )

    scaler = joblib.load(SCALER_PATH)
    input_size = int(scaler.n_features_in_)

    model = WaterNet(
        input_size=input_size,
        hidden_layers=[32, 16, 1],
    )

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu"),
        )

    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler


# ============================================================
# SCORE HELPERS
# These are transparent product rules.
# They are not outputs learned by the potability model.
# ============================================================

def cap_score(score):
    return float(max(0.0, min(100.0, score)))


def classify_score(score):

    if score >= 90:
        return "Excellent", COLORS["green"]

    if score >= 75:
        return "Good", COLORS["green"]

    if score >= 50:
        return "Caution", COLORS["amber"]

    if score >= 25:
        return "Poor", COLORS["red"]

    return "High concern", COLORS["red"]


def high_parameter_penalty(
    value,
    good_limit,
    maximum_penalty,
):

    if value <= good_limit:
        return 0.0

    ratio = value / good_limit

    severity = min(
        1.0,
        (ratio - 1.0) / 2.0,
    )

    return maximum_penalty * severity


def range_penalty(
    value,
    low,
    high,
    maximum_penalty,
):

    if low <= value <= high:
        return 0.0

    if value < low:
        distance_ratio = (
            low - value
        ) / max(low, 1e-9)

    else:
        distance_ratio = (
            value - high
        ) / max(high, 1e-9)

    return maximum_penalty * min(
        1.0,
        distance_ratio * 3.0,
    )


def calculate_scores(water):

    drinking_penalties = {
        "pH": range_penalty(
            water["ph"], 6.5, 8.5, 10
        ),
        "Hardness": high_parameter_penalty(
            water["Hardness"], 180, 5
        ),
        "TDS": high_parameter_penalty(
            water["Solids"], 500, 30
        ),
        "Chloramines": high_parameter_penalty(
            water["Chloramines"], 4, 20
        ),
        "Sulfate": high_parameter_penalty(
            water["Sulfate"], 250, 10
        ),
        "TTHMs": high_parameter_penalty(
            water["Trihalomethanes"], 80, 15
        ),
        "Turbidity": high_parameter_penalty(
            water["Turbidity"], 5, 10
        ),
    }

    skin_penalties = {
        "Hardness": high_parameter_penalty(
            water["Hardness"], 120, 40
        ),
        "Chloramines": high_parameter_penalty(
            water["Chloramines"], 4, 25
        ),
        "TDS": high_parameter_penalty(
            water["Solids"], 500, 20
        ),
        "pH": range_penalty(
            water["ph"], 6.5, 8.5, 10
        ),
        "Turbidity": high_parameter_penalty(
            water["Turbidity"], 5, 5
        ),
    }

    hair_penalties = {
        "Hardness": high_parameter_penalty(
            water["Hardness"], 120, 45
        ),
        "TDS": high_parameter_penalty(
            water["Solids"], 500, 25
        ),
        "Chloramines": high_parameter_penalty(
            water["Chloramines"], 4, 15
        ),
        "pH": range_penalty(
            water["ph"], 6.5, 8.5, 10
        ),
        "Turbidity": high_parameter_penalty(
            water["Turbidity"], 5, 5
        ),
    }

    scores = {
        "Drinking": cap_score(
            100 - sum(drinking_penalties.values())
        ),
        "Skin": cap_score(
            100 - sum(skin_penalties.values())
        ),
        "Hair": cap_score(
            100 - sum(hair_penalties.values())
        ),
    }

    return (
        scores,
        drinking_penalties,
        skin_penalties,
        hair_penalties,
    )


# ============================================================
# PARAMETER STATUS AND FINDINGS
# ============================================================

def parameter_status(name, value):

    reference = REFERENCES[name]

    if "low" in reference and "high" in reference:

        if reference["low"] <= value <= reference["high"]:
            return "Within selected range", COLORS["green"]

        return "Outside selected range", COLORS["red"]

    if "high" in reference:

        ratio = value / reference["high"]

        if ratio < 0.9:
            return "Within selected reference", COLORS["green"]

        if ratio <= 1.0:
            return "Near selected threshold", COLORS["amber"]

        return "Above selected reference", COLORS["red"]

    return "Indicator only", COLORS["blue"]


def build_findings(water):

    findings = []

    if water["Solids"] > 500:
        findings.append(
            (
                water["Solids"] / 500,
                "TDS is above the selected 500 mg/L reference.",
            )
        )

    if water["Chloramines"] > 4:
        findings.append(
            (
                water["Chloramines"] / 4,
                "Chloramines are above the selected 4 mg/L reference.",
            )
        )

    if water["Hardness"] > 180:
        findings.append(
            (
                water["Hardness"] / 180,
                "Water is classified as very hard by the selected scale.",
            )
        )

    if water["Sulfate"] > 250:
        findings.append(
            (
                water["Sulfate"] / 250,
                "Sulfate is above the selected 250 mg/L reference.",
            )
        )

    if water["Trihalomethanes"] > 80:
        findings.append(
            (
                water["Trihalomethanes"] / 80,
                "TTHMs are above the selected 80 ug/L reference.",
            )
        )

    if water["Turbidity"] > 5:
        findings.append(
            (
                water["Turbidity"] / 5,
                "Turbidity is above the selected 5 NTU benchmark.",
            )
        )

    if not 6.5 <= water["ph"] <= 8.5:
        findings.append(
            (
                2.0,
                "pH is outside the selected 6.5 to 8.5 range.",
            )
        )

    if not findings:
        findings.append(
            (
                0,
                "No scored parameter exceeded its selected reference.",
            )
        )

    findings.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        text for _, text in findings
    ]


def reference_ratios(water):

    return {
        "Hardness": water["Hardness"] / 180,
        "TDS": water["Solids"] / 500,
        "Chloramines": water["Chloramines"] / 4,
        "Sulfate": water["Sulfate"] / 250,
        "TTHMs": water["Trihalomethanes"] / 80,
        "Turbidity": water["Turbidity"] / 5,
    }


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_potability(
    model,
    scaler,
    water_sample,
):

    ordered_features = (
        PARAMETER_NAMES
        + [
            f"{name}_missing"
            for name in PARAMETER_NAMES
        ]
    )

    raw_array = np.array(
        [[
            water_sample[name]
            for name in ordered_features
        ]],
        dtype=np.float64,
    )

    if raw_array.shape[1] != scaler.n_features_in_:

        raise ValueError(
            f"The scaler expects "
            f"{scaler.n_features_in_} features, "
            f"but the application created "
            f"{raw_array.shape[1]} features."
        )

    scaled_array = scaler.transform(raw_array)

    tensor = torch.tensor(
        scaled_array,
        dtype=torch.float32,
    )

    with torch.no_grad():
        probability = float(
            model(tensor).item()
        )

    predicted_label = int(
        probability >= 0.5
    )

    confidence = max(
        probability,
        1.0 - probability,
    ) * 100

    return (
        probability,
        predicted_label,
        confidence,
    )


# ============================================================
# GRAPH 1: OVERALL SCORES
# ============================================================

def create_score_chart(scores):

    labels = list(scores.keys())
    values = list(scores.values())

    colors = [
        classify_score(score)[1]
        for score in values
    ]

    fig, ax = plt.subplots(
        figsize=(8, 3.8)
    )

    y_positions = np.arange(
        len(labels)
    )

    ax.barh(
        y_positions,
        values,
        color=colors,
        height=0.5,
    )

    ax.set_xlim(0, 100)
    ax.set_yticks(
        y_positions,
        labels,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Score out of 100")
    ax.set_title(
        "Overall assessment scores",
        loc="left",
        fontweight="bold",
    )

    ax.axvline(
        50,
        color=COLORS["amber"],
        linestyle="--",
    )

    ax.axvline(
        75,
        color=COLORS["green"],
        linestyle="--",
    )

    for index, value in enumerate(values):

        ax.text(
            min(value + 2, 95),
            index,
            f"{value:.0f}",
            va="center",
            fontweight="bold",
        )

    ax.spines[
        ["top", "right", "left"]
    ].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    return fig


# ============================================================
# GRAPH 2: PARAMETER REFERENCE RATIOS
# ============================================================

def create_ratio_chart(water):

    ratios = reference_ratios(water)

    labels = list(ratios.keys())
    actual_values = list(ratios.values())

    display_values = [
        min(value, 5.0)
        for value in actual_values
    ]

    colors = []

    for value in actual_values:

        if value < 0.9:
            colors.append(COLORS["green"])

        elif value <= 1.0:
            colors.append(COLORS["amber"])

        else:
            colors.append(COLORS["red"])

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    y_positions = np.arange(
        len(labels)
    )

    ax.barh(
        y_positions,
        display_values,
        color=colors,
        height=0.52,
    )

    ax.set_yticks(
        y_positions,
        labels,
    )

    ax.invert_yaxis()

    ax.axvline(
        1.0,
        color=COLORS["dark"],
        linestyle="--",
        linewidth=1.3,
        label="Selected reference",
    )

    ax.set_xlim(0, 5.5)

    ax.set_xlabel(
        "Measured value / selected reference"
    )

    ax.set_title(
        "Parameter-to-reference comparison",
        loc="left",
        fontweight="bold",
    )

    for index, (
        shown_value,
        actual_value,
    ) in enumerate(
        zip(
            display_values,
            actual_values,
        )
    ):

        if actual_value > 5:
            label = f">5x, actual {actual_value:.1f}x"
        else:
            label = f"{actual_value:.2f}x"

        ax.text(
            min(shown_value + 0.12, 5.05),
            index,
            label,
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.legend(
        loc="lower right",
        frameon=False,
    )

    ax.spines[
        ["top", "right", "left"]
    ].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    return fig


# ============================================================
# GRAPHS 3, 4, AND 5: SCORE PENALTIES
# ============================================================

def create_penalty_chart(
    penalties,
    title,
):

    sorted_items = sorted(
        penalties.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    labels = [
        item[0]
        for item in sorted_items
    ]

    values = [
        item[1]
        for item in sorted_items
    ]

    total = sum(values)

    percentages = [
        value / total * 100
        if total
        else 0
        for value in values
    ]

    colors = []

    for percentage in percentages:

        if percentage >= 20:
            colors.append(COLORS["red"])

        elif percentage >= 8:
            colors.append(COLORS["amber"])

        else:
            colors.append(COLORS["blue"])

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    y_positions = np.arange(
        len(labels)
    )

    ax.barh(
        y_positions,
        percentages,
        color=colors,
        height=0.5,
    )

    ax.set_yticks(
        y_positions,
        labels,
    )

    ax.invert_yaxis()

    maximum = max(
        percentages,
        default=0,
    )

    ax.set_xlim(
        0,
        max(45, maximum + 12),
    )

    ax.set_xlabel(
        "Share of score penalty (%)"
    )

    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
    )

    for index, value in enumerate(
        percentages
    ):

        ax.text(
            value + 0.7,
            index,
            f"{value:.0f}%",
            va="center",
            fontsize=8,
        )

    ax.spines[
        ["top", "right", "left"]
    ].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    return fig


# ============================================================
# GRAPH 6: DATA COMPLETENESS
# ============================================================

def create_completeness_chart(water):

    measured = sum(
        1
        for name in PARAMETER_NAMES
        if water[f"{name}_missing"] == 0
    )

    estimated = (
        len(PARAMETER_NAMES)
        - measured
    )

    if estimated == 0:

        values = [measured]

        labels = [
            f"Measured\n{measured}"
        ]

        colors = [
            COLORS["green"]
        ]

    else:

        values = [
            measured,
            estimated,
        ]

        labels = [
            f"Measured\n{measured}",
            f"Estimated\n{estimated}",
        ]

        colors = [
            COLORS["green"],
            COLORS["gray"],
        ]

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops={
            "width": 0.38,
            "edgecolor": "white",
        },
    )

    ax.text(
        0,
        0,
        f"{measured}/{len(PARAMETER_NAMES)}",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["navy"],
    )

    ax.set_title(
        "Data completeness",
        fontweight="bold",
    )

    fig.tight_layout()

    return fig


# ============================================================
# GRAPH 7: HISTORICAL TREND PLACEHOLDER
# ============================================================

def create_trend_placeholder():

    fig, ax = plt.subplots(
        figsize=(8, 3.5)
    )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.text(
        5,
        5.7,
        "Historical trend unavailable",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=COLORS["gray"],
    )

    ax.text(
        5,
        4.3,
        "At least two measurement periods are required.",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["gray"],
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        "Historical trend",
        loc="left",
        fontweight="bold",
    )

    for spine in ax.spines.values():
        spine.set_color("#D1D5DB")

    fig.tight_layout()

    return fig


# ============================================================
# PDF DRAWING HELPERS
# ============================================================

def set_pdf_page_style(
    fig,
    title,
    subtitle,
):

    fig.patch.set_facecolor(
        COLORS["white"]
    )

    fig.text(
        0.06,
        0.965,
        title,
        fontsize=22,
        fontweight="bold",
        color=COLORS["navy"],
    )

    fig.text(
        0.06,
        0.935,
        subtitle,
        fontsize=10,
        color=COLORS["gray"],
    )

    fig.add_artist(
        plt.Line2D(
            [0.06, 0.94],
            [0.918, 0.918],
            transform=fig.transFigure,
            color=COLORS["blue"],
            linewidth=2,
        )
    )


def add_pdf_footer(
    fig,
    page_number,
):

    footer = (
        "Informational decision-support only. "
        "Not laboratory certification or medical advice."
    )

    fig.text(
        0.06,
        0.025,
        footer,
        fontsize=7.5,
        color=COLORS["gray"],
    )

    fig.text(
        0.94,
        0.025,
        f"Page {page_number}",
        fontsize=8,
        color=COLORS["gray"],
        ha="right",
    )


def draw_pdf_score_card(
    fig,
    x,
    y,
    width,
    height,
    title,
    score,
):

    status, color = classify_score(
        score
    )

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=(
            "round,pad=0.012,"
            "rounding_size=0.018"
        ),
        transform=fig.transFigure,
        facecolor=COLORS["light_gray"],
        edgecolor=color,
        linewidth=2,
    )

    fig.add_artist(box)

    fig.text(
        x + 0.02,
        y + height - 0.055,
        title,
        fontsize=11,
        fontweight="bold",
        color=COLORS["dark"],
    )

    fig.text(
        x + 0.02,
        y + 0.065,
        f"{score:.0f}/100",
        fontsize=25,
        fontweight="bold",
        color=color,
    )

    fig.text(
        x + width - 0.02,
        y + 0.07,
        status,
        fontsize=10,
        fontweight="bold",
        color=color,
        ha="right",
    )


def add_pdf_text_box(
    fig,
    x,
    y,
    width,
    height,
    title,
    lines,
    color,
):

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=(
            "round,pad=0.012,"
            "rounding_size=0.014"
        ),
        transform=fig.transFigure,
        facecolor=COLORS["light_gray"],
        edgecolor=color,
        linewidth=1.5,
    )

    fig.add_artist(box)

    fig.text(
        x + 0.02,
        y + height - 0.04,
        title,
        fontsize=11,
        fontweight="bold",
        color=color,
    )

    current_y = (
        y + height - 0.075
    )

    for line in lines:

        fig.text(
            x + 0.025,
            current_y,
            f"• {line}",
            fontsize=8.2,
            color=COLORS["dark"],
            va="top",
        )

        current_y -= 0.035


# ============================================================
# CREATE FIVE-PAGE PDF IN MEMORY
# No permanent server file is necessary.
# ============================================================

def create_dashboard_pdf(
    water,
    probability,
    predicted_label,
    model_confidence,
    scores,
    drinking_penalties,
    skin_penalties,
    hair_penalties,
):

    pdf_buffer = BytesIO()

    generated_at = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )

    findings = build_findings(water)

    with PdfPages(pdf_buffer) as pdf:

        # ----------------------------------------------------
        # PAGE 1
        # ----------------------------------------------------

        fig = plt.figure(
            figsize=(11.69, 8.27)
        )

        set_pdf_page_style(
            fig,
            "AquaSkin AI Water Dashboard",
            (
                "Drinking, skin, and hair assessment"
                f" | Generated {generated_at}"
            ),
        )

        draw_pdf_score_card(
            fig,
            0.06,
            0.71,
            0.27,
            0.16,
            "Drinking Water",
            scores["Drinking"],
        )

        draw_pdf_score_card(
            fig,
            0.365,
            0.71,
            0.27,
            0.16,
            "Skin Comfort",
            scores["Skin"],
        )

        draw_pdf_score_card(
            fig,
            0.67,
            0.71,
            0.27,
            0.16,
            "Hair Comfort",
            scores["Hair"],
        )

        score_ax = fig.add_axes(
            [0.08, 0.36, 0.45, 0.27]
        )

        labels = list(scores.keys())
        values = list(scores.values())

        score_ax.barh(
            labels,
            values,
            color=[
                classify_score(value)[1]
                for value in values
            ],
        )

        score_ax.set_xlim(0, 100)
        score_ax.invert_yaxis()

        score_ax.set_title(
            "Overall assessment scores",
            loc="left",
            fontweight="bold",
        )

        score_ax.spines[
            ["top", "right", "left"]
        ].set_visible(False)

        model_status = (
            "Potable"
            if predicted_label == 1
            else "Not potable"
        )

        add_pdf_text_box(
            fig,
            0.58,
            0.43,
            0.35,
            0.20,
            "AI model result",
            [
                f"Model probability: {probability:.1%}",
                f"Threshold decision: {model_status}",
                f"Model confidence: {model_confidence:.1f}%",
                "This output is not laboratory certification.",
            ],
            COLORS["blue"],
        )

        add_pdf_text_box(
            fig,
            0.06,
            0.10,
            0.42,
            0.20,
            "Top findings",
            findings[:3],
            COLORS["amber"],
        )

        add_pdf_text_box(
            fig,
            0.52,
            0.10,
            0.41,
            0.20,
            "Recommended next steps",
            [
                "Confirm concerning results in a laboratory.",
                "Inspect treatment and filtration equipment.",
                "Retest after filtration or softening changes.",
                "Seek professional advice for persistent symptoms.",
            ],
            COLORS["green"],
        )

        add_pdf_footer(fig, 1)

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # PAGE 2
        # ----------------------------------------------------

        fig = plt.figure(
            figsize=(11.69, 8.27)
        )

        drinking_status, drinking_color = (
            classify_score(
                scores["Drinking"]
            )
        )

        set_pdf_page_style(
            fig,
            "Drinking Water Assessment",
            (
                f"Score {scores['Drinking']:.0f}/100"
                f" | {drinking_status}"
            ),
        )

        ratio_ax = fig.add_axes(
            [0.08, 0.49, 0.84, 0.35]
        )

        ratios = reference_ratios(water)
        labels = list(ratios.keys())
        values = list(ratios.values())
        displayed = [
            min(value, 5)
            for value in values
        ]

        ratio_ax.barh(
            labels,
            displayed,
            color=[
                COLORS["green"]
                if value < 0.9
                else COLORS["amber"]
                if value <= 1
                else COLORS["red"]
                for value in values
            ],
        )

        ratio_ax.axvline(
            1,
            color=COLORS["dark"],
            linestyle="--",
        )

        ratio_ax.invert_yaxis()

        ratio_ax.set_title(
            "Parameter-to-reference comparison",
            loc="left",
            fontweight="bold",
        )

        penalty_ax = fig.add_axes(
            [0.08, 0.09, 0.44, 0.30]
        )

        sorted_penalties = sorted(
            drinking_penalties.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        penalty_ax.barh(
            [
                item[0]
                for item in sorted_penalties
            ],
            [
                item[1]
                for item in sorted_penalties
            ],
            color=COLORS["amber"],
        )

        penalty_ax.invert_yaxis()

        penalty_ax.set_title(
            "Drinking score penalties",
            loc="left",
            fontweight="bold",
        )

        add_pdf_text_box(
            fig,
            0.57,
            0.09,
            0.36,
            0.30,
            "Interpretation",
            [
                findings[0],
                "TDS does not identify individual substances.",
                "Hardness is mainly a scaling and comfort indicator.",
                "Critical results should be laboratory verified.",
            ],
            drinking_color,
        )

        add_pdf_footer(fig, 2)

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # PAGE 3
        # ----------------------------------------------------

        fig = plt.figure(
            figsize=(11.69, 8.27)
        )

        skin_status, skin_color = (
            classify_score(scores["Skin"])
        )

        set_pdf_page_style(
            fig,
            "Skin Comfort Assessment",
            (
                f"Score {scores['Skin']:.0f}/100"
                f" | {skin_status}"
            ),
        )

        draw_pdf_score_card(
            fig,
            0.06,
            0.72,
            0.30,
            0.15,
            "Skin Comfort",
            scores["Skin"],
        )

        skin_ax = fig.add_axes(
            [0.08, 0.25, 0.49, 0.38]
        )

        skin_items = sorted(
            skin_penalties.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        skin_ax.barh(
            [
                item[0]
                for item in skin_items
            ],
            [
                item[1]
                for item in skin_items
            ],
            color=skin_color,
        )

        skin_ax.invert_yaxis()

        skin_ax.set_title(
            "Skin comfort factor contribution",
            loc="left",
            fontweight="bold",
        )

        add_pdf_text_box(
            fig,
            0.61,
            0.50,
            0.32,
            0.27,
            "Interpretation",
            [
                "Hardness may influence mineral residue.",
                "TDS may affect rinsing comfort.",
                "Chloramine may reduce comfort for some users.",
                "This score does not diagnose skin conditions.",
            ],
            skin_color,
        )

        add_pdf_text_box(
            fig,
            0.61,
            0.17,
            0.32,
            0.25,
            "Recommended actions",
            [
                "Confirm hardness and disinfectant measurements.",
                "Inspect filter maintenance.",
                "Retest after treatment changes.",
                "Seek medical advice for persistent symptoms.",
            ],
            COLORS["green"],
        )

        add_pdf_footer(fig, 3)

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # PAGE 4
        # ----------------------------------------------------

        fig = plt.figure(
            figsize=(11.69, 8.27)
        )

        hair_status, hair_color = (
            classify_score(scores["Hair"])
        )

        set_pdf_page_style(
            fig,
            "Hair Comfort Assessment",
            (
                f"Score {scores['Hair']:.0f}/100"
                f" | {hair_status}"
            ),
        )

        draw_pdf_score_card(
            fig,
            0.06,
            0.72,
            0.30,
            0.15,
            "Hair Comfort",
            scores["Hair"],
        )

        hair_ax = fig.add_axes(
            [0.08, 0.25, 0.49, 0.38]
        )

        hair_items = sorted(
            hair_penalties.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        hair_ax.barh(
            [
                item[0]
                for item in hair_items
            ],
            [
                item[1]
                for item in hair_items
            ],
            color=hair_color,
        )

        hair_ax.invert_yaxis()

        hair_ax.set_title(
            "Hair comfort factor contribution",
            loc="left",
            fontweight="bold",
        )

        add_pdf_text_box(
            fig,
            0.61,
            0.50,
            0.32,
            0.27,
            "Interpretation",
            [
                "Hard water may increase mineral residue.",
                "TDS may affect rinsing comfort.",
                "Chloramine may reduce comfort for some users.",
                "This score does not diagnose hair loss.",
            ],
            hair_color,
        )

        add_pdf_text_box(
            fig,
            0.61,
            0.17,
            0.32,
            0.25,
            "Recommended actions",
            [
                "Confirm the hardness measurement.",
                "Investigate appropriate hardness treatment.",
                "Do not assume all shower filters soften water.",
                "Seek medical advice for persistent hair loss.",
            ],
            COLORS["green"],
        )

        add_pdf_footer(fig, 4)

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # PAGE 5
        # ----------------------------------------------------

        fig = plt.figure(
            figsize=(11.69, 8.27)
        )

        set_pdf_page_style(
            fig,
            "Data Quality and Traceability",
            "Input record, sensor status, and limitations",
        )

        donut_ax = fig.add_axes(
            [0.07, 0.49, 0.33, 0.35]
        )

        donut_ax.pie(
            [9],
            colors=[COLORS["green"]],
            startangle=90,
            wedgeprops={
                "width": 0.38,
                "edgecolor": "white",
            },
        )

        donut_ax.text(
            0,
            0,
            "9/9",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )

        donut_ax.set_title(
            "Data completeness",
            fontweight="bold",
        )

        table_ax = fig.add_axes(
            [0.45, 0.43, 0.49, 0.44]
        )

        table_ax.axis("off")

        rows = []

        for name in PARAMETER_NAMES:

            status, _ = parameter_status(
                name,
                water[name],
            )

            rows.append([DISPLAY_NAMES[name],f"{water[name]:g}",UNITS[name],status])

        table = table_ax.table(
            cellText=rows,
            colLabels=[
                "Parameter",
                "Value",
                "Unit",
                "Status",
            ],
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[
                0.24,
                0.14,
                0.24,
                0.38,
            ],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(7.3)
        table.scale(1, 1.35)

        trend_ax = fig.add_axes(
            [0.08, 0.11, 0.50, 0.23]
        )

        trend_ax.text(
            0.5,
            0.58,
            "Historical trend unavailable",
            transform=trend_ax.transAxes,
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=COLORS["gray"],
        )

        trend_ax.text(
            0.5,
            0.40,
            "At least two measurements are required.",
            transform=trend_ax.transAxes,
            ha="center",
            color=COLORS["gray"],
        )

        trend_ax.set_xticks([])
        trend_ax.set_yticks([])

        add_pdf_text_box(
            fig,
            0.63,
            0.11,
            0.30,
            0.23,
            "Traceability",
            [
                "Model architecture: 18-32-16-1",
                "Scoring-rule version: MVP 1.0",
                "All nine values were supplied by the user.",
                f"Generated: {generated_at}",
            ],
            COLORS["blue"],
        )

        add_pdf_footer(fig, 5)

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

        plt.close(fig)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ============================================================
# LOAD ASSETS
# ============================================================

try:
    model, scaler = load_prediction_assets()

except Exception as error:

    st.error(
        f"Application startup error: {error}"
    )

    st.info(
        "Make sure water_model.pth and scaler.pkl "
        "are in the same directory as app.py."
    )

    st.stop()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("💧 AquaSkin AI")

st.subheader(
    "Skin, Hair, and Water Quality Dashboard"
)

st.write(
    "Enter all nine laboratory or sensor measurements. "
    "The application will run the trained potability model "
    "and create transparent skin, hair, and drinking-water "
    "reference scores."
)

st.warning(
    "This application is informational only. "
    "It does not diagnose skin conditions or hair loss, "
    "and it does not certify that water is safe to drink."
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("water_input_form"):

    left_column, middle_column, right_column = (
        st.columns(3)
    )

    with left_column:

        ph = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=7.8,
            step=0.1,
        )

        hardness = st.number_input(
            "Hardness (mg/L as CaCO3)",
            min_value=0.0,
            value=320.0,
            step=1.0,
        )

        solids = st.number_input(
            "TDS / Solids (mg/L)",
            min_value=0.0,
            value=22000.0,
            step=10.0,
        )

    with middle_column:

        chloramines = st.number_input(
            "Chloramines (mg/L)",
            min_value=0.0,
            value=8.0,
            step=0.1,
        )

        sulfate = st.number_input(
            "Sulfate (mg/L)",
            min_value=0.0,
            value=350.0,
            step=1.0,
        )

        conductivity = st.number_input(
            "Conductivity (uS/cm)",
            min_value=0.0,
            value=420.0,
            step=1.0,
        )

    with right_column:

        organic_carbon = st.number_input(
            "Organic carbon (mg/L)",
            min_value=0.0,
            value=12.0,
            step=0.1,
        )

        trihalomethanes = st.number_input(
            "Trihalomethanes / TTHMs (ug/L)",
            min_value=0.0,
            value=80.0,
            step=1.0,
        )

        turbidity = st.number_input(
            "Turbidity (NTU)",
            min_value=0.0,
            value=4.0,
            step=0.1,
        )

    analyze_button = st.form_submit_button(
        "Analyze Water",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    water_sample = {
        "ph": float(ph),
        "Hardness": float(hardness),
        "Solids": float(solids),
        "Chloramines": float(chloramines),
        "Sulfate": float(sulfate),
        "Conductivity": float(conductivity),
        "Organic_carbon": float(organic_carbon),
        "Trihalomethanes": float(trihalomethanes),
        "Turbidity": float(turbidity),
    }

    # This version requires all nine values.
    # Therefore all missing flags are zero.
    for parameter_name in PARAMETER_NAMES:

        water_sample[
            f"{parameter_name}_missing"
        ] = 0.0

    try:

        (
            probability,
            predicted_label,
            model_confidence,
        ) = predict_potability(
            model,
            scaler,
            water_sample,
        )

        (
            scores,
            drinking_penalties,
            skin_penalties,
            hair_penalties,
        ) = calculate_scores(
            water_sample
        )

    except Exception as error:

        st.error(
            f"Analysis failed: {error}"
        )

        st.stop()

    st.success(
        "Water analysis completed."
    )

    # --------------------------------------------------------
    # SCORE CARDS
    # --------------------------------------------------------

    st.header("Management Summary")

    score_column_1, score_column_2, score_column_3 = (
        st.columns(3)
    )

    with score_column_1:

        drinking_status, _ = classify_score(
            scores["Drinking"]
        )

        st.metric(
            "Drinking Water Score",
            f"{scores['Drinking']:.0f}/100",
        )

        st.caption(drinking_status)

    with score_column_2:

        skin_status, _ = classify_score(
            scores["Skin"]
        )

        st.metric(
            "Skin Comfort Score",
            f"{scores['Skin']:.0f}/100",
        )

        st.caption(skin_status)

    with score_column_3:

        hair_status, _ = classify_score(
            scores["Hair"]
        )

        st.metric(
            "Hair Comfort Score",
            f"{scores['Hair']:.0f}/100",
        )

        st.caption(hair_status)

    model_label = (
        "Potable"
        if predicted_label == 1
        else "Not potable"
    )

    model_column_1, model_column_2, model_column_3 = (
        st.columns(3)
    )

    model_column_1.metric(
        "Model Potability Probability",
        f"{probability:.1%}",
    )

    model_column_2.metric(
        "Threshold Decision",
        model_label,
    )

    model_column_3.metric(
        "Model Confidence",
        f"{model_confidence:.1f}%",
    )

    st.caption(
        "Model confidence means distance from the 0.50 "
        "classification threshold. It does not measure "
        "laboratory certainty."
    )

    # --------------------------------------------------------
    # GRAPH 1
    # --------------------------------------------------------

    st.header("1. Overall Assessment Scores")

    score_fig = create_score_chart(
        scores
    )

    st.pyplot(score_fig)
    plt.close(score_fig)

    # --------------------------------------------------------
    # FINDINGS
    # --------------------------------------------------------

    st.header("Top Findings")

    findings = build_findings(
        water_sample
    )

    for finding in findings[:4]:
        st.write(f"• {finding}")

    # --------------------------------------------------------
    # GRAPH 2
    # --------------------------------------------------------

    st.header(
        "2. Parameter-to-Reference Comparison"
    )

    st.write(
        "A ratio of 1.0 means the measurement equals "
        "the selected reference. The visual display is "
        "capped at five times the reference."
    )

    ratio_fig = create_ratio_chart(
        water_sample
    )

    st.pyplot(ratio_fig)
    plt.close(ratio_fig)

    # --------------------------------------------------------
    # GRAPHS 3, 4, 5
    # --------------------------------------------------------

    st.header("Score Factor Contributions")

    penalty_tab_1, penalty_tab_2, penalty_tab_3 = (
        st.tabs(
            [
                "3. Drinking",
                "4. Skin",
                "5. Hair",
            ]
        )
    )

    with penalty_tab_1:

        drinking_fig = create_penalty_chart(
            drinking_penalties,
            "Why the drinking score decreased",
        )

        st.pyplot(drinking_fig)
        plt.close(drinking_fig)

    with penalty_tab_2:

        skin_fig = create_penalty_chart(
            skin_penalties,
            "Skin comfort factor contribution",
        )

        st.pyplot(skin_fig)
        plt.close(skin_fig)

        st.info(
            "This score describes water-comfort factors. "
            "It does not diagnose eczema, irritation, "
            "or another medical condition."
        )

    with penalty_tab_3:

        hair_fig = create_penalty_chart(
            hair_penalties,
            "Hair comfort factor contribution",
        )

        st.pyplot(hair_fig)
        plt.close(hair_fig)

        st.info(
            "This score describes water-comfort factors. "
            "It does not diagnose or explain hair loss."
        )

    # --------------------------------------------------------
    # GRAPH 6 AND INPUT TABLE
    # --------------------------------------------------------

    st.header("6. Data Quality and Input Status")

    completeness_column, table_column = (
        st.columns([1, 2])
    )

    with completeness_column:

        completeness_fig = (
            create_completeness_chart(
                water_sample
            )
        )

        st.pyplot(completeness_fig)
        plt.close(completeness_fig)

    with table_column:

        status_rows = []

        for name in PARAMETER_NAMES:

            status, _ = parameter_status(
                name,
                water_sample[name],
            )

            status_rows.append(
                {
                    "Parameter": DISPLAY_NAMES[name],
                    "Value": water_sample[name],
                    "Unit": UNITS[name],
                    "Status": status,
                    "Reference": REFERENCES[name]["label"],
                }
            )

        status_dataframe = pd.DataFrame(
            status_rows
        )

        st.dataframe(
            status_dataframe,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # GRAPH 7
    # --------------------------------------------------------

    st.header("7. Historical Trend")

    trend_fig = create_trend_placeholder()

    st.pyplot(trend_fig)
    plt.close(trend_fig)

    st.caption(
        "When the IoT device begins storing measurements, "
        "this section can show changes in hardness, pH, "
        "turbidity, TDS, and comfort scores over time."
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.header("Recommended Next Steps")

    recommendations = []

    if water_sample["Hardness"] > 180:

        recommendations.append(
            "Confirm hardness with a suitable test. "
            "Investigate water-softening options if scaling "
            "and mineral residue are household problems."
        )

    if water_sample["Solids"] > 500:

        recommendations.append(
            "Confirm TDS through laboratory testing. "
            "TDS alone does not identify the dissolved substances."
        )

    if water_sample["Chloramines"] > 4:

        recommendations.append(
            "Confirm chloramine concentration and investigate "
            "treatment certified for chloramine reduction."
        )

    if water_sample["Turbidity"] > 5:

        recommendations.append(
            "Inspect the water source and filtration equipment "
            "and confirm turbidity using a calibrated instrument."
        )

    if not 6.5 <= water_sample["ph"] <= 8.5:

        recommendations.append(
            "Confirm pH using a calibrated meter and investigate "
            "the source of the unusual result."
        )

    if not recommendations:

        recommendations.append(
            "No scored measurement exceeded its selected reference. "
            "Continue periodic monitoring."
        )

    recommendations.extend(
        [
            "Retest after changing filters or treatment equipment.",
            "Consult a healthcare professional for persistent "
            "skin symptoms or significant hair loss.",
        ]
    )

    for recommendation in recommendations:
        st.write(f"✅ {recommendation}")

    # --------------------------------------------------------
    # PDF GENERATION AND DOWNLOAD
    # --------------------------------------------------------

    st.header("Download Customer Report")

    with st.spinner(
        "Generating the five-page dashboard report..."
    ):

        pdf_bytes = create_dashboard_pdf(
            water_sample,
            probability,
            predicted_label,
            model_confidence,
            scores,
            drinking_penalties,
            skin_penalties,
            hair_penalties,
        )

    st.download_button(
        label="Download AquaSkin AI PDF Report",
        data=pdf_bytes,
        file_name="aquaskin_dashboard_report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

    st.warning(
        "The model was trained to estimate potability. "
        "The Drinking, Skin, and Hair scores are transparent "
        "rule-based MVP scores and are not learned medical outcomes."
    )