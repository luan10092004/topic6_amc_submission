from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from scipy.ndimage import zoom
from scipy.signal import stft


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

IMAGE_DIR = (
    DATA_DIR
    / "images"
)

DEMO_DATA_DIR = (
    ROOT
    / "data"
    / "demo"
)

DEVICE = torch.device("cpu")

IMAGE_SIZE = 32


# =========================================================
# 2. MODEL ARCHITECTURES
# Must remain identical to training architectures
# =========================================================

class CNN1D(nn.Module):

    def __init__(
        self,
        num_classes=8,
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                2,
                32,
                kernel_size=7,
                padding=3,
            ),

            nn.BatchNorm1d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                2
            ),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                2
            ),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(
                1
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        x = self.features(
            x
        )

        return self.classifier(
            x
        )


class CNN2D(nn.Module):

    def __init__(
        self,
        num_classes=8,
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                16
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        x = self.features(
            x
        )

        return self.classifier(
            x
        )


# =========================================================
# 3. ON-THE-FLY SPECTROGRAM
# Must match dataset-generation preprocessing
# =========================================================

def make_spectrogram(
    iq,
):

    """
    Convert one I/Q burst [2, 128]
    to a normalized 32 x 32 spectrogram.
    """

    I = iq[0].astype(
        np.float32
    )

    Q = iq[1].astype(
        np.float32
    )

    complex_signal = (
        I
        + 1j * Q
    )

    _, _, Z = stft(
        complex_signal,
        fs=1.0,
        nperseg=32,
        noverlap=16,
        nfft=32,
        boundary=None,
        padded=False,
        return_onesided=False,
    )

    power = (
        np.abs(Z) ** 2
    )

    spec_db = (
        10.0
        * np.log10(
            power
            + 1e-12
        )
    )

    # -----------------------------------------------------
    # Normalize each spectrogram to [0, 1]
    # -----------------------------------------------------

    spec_db = (
        spec_db
        - spec_db.min()
    )

    max_value = (
        spec_db.max()
    )

    if max_value > 0:

        spec_db = (
            spec_db
            / max_value
        )

    # -----------------------------------------------------
    # Resize to 32 x 32
    # -----------------------------------------------------

    zoom_factors = (

        IMAGE_SIZE
        / spec_db.shape[0],

        IMAGE_SIZE
        / spec_db.shape[1],
    )

    resized = zoom(
        spec_db,
        zoom_factors,
        order=1,
    )

    # -----------------------------------------------------
    # Safety: guarantee exact 32 x 32
    # -----------------------------------------------------

    result = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=np.float32,
    )

    h = min(
        IMAGE_SIZE,
        resized.shape[0],
    )

    w = min(
        IMAGE_SIZE,
        resized.shape[1],
    )

    result[
        :h,
        :w,
    ] = resized[
        :h,
        :w,
    ]

    return result


# =========================================================
# 4. ON-THE-FLY CONSTELLATION DENSITY
# Must match dataset-generation preprocessing
# =========================================================

def make_constellation_density(
    iq,
):

    """
    Convert one I/Q burst [2, 128]
    to a normalized 32 x 32
    constellation-density image.
    """

    I = iq[0].astype(
        np.float32
    )

    Q = iq[1].astype(
        np.float32
    )

    amplitude = np.sqrt(
        I ** 2
        + Q ** 2
    )

    scale = (
        amplitude.max()
    )

    if scale > 1e-12:

        I = (
            I / scale
        )

        Q = (
            Q / scale
        )

    hist, _, _ = np.histogram2d(
        I,
        Q,
        bins=IMAGE_SIZE,
        range=[
            [-1, 1],
            [-1, 1],
        ],
    )

    hist = np.log1p(
        hist
    )

    max_value = (
        hist.max()
    )

    if max_value > 0:

        hist = (
            hist
            / max_value
        )

    return hist.T.astype(
        np.float32
    )


# =========================================================
# 5. LOAD DATA
#
# MODE A:
# Full dataset exists locally
#
# MODE B:
# Lightweight GitHub demo bundle
# =========================================================

@st.cache_resource
def load_data():

    # -----------------------------------------------------
    # Full-data files
    # -----------------------------------------------------

    full_data_file = (
        DATA_DIR
        / "radio_ml_split.npz"
    )

    full_spec_file = (
        IMAGE_DIR
        / "spectrograms.npy"
    )

    full_density_file = (
        IMAGE_DIR
        / "constellation_density.npy"
    )

    # -----------------------------------------------------
    # Lightweight GitHub bundle
    # -----------------------------------------------------

    demo_bundle_file = (
        DEMO_DATA_DIR
        / "demo_samples.npz"
    )

    # -----------------------------------------------------
    # Optional environment variable:
    #
    # AMC_DEMO_MODE=quick
    #
    # Forces lightweight mode even when full data exists.
    # -----------------------------------------------------

    force_quick_demo = (
        os.getenv(
            "AMC_DEMO_MODE",
            "",
        )
        .strip()
        .lower()
        == "quick"
    )

    full_data_available = (

        full_data_file.exists()

        and full_spec_file.exists()

        and full_density_file.exists()
    )

    # =====================================================
    # MODE A — FULL TEST DATA
    # =====================================================

    if (
        full_data_available
        and not force_quick_demo
    ):

        raw = np.load(
            full_data_file,
            allow_pickle=True,
        )

        X = raw[
            "X"
        ]

        y = raw[
            "y"
        ]

        snrs = raw[
            "snrs"
        ]

        test_idx = np.asarray(
            raw[
                "test_idx"
            ],
            dtype=np.int64,
        )

        class_names = [

            str(name)

            for name
            in raw[
                "class_names"
            ]
        ]

        spec_images = np.load(
            full_spec_file,
            mmap_mode="r",
        )

        density_images = np.load(
            full_density_file,
            mmap_mode="r",
        )

        # In full mode these are already
        # global dataset indices.
        display_indices = (
            test_idx.copy()
        )

        source_mode = (
            "Full test set"
        )

        return (
            X,
            y,
            snrs,
            test_idx,
            class_names,
            spec_images,
            density_images,
            display_indices,
            source_mode,
        )

    # =====================================================
    # MODE B — QUICK GITHUB DEMO
    # =====================================================

    if not demo_bundle_file.exists():

        raise FileNotFoundError(

            "Neither the full dataset nor "
            "data/demo/demo_samples.npz was found."
        )

    demo = np.load(
        demo_bundle_file,
        allow_pickle=True,
    )

    X = np.asarray(
        demo[
            "X"
        ],
        dtype=np.float32,
    )

    y = np.asarray(
        demo[
            "y"
        ]
    )

    snrs = np.asarray(
        demo[
            "snrs"
        ]
    )

    class_names = [

        str(name)

        for name
        in demo[
            "class_names"
        ]
    ]

    spec_images = np.asarray(
        demo[
            "spectrograms"
        ],
        dtype=np.float32,
    )

    density_images = np.asarray(
        demo[
            "constellation_density"
        ],
        dtype=np.float32,
    )

    # Original full-dataset indices,
    # retained for traceability.
    display_indices = np.asarray(
        demo[
            "original_indices"
        ],
        dtype=np.int64,
    )

    # Every sample in the bundle is a test sample.
    # Therefore bundle-local indices become test_idx.
    test_idx = np.arange(
        len(X),
        dtype=np.int64,
    )

    source_mode = (
        "Quick GitHub demo bundle"
    )

    return (
        X,
        y,
        snrs,
        test_idx,
        class_names,
        spec_images,
        density_images,
        display_indices,
        source_mode,
    )


# =========================================================
# 6. LOAD MODELS
# =========================================================

def load_model(
    model_class,
    filename,
    num_classes,
):

    path = (
        MODEL_DIR
        / filename
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find model: {path}"
        )

    model = model_class(
        num_classes=num_classes
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True,
    )

    # Compatibility if a checkpoint is ever
    # stored as {"state_dict": ...}
    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "state_dict"
        in checkpoint
    ):

        checkpoint = (
            checkpoint[
                "state_dict"
            ]
        )

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


@st.cache_resource
def load_models(
    num_classes,
):

    iq_model = load_model(
        CNN1D,
        "iq_cnn1d.pt",
        num_classes,
    )

    spec_model = load_model(
        CNN2D,
        "spectrogram_cnn2d.pt",
        num_classes,
    )

    density_model = load_model(
        CNN2D,
        "density_cnn2d.pt",
        num_classes,
    )

    return (
        iq_model,
        spec_model,
        density_model,
    )


# =========================================================
# 7. LOAD FINAL BENCHMARK RESULTS
# =========================================================

@st.cache_data
def load_final_results():

    path = (
        MODEL_DIR
        / "final_results_complete.csv"
    )

    if not path.exists():

        return None

    return pd.read_csv(
        path
    )


# =========================================================
# 8. PREDICTION — RAW I/Q
# =========================================================

def predict_iq(
    model,
    iq,
):

    tensor = torch.from_numpy(

        np.array(
            iq[
                None,
                :,
                :,
            ],
            dtype=np.float32,
            copy=True,
        )

    ).to(
        DEVICE
    )

    with torch.inference_mode():

        logits = model(
            tensor
        )

        probabilities = (

            torch.softmax(
                logits,
                dim=1,
            )[0]

            .cpu()

            .numpy()
        )

    prediction = int(
        np.argmax(
            probabilities
        )
    )

    return (
        prediction,
        probabilities,
    )


# =========================================================
# 9. PREDICTION — 2D IMAGE
# =========================================================

def predict_image(
    model,
    image,
):

    tensor = torch.from_numpy(

        np.array(
            image[
                None,
                None,
                :,
                :,
            ],
            dtype=np.float32,
            copy=True,
        )

    ).to(
        DEVICE
    )

    with torch.inference_mode():

        logits = model(
            tensor
        )

        probabilities = (

            torch.softmax(
                logits,
                dim=1,
            )[0]

            .cpu()

            .numpy()
        )

    prediction = int(
        np.argmax(
            probabilities
        )
    )

    return (
        prediction,
        probabilities,
    )


# =========================================================
# 10. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title=(
        "AMC Multi-Domain Demo"
    ),
    layout="wide",
)


st.title(
    "Automatic Modulation Classification"
)

st.subheader(
    "Multi-Domain 1D / 2D Signal Representation"
)

st.write(
    """
This interactive demo compares three AMC pipelines:

**Raw I/Q → 1D CNN**

**Raw I/Q → STFT Spectrogram → 2D CNN**

**Raw I/Q → Constellation Density → 2D CNN**
"""
)

st.caption(
    "The Spectrogram and Constellation Density "
    "used for inference are generated on-the-fly "
    "from the currently selected raw I/Q burst."
)


# =========================================================
# 11. LOAD PROJECT FILES
# =========================================================

try:

    (
        X,
        y,
        snr_values,
        test_idx,
        class_names,
        saved_spec_images,
        saved_density_images,
        display_indices,
        data_source_mode,
    ) = load_data()

    num_classes = len(
        class_names
    )

    (
        iq_model,
        spec_model,
        density_model,
    ) = load_models(
        num_classes
    )

except Exception as error:

    st.error(
        "Error loading project files:"
    )

    st.exception(
        error
    )

    st.stop()


# =========================================================
# 12. DATA MODE INFORMATION
# =========================================================

if (
    data_source_mode
    == "Full test set"
):

    st.success(
        "Data mode: Full test set "
        f"({len(test_idx):,} unseen test bursts)"
    )

else:

    st.info(
        "Data mode: Quick GitHub demo bundle "
        f"({len(test_idx):,} balanced unseen test bursts)"
    )


# =========================================================
# 13. SIDEBAR — TEST BURST SELECTION
# =========================================================

st.sidebar.header(
    "Test Burst Selection"
)

st.sidebar.write(
    f"Mode: **{data_source_mode}**"
)


# ---------------------------------------------------------
# Initialize selected position
# ---------------------------------------------------------

if (
    "burst_position"
    not in st.session_state
):

    st.session_state[
        "burst_position"
    ] = 0


# Defensive check if the available dataset changes
if (
    st.session_state[
        "burst_position"
    ]
    >= len(test_idx)
):

    st.session_state[
        "burst_position"
    ] = 0


# ---------------------------------------------------------
# SNR values available in current test data
# ---------------------------------------------------------

test_snrs = (
    snr_values[
        test_idx
    ]
)

unique_test_snrs = sorted(
    {
        int(value)
        for value
        in test_snrs
    }
)


# ---------------------------------------------------------
# Random-burst SNR filter
# ---------------------------------------------------------

snr_filter = st.sidebar.selectbox(
    "Random-burst SNR filter",
    options=[
        "Any"
    ]
    + unique_test_snrs,
)


st.sidebar.caption(
    "This filter selects an existing test burst. "
    "It does not add synthetic noise or modify the signal."
)


# ---------------------------------------------------------
# Random test burst
# ---------------------------------------------------------

if st.sidebar.button(
    "Random test burst"
):

    if (
        snr_filter
        == "Any"
    ):

        candidate_positions = (
            np.arange(
                len(test_idx)
            )
        )

    else:

        candidate_positions = (
            np.flatnonzero(
                test_snrs
                == float(
                    snr_filter
                )
            )
        )

    if (
        len(
            candidate_positions
        )
        > 0
    ):

        rng = (
            np.random.default_rng()
        )

        st.session_state[
            "burst_position"
        ] = int(
            rng.choice(
                candidate_positions
            )
        )


# ---------------------------------------------------------
# Manual burst selection
# ---------------------------------------------------------

position = st.sidebar.number_input(
    "Position in test set",
    min_value=0,
    max_value=(
        len(test_idx)
        - 1
    ),
    step=1,
    key="burst_position",
)

position = int(
    position
)


# =========================================================
# 14. RESOLVE SELECTED BURST
# =========================================================

sample_index = int(
    test_idx[
        position
    ]
)

# Global index in original full dataset
original_dataset_index = int(
    display_indices[
        position
    ]
)

signal = np.array(
    X[
        sample_index
    ],
    dtype=np.float32,
    copy=True,
)

true_label_id = int(
    y[
        sample_index
    ]
)

true_label = (
    class_names[
        true_label_id
    ]
)

sample_snr = float(
    snr_values[
        sample_index
    ]
)


# =========================================================
# 15. CREATE REPRESENTATIONS ON-THE-FLY
# =========================================================

spectrogram = make_spectrogram(
    signal
)

density = (
    make_constellation_density(
        signal
    )
)


# =========================================================
# 16. VERIFY AGAINST SAVED REPRESENTATIONS
# =========================================================

saved_spectrogram = np.array(
    saved_spec_images[
        sample_index
    ],
    dtype=np.float32,
    copy=True,
)

saved_density = np.array(
    saved_density_images[
        sample_index
    ],
    dtype=np.float32,
    copy=True,
)


spec_difference = float(

    np.max(
        np.abs(
            spectrogram
            - saved_spectrogram
        )
    )
)


density_difference = float(

    np.max(
        np.abs(
            density
            - saved_density
        )
    )
)


# =========================================================
# 17. RUN THREE MODELS
# =========================================================

iq_prediction, iq_probs = (
    predict_iq(
        iq_model,
        signal,
    )
)

spec_prediction, spec_probs = (
    predict_image(
        spec_model,
        spectrogram,
    )
)

density_prediction, density_probs = (
    predict_image(
        density_model,
        density,
    )
)


# =========================================================
# 18. SELECTED BURST INFORMATION
# =========================================================

st.header(
    "Selected Test Burst"
)

col1, col2, col3, col4 = (
    st.columns(
        4
    )
)


col1.metric(
    "Test position",
    position,
)

col2.metric(
    "Original dataset index",
    original_dataset_index,
)

col3.metric(
    "SNR",
    f"{sample_snr:.0f} dB",
)

col4.metric(
    "True modulation",
    true_label,
)


# =========================================================
# 19. MODEL PREDICTIONS
# =========================================================

st.header(
    "Model Predictions"
)

p1, p2, p3 = (
    st.columns(
        3
    )
)


with p1:

    st.subheader(
        "IQ CNN 1D"
    )

    st.metric(
        "Prediction",
        class_names[
            iq_prediction
        ],
    )

    st.write(
        "Confidence: "
        f"{iq_probs[iq_prediction] * 100:.2f}%"
    )

    if (
        iq_prediction
        == true_label_id
    ):

        st.success(
            "Correct"
        )

    else:

        st.error(
            "Incorrect"
        )


with p2:

    st.subheader(
        "Spectrogram CNN 2D"
    )

    st.metric(
        "Prediction",
        class_names[
            spec_prediction
        ],
    )

    st.write(
        "Confidence: "
        f"{spec_probs[spec_prediction] * 100:.2f}%"
    )

    if (
        spec_prediction
        == true_label_id
    ):

        st.success(
            "Correct"
        )

    else:

        st.error(
            "Incorrect"
        )


with p3:

    st.subheader(
        "Constellation Density CNN 2D"
    )

    st.metric(
        "Prediction",
        class_names[
            density_prediction
        ],
    )

    st.write(
        "Confidence: "
        f"{density_probs[density_prediction] * 100:.2f}%"
    )

    if (
        density_prediction
        == true_label_id
    ):

        st.success(
            "Correct"
        )

    else:

        st.error(
            "Incorrect"
        )


# =========================================================
# 20. RAW I/Q WAVEFORM
# =========================================================

st.header(
    "Raw I/Q Signal"
)

fig, ax = plt.subplots(
    figsize=(10, 3)
)

ax.plot(
    signal[0],
    label="I",
)

ax.plot(
    signal[1],
    label="Q",
)

ax.set_xlabel(
    "Sample"
)

ax.set_ylabel(
    "Amplitude"
)

ax.legend()

ax.grid(
    alpha=0.3
)

st.pyplot(
    fig
)

plt.close(
    fig
)


# =========================================================
# 21. RAW CONSTELLATION
# =========================================================

st.header(
    "Raw Constellation"
)

fig, ax = plt.subplots(
    figsize=(5, 5)
)

ax.scatter(
    signal[0],
    signal[1],
    s=18,
    alpha=0.7,
)

ax.set_xlabel(
    "I"
)

ax.set_ylabel(
    "Q"
)

ax.set_title(
    f"{true_label} | "
    f"SNR {sample_snr:.0f} dB"
)

ax.grid(
    alpha=0.3
)

st.pyplot(
    fig
)

plt.close(
    fig
)


# =========================================================
# 22. 2D SIGNAL REPRESENTATIONS
# =========================================================

st.header(
    "2D Signal Representations"
)

st.caption(
    "Both images below are generated directly "
    "from the selected raw I/Q burst."
)

c1, c2 = (
    st.columns(
        2
    )
)


with c1:

    st.subheader(
        "Spectrogram 32 × 32"
    )

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    image_plot = ax.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Frequency bin"
    )

    fig.colorbar(
        image_plot,
        ax=ax,
        fraction=0.046,
    )

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


with c2:

    st.subheader(
        "Constellation Density 32 × 32"
    )

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    image_plot = ax.imshow(
        density,
        origin="lower",
        extent=[
            -1,
            1,
            -1,
            1,
        ],
        aspect="equal",
    )

    ax.set_xlabel(
        "I"
    )

    ax.set_ylabel(
        "Q"
    )

    fig.colorbar(
        image_plot,
        ax=ax,
        fraction=0.046,
    )

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


# =========================================================
# 23. PIPELINE VERIFICATION
# =========================================================

st.header(
    "Pipeline Verification"
)

with st.expander(
    "Compare on-the-fly images with saved training representations",
    expanded=True,
):

    v1, v2 = (
        st.columns(
            2
        )
    )

    with v1:

        st.write(
            "Spectrogram maximum absolute difference"
        )

        st.code(
            f"{spec_difference:.10f}"
        )


    with v2:

        st.write(
            "Constellation Density maximum absolute difference"
        )

        st.code(
            f"{density_difference:.10f}"
        )


    if (
        spec_difference
        < 1e-5

        and density_difference
        < 1e-5
    ):

        st.success(
            "On-the-fly preprocessing matches "
            "the saved representations."
        )

    else:

        st.warning(
            "The on-the-fly representation does not "
            "exactly match the saved representation. "
            "Check the dataset-generation preprocessing."
        )


# =========================================================
# 24. CLASS PROBABILITIES
# =========================================================

st.header(
    "Class Probabilities"
)

tab1, tab2, tab3 = st.tabs(
    [
        "IQ CNN 1D",
        "Spectrogram CNN 2D",
        "Density CNN 2D",
    ]
)


def show_probabilities(
    probabilities,
):

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.bar(
        class_names,
        probabilities,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "Probability"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


with tab1:

    show_probabilities(
        iq_probs
    )


with tab2:

    show_probabilities(
        spec_probs
    )


with tab3:

    show_probabilities(
        density_probs
    )


# =========================================================
# 25. PREDICTION SUMMARY
# =========================================================

st.header(
    "Prediction Summary"
)


summary_df = pd.DataFrame(
    {
        "Model": [
            "IQ CNN 1D",
            "Spectrogram CNN 2D",
            "Constellation Density CNN 2D",
        ],

        "Input": [
            "Raw I/Q",
            "On-the-fly Spectrogram",
            "On-the-fly Density",
        ],

        "Prediction": [
            class_names[
                iq_prediction
            ],

            class_names[
                spec_prediction
            ],

            class_names[
                density_prediction
            ],
        ],

        "Confidence (%)": [

            round(
                iq_probs[
                    iq_prediction
                ]
                * 100,
                2,
            ),

            round(
                spec_probs[
                    spec_prediction
                ]
                * 100,
                2,
            ),

            round(
                density_probs[
                    density_prediction
                ]
                * 100,
                2,
            ),
        ],

        "Correct": [

            (
                iq_prediction
                == true_label_id
            ),

            (
                spec_prediction
                == true_label_id
            ),

            (
                density_prediction
                == true_label_id
            ),
        ],
    }
)


st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 26. PROCESSING PIPELINE
# =========================================================

st.header(
    "Processing Pipeline"
)

st.code(
    """
Raw I/Q burst [2 × 128]
        |
        |-----------------> 1D CNN -----------------> Prediction
        |
        +--> STFT
        |     |
        |     +--> Spectrogram [32 × 32]
        |               |
        |               +--> 2D CNN --> Prediction
        |
        +--> I/Q Histogram
              |
              +--> Constellation Density [32 × 32]
                              |
                              +--> 2D CNN --> Prediction
"""
)


# =========================================================
# 27. FINAL BENCHMARK SUMMARY
# =========================================================

st.divider()

st.header(
    "Final Benchmark Summary"
)

st.caption(
    "Final controlled evaluation. "
    "These values are fixed aggregate benchmark results "
    "and therefore do not change when another demo burst "
    "is selected."
)


benchmark_df = (
    load_final_results()
)


if benchmark_df is None:

    st.warning(
        "Cannot find "
        "models/final_results_complete.csv"
    )

else:

    display_df = (
        benchmark_df.copy()
    )

    # -----------------------------------------------------
    # Accuracy / Macro-F1 → %
    # -----------------------------------------------------

    display_df[
        "Accuracy"
    ] = (

        display_df[
            "Accuracy"
        ]
        * 100

    ).map(
        lambda value:
        f"{value:.2f}%"
    )


    display_df[
        "Macro-F1"
    ] = (

        display_df[
            "Macro-F1"
        ]
        * 100

    ).map(
        lambda value:
        f"{value:.2f}%"
    )


    # -----------------------------------------------------
    # Latency formatting
    # -----------------------------------------------------

    latency_columns = [

        "Inference Median (ms/sample)",

        "Inference P95 (ms/sample)",

        "E2E Median (ms/sample)",

        "E2E P95 (ms/sample)",
    ]


    for column in latency_columns:

        if (
            column
            in display_df.columns
        ):

            display_df[
                column
            ] = display_df[
                column
            ].map(
                lambda value:
                f"{value:.4f}"
            )


    # -----------------------------------------------------
    # Parameter formatting
    # -----------------------------------------------------

    if (
        "Parameters"
        in display_df.columns
    ):

        display_df[
            "Parameters"
        ] = display_df[
            "Parameters"
        ].map(
            lambda value:
            f"{int(value):,}"
        )


    # -----------------------------------------------------
    # Model file size formatting
    # -----------------------------------------------------

    if (
        "Model Size (MB)"
        in display_df.columns
    ):

        display_df[
            "Model Size (MB)"
        ] = display_df[
            "Model Size (MB)"
        ].map(
            lambda value:
            f"{value:.5f}"
        )


    # -----------------------------------------------------
    # Tensor-memory formatting
    # -----------------------------------------------------

    if (
        "Model Tensor Memory (MB)"
        in display_df.columns
    ):

        display_df[
            "Model Tensor Memory (MB)"
        ] = display_df[
            "Model Tensor Memory (MB)"
        ].map(
            lambda value:
            f"{value:.5f}"
        )


    # -----------------------------------------------------
    # Shorter display column names
    # -----------------------------------------------------

    display_df = display_df.rename(
        columns={

            "Inference Median (ms/sample)":
                "Inference Median (ms)",

            "Inference P95 (ms/sample)":
                "Inference P95 (ms)",

            "E2E Median (ms/sample)":
                "E2E Median (ms)",

            "E2E P95 (ms/sample)":
                "E2E P95 (ms)",

            "Model Tensor Memory (MB)":
                "Tensor Memory (MB)",
        }
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


    st.info(
        "Inference-only latency measures only the "
        "neural-network forward pass on prepared tensors. "
        "End-to-end latency additionally includes "
        "representation preprocessing such as STFT or "
        "constellation-density generation."
    )


    # -----------------------------------------------------
    # Main technical conclusion
    # -----------------------------------------------------

    st.subheader(
        "Engineering Trade-off"
    )

    st.markdown(
        """
- **IQ CNN 1D** achieves the highest overall Accuracy and Macro-F1, while also providing the lowest inference-only and end-to-end latency.
- **Constellation Density CNN 2D** outperforms the Spectrogram CNN 2D in overall Accuracy and Macro-F1.
- Both **2D CNN models use fewer parameters and less model tensor memory** than the IQ CNN 1D.
- The **2D pipelines have higher end-to-end latency** because the 2D representations must be generated before CNN inference.
- In this experiment, converting the 1D I/Q burst to a 2D representation **did not improve overall classification accuracy**, although Constellation Density becomes competitive at high SNR.
"""
    )


# =========================================================
# 28. DEMO / REPRODUCIBILITY NOTE
# =========================================================

st.divider()

st.caption(
    "Reproducibility note: saved Spectrogram and "
    "Constellation Density arrays are used only to verify "
    "the preprocessing pipeline. Predictions for the 2D "
    "models use representations generated on-the-fly from "
    "the currently selected raw I/Q burst."
)