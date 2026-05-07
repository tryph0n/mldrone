"""
Home page view - Project presentation.
"""

from pathlib import Path

import streamlit as st

from settings import DRONE_INFO


class HomeView:
    """Home page with project overview."""

    def render(self):
        """Render the home page."""
        st.header("DroneDetect V2 - RF-based Drone Classification")

        st.subheader("Project Overview")
        st.markdown("""
        This project implements **drone classification using RF (Radio Frequency) signals**
        captured at 2.4 GHz. The goal is to identify drone models based on their unique
        transmission signatures.

        **Dataset**: DroneDetect V2 (Swinney & Woods, 2021)

        **Task**: 7-class drone type classification from 20ms RF segments
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Key Findings")
            st.markdown("""
            - **PSD features outperform spectrograms** for robustness against
              WiFi/Bluetooth interference
            - **File-level split is critical** to prevent data leakage
              (segments from same file must stay together)
            - **OcuSync vs WiFi protocols** show distinct spectral signatures
            - **MA1/MAV confusion** is expected (both use similar OcuSync variants)
            - **RF-UAVNet limited by 120x downsampling**: 1.2M to 10K samples per segment causes major information loss (~63% accuracy vs ~89% for spectrograms)
            """)

        with col2:
            st.subheader("Methodology Choices")
            st.markdown("""
            - **20ms segments** (1.2M samples at 60 MHz sample rate)
            - **Welch PSD**: 1024-point FFT, Hanning window, noverlap=120
            - **Spectrograms**: 224x224 RGB (Viridis colormap)
            - **Stratified file-level split** to prevent leakage
            - **4 interference conditions**: CLEAN, WIFI, BLUE, BOTH
            """)

        st.divider()

        st.subheader("Models Implemented")
        cols = st.columns(4)

        models_info = [
            ("SVM", "PSD features", "Hand-crafted frequency domain", "#3498db"),
            ("VGG16", "Spectrograms", "Transfer learning (frozen)", "#e74c3c"),
            ("ResNet50", "Spectrograms", "Transfer learning (frozen)", "#2ecc71"),
            ("RFUAVNet", "Raw IQ", "1D CNN specialized for RF", "#9b59b6"),
        ]

        for col, (name, features, desc, color) in zip(cols, models_info):
            with col:
                st.markdown(
                    f"""
                <div style="border-left: 4px solid {color}; padding-left: 10px;">
                    <h4>{name}</h4>
                    <p><strong>Input:</strong> {features}</p>
                    <p><em>{desc}</em></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.divider()

        st.subheader("Research References")
        st.markdown("""
        | Paper | Contribution |
        |-------|--------------|
        | [Swinney & Woods, 2021](https://doi.org/10.3390/aerospace8070179) | DroneDetect dataset, PSD vs spectrogram comparison |
        | [RF-UAVNet (IEEE, 2022)](https://ieeexplore.ieee.org/document/9768809) | 1D CNN architecture for drone classification (raw IQ signals) |
        | [ Kiliç et al., 2021](https://doi.org/10.1016/j.jestch.2021.06.008) | RF signal classification with machine learning |
        | [ Swinney et al., 2021](https://doi.org/10.3390/aerospace8030079) | Flying mode classification via ResNet50 |
        """)

        st.subheader("Pitfalls to Avoid")
        st.error("""
        **Data Leakage Warning**: Random segment-level splits cause artificial 99%+ accuracy.\n
        Segments from the same 2-second recording share temporal correlations.\n
        Always use **file-level stratified splits**.
        """)

        st.warning("""
        **[RFClassification](https://github.com/IQTLabs/RFClassification) repository results (99.8% accuracy)** are inflated due to:
        1. Global normalization BEFORE train/test split
        2. Segment-level split instead of file-level
        3. No file boundary tracking
        """)

        st.divider()

        st.subheader("Dataset: DroneDetect V2")

        drone_data = {
            "Code": list(DRONE_INFO.keys()),
            "Drone": [d["name"] for d in DRONE_INFO.values()],
            "Protocol": [d["protocol"] for d in DRONE_INFO.values()],
            "Weight (g)": [d["weight"] for d in DRONE_INFO.values()],
        }
        st.table(drone_data)

        st.caption("""
        **Interference conditions**: CLEAN (no interference), WIFI (WiFi only), BLUE (Bluetooth only), BOTH (WiFi + Bluetooth)
        \n**Center frequency**: 2.4375 GHz
        \n**Bandwidth**: 28 MHz
        \n**Sample rate**: 60 MHz
        """)

        st.divider()

        st.subheader("Pipeline Architecture")

        pipeline_option = st.selectbox(
            "Select pipeline to view:",
            ["Global Overview", "SVM Pipeline", "CNN Pipeline", "RF-UAVNet Pipeline"],
        )

        diagrams_path = Path(__file__).parent.parent.parent / "docs" / "diagrams"

        pipeline_files = {
            "Global Overview": "pipeline_global.dot",
            "SVM Pipeline": "pipeline_svm.dot",
            "CNN Pipeline": "pipeline_cnn.dot",
            "RF-UAVNet Pipeline": "pipeline_rfuavnet.dot",
        }

        dot_file = diagrams_path / pipeline_files[pipeline_option]

        if dot_file.exists():
            dot_content = dot_file.read_text()
            st.graphviz_chart(dot_content)
        else:
            st.warning(f"Diagram not found: {dot_file}")

        descriptions = {
            "Global Overview": "Raw IQ file → segmentation → PSD / spectrogram / IQ features → 4 models → drone class",
            "SVM Pipeline": "IQ → Z-score → Welch PSD (1024, overlap 512) → fftshift → max norm → SVM (RBF)",
            "CNN Pipeline": "IQ → Z-score → STFT(1024, hop 904) → fftshift → resize 224x224 → dB → min-max → Viridis RGB → VGG16/ResNet50",
            "RF-UAVNet Pipeline": "IQ → min-max [0,1] → np.interp 10K (120x) → RF-UAVNet 1D CNN (grouped conv + residual, 4615 params)",
        }
        st.info(descriptions[pipeline_option])
