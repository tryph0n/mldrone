"""
Model comparison view - Compare model metrics.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from models.rfuavnet import RFUAVNet
from models.cnn import VGG16FC, ResNet50FC
from services.model_viz_service import ModelVisualizationService
from settings import COLORS, NUM_CLASSES, MODEL_INPUT_SIZES


COMPARE_CSV_PATH = Path(__file__).parent.parent / "media/model_comparison_results.csv"


class ModelComparisonView:
    """Model comparison page."""

    def __init__(self):
        self.df = self._load_results()

    def _load_results(self) -> pd.DataFrame:
        """Load results from CSV file."""
        if COMPARE_CSV_PATH.exists():
            df = pd.read_csv(COMPARE_CSV_PATH)
            # Normalize column names
            df.columns = [c.strip() for c in df.columns]
            return df

        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'Model', 'Features', 'Accuracy', 'F1-Score',
            'Inference_p50_ms', 'Inference_p95_ms', 'Inference_p99_ms', 'Model_Size_MB'
        ])

    def render(self):
        """Render the model comparison page."""
        st.header("Model Comparison")

        if self.df.empty:
            st.warning(
                "No model results found. Place `model_comparison_results.csv` in "
                "`interface/media/`."
            )
            st.info("Expected columns: Model, Features, Accuracy, F1-Score, "
                   "Inference_p50_ms, Inference_p95_ms, Model_Size_MB")
            return

        # Summary metrics
        self._render_summary_cards()

        st.divider()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            self._render_accuracy_chart()

        with col2:
            self._render_inference_time_chart()

        st.divider()

        # Detailed table
        self._render_detailed_table()

        st.divider()

        # Radar chart
        self._render_radar_chart()

        st.divider()

        # Architecture comparison
        self._render_architecture_comparison()

    def _render_summary_cards(self):
        """Render summary metric cards."""
        st.subheader("Summary")
        cols = st.columns(len(self.df))

        for col, (_, row) in zip(cols, self.df.iterrows()):
            model_name = row['Model']
            with col:
                color = COLORS.get(model_name, "#888888")
                st.markdown(f"""
                <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; text-align: center;">
                    <h3 style="color: {color}; margin: 0;">{model_name}</h3>
                    <p style="margin: 5px 0;"><strong>Accuracy:</strong> {row['Accuracy']*100:.1f}%</p>
                    <p style="margin: 5px 0;"><strong>F1-Score:</strong> {row['F1-Score']*100:.1f}%</p>
                    <p style="margin: 5px 0; font-size: 0.9em; color: #666;">{row['Features']}</p>
                </div>
                """, unsafe_allow_html=True)

    def _render_accuracy_chart(self):
        """Render accuracy comparison bar chart."""
        st.subheader("Accuracy & F1-Score")

        plot_data = []
        for _, row in self.df.iterrows():
            plot_data.append({"Model": row['Model'], "Metric": "Accuracy", "Value": row['Accuracy']})
            plot_data.append({"Model": row['Model'], "Metric": "F1-Score", "Value": row['F1-Score']})

        plot_df = pd.DataFrame(plot_data)

        fig = px.bar(
            plot_df,
            x="Model",
            y="Value",
            color="Metric",
            barmode="group",
            color_discrete_sequence=["#3498db", "#e74c3c"],
        )
        fig.update_layout(
            yaxis_title="Score",
            yaxis_tickformat=".0%",
            legend_title="",
            height=350,
        )
        st.plotly_chart(fig, width='stretch')

    def _render_inference_time_chart(self):
        """Render inference time comparison."""
        st.subheader("Inference Time (ms)")

        models = self.df['Model'].tolist()
        p50 = self.df['Inference_p50_ms'].tolist()
        p95 = self.df['Inference_p95_ms'].tolist()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="p50 (median)",
            x=models,
            y=p50,
            marker_color=[COLORS.get(m, "#888888") for m in models],
        ))
        fig.add_trace(go.Scatter(
            name="p95",
            x=models,
            y=p95,
            mode="markers",
            marker=dict(size=12, symbol="diamond", color="black"),
        ))
        fig.update_layout(
            yaxis_title="Time (ms)",
            height=350,
            showlegend=True,
        )
        st.plotly_chart(fig, width='stretch')

    def _render_detailed_table(self):
        """Render detailed metrics table."""
        st.subheader("Detailed Metrics")

        display_df = self.df.copy()
        display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
        display_df['F1-Score'] = display_df['F1-Score'].apply(lambda x: f"{x*100:.1f}%")
        display_df['Inference_p50_ms'] = display_df['Inference_p50_ms'].apply(lambda x: f"{x:.2f}")
        display_df['Inference_p95_ms'] = display_df['Inference_p95_ms'].apply(lambda x: f"{x:.2f}")
        display_df['Inference_p99_ms'] = display_df['Inference_p99_ms'].apply(lambda x: f"{x:.2f}")
        display_df['Model_Size_MB'] = display_df['Model_Size_MB'].apply(lambda x: f"{x:.1f}")

        display_df.columns = ['Model', 'Features', 'Accuracy', 'F1-Score',
                             'p50 (ms)', 'p95 (ms)', 'p99 (ms)', 'Size (MB)']

        st.dataframe(display_df, width='stretch', hide_index=True)

    def _render_radar_chart(self):
        """Render normalized radar chart."""
        st.subheader("Normalized Comparison (Radar)")

        max_time = self.df['Inference_p50_ms'].max() or 1
        max_size = self.df['Model_Size_MB'].max() or 1

        fig = go.Figure()
        categories = ["Accuracy", "F1-Score", "Speed", "Size (compact)"]

        for _, row in self.df.iterrows():
            model_name = row['Model']
            values = [
                row['Accuracy'],
                row['F1-Score'],
                1 - (row['Inference_p50_ms'] / max_time) if max_time > 0 else 0,
                1 - (row['Model_Size_MB'] / max_size) if max_size > 0 else 0,
            ]
            values.append(values[0])

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=model_name,
                line=dict(color=COLORS.get(model_name, "#888888")),
                fill="toself",
                opacity=0.3,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=600,
        )
        st.plotly_chart(fig, width='stretch')

        st.caption(
            "Speed and Size are inverted (higher = faster/smaller). "
            "All metrics normalized to 0-1 scale."
        )

    def _render_architecture_comparison(self):
        """Render model architecture comparison."""

        st.subheader("Architecture Comparison")

        # Get available PyTorch models from results
        pytorch_models = ['VGG16', 'ResNet50', 'RFUAVNet']
        available_models = [row['Model'] for _, row in self.df.iterrows()
                           if row['Model'] in pytorch_models]

        if not available_models:
            st.info("Architecture visualization available only for PyTorch models")
            return

        # Select models to visualize
        selected = st.multiselect(
            "Select models to compare",
            available_models,
            default=available_models[:2] if len(available_models) >= 2 else available_models
        )

        if not selected:
            return

        # Create columns for side-by-side comparison
        cols = st.columns(len(selected))
        viz_service = ModelVisualizationService()

        for col, model_name in zip(cols, selected):
            with col:
                # Header
                color = COLORS.get(model_name, "#888888")
                st.markdown(
                    f"<h4 style='color: {color};'>{model_name}</h4>",
                    unsafe_allow_html=True
                )

                # Load model instance
                model = self._get_model_instance(model_name)
                if model is None:
                    st.error("Model not available")
                    continue

                # Summary metrics
                summary = viz_service.get_simple_summary(model)
                st.metric("Parameters", f"{summary['total_params']:,}")
                st.metric("Size (MB)", f"{summary['size_mb']:.2f}")

                # Architecture details in expander
                with st.expander("Architecture", expanded=False):
                    st.text(summary['architecture_str'])

    def _get_model_instance(self, model_name: str):
        """Get model instance by name."""

        models = {
            'RFUAVNet': RFUAVNet(num_classes=NUM_CLASSES),
            'VGG16': VGG16FC(num_classes=NUM_CLASSES),
            'ResNet50': ResNet50FC(num_classes=NUM_CLASSES)
        }

        return models.get(model_name)
