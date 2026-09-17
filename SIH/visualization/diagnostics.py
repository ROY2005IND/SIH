"""
Diagnostic Visualization Engine for Residuals and Error Vectors.
"""
import numpy as np
import plotly.graph_objects as go

def plot_residual_histogram(
    residuals: np.ndarray | None,
    inlier_mask: np.ndarray | None = None,
    ransac_threshold: float = 3.0,
) -> go.Figure:
    """Generate interactive Plotly histogram showing geometric transfer residual distribution."""
    fig = go.Figure()

    if residuals is None or len(residuals) == 0:
        fig.add_annotation(text="No residual data available", showarrow=False)
        return fig

    if inlier_mask is not None:
        inlier_res = residuals[inlier_mask]
        outlier_res = residuals[~inlier_mask]

        fig.add_trace(go.Histogram(
            x=inlier_res,
            nbinsx=40,
            name=f"Inliers ({len(inlier_res)})",
            marker_color="#22c55e",
            opacity=0.75,
        ))
        if len(outlier_res) > 0:
            fig.add_trace(go.Histogram(
                x=outlier_res,
                nbinsx=40,
                name=f"Outliers ({len(outlier_res)})",
                marker_color="#ef4444",
                opacity=0.5,
            ))
    else:
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=40,
            name="Residuals",
            marker_color="#38bdf8",
        ))

    fig.add_vline(
        x=ransac_threshold,
        line_dash="dash",
        line_color="#facc15",
        annotation_text=f"RANSAC Threshold ({ransac_threshold:.1f}px)",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Geometric Transfer Residual Distribution",
        xaxis_title="Euclidean Error (Pixels)",
        yaxis_title="Match Count",
        barmode="overlay",
        paper_bgcolor="#070b14",
        plot_bgcolor="#0d1527",
        font={"color": "#94a3b8", "family": "JetBrains Mono"},
        margin=dict(l=30, r=30, t=40, b=30),
        height=300,
    )
    return fig

def plot_error_vectors(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inlier_mask: np.ndarray | None,
    image_shape: tuple[int, int],
) -> go.Figure:
    """Plot quiver/vector error field showing directional geometric distortion."""
    fig = go.Figure()
    h, w = image_shape[:2]

    if len(src_pts) == 0:
        fig.add_annotation(text="No vectors to display", showarrow=False)
        return fig

    mask = inlier_mask if inlier_mask is not None else np.ones(len(src_pts), dtype=bool)
    inliers = src_pts[mask]
    outliers = src_pts[~mask]

    # Plot Inliers
    fig.add_trace(go.Scatter(
        x=inliers[:, 0],
        y=inliers[:, 1],
        mode="markers",
        marker=dict(color="#22c55e", size=6, symbol="circle"),
        name="Verified Inliers",
    ))

    # Plot Outliers
    if len(outliers) > 0:
        fig.add_trace(go.Scatter(
            x=outliers[:, 0],
            y=outliers[:, 1],
            mode="markers",
            marker=dict(color="#ef4444", size=5, symbol="x"),
            name="Rejected Outliers",
        ))

    fig.update_layout(
        title="Spatial Distribution of Correspondences",
        xaxis=dict(range=[0, w], title="Image X (Pixels)"),
        yaxis=dict(range=[h, 0], title="Image Y (Pixels)", scaleanchor="x", scaleratio=1),
        paper_bgcolor="#070b14",
        plot_bgcolor="#0d1527",
        font={"color": "#94a3b8", "family": "JetBrains Mono"},
        margin=dict(l=30, r=30, t=40, b=30),
        height=380,
    )
    return fig
