import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0b1930 55%, #071827 100%);
        color: #edf4ff;
    }
    [data-testid="stSidebar"] {
        background: #091426;
        border-right: 1px solid #183252;
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border: 1px solid #1d3b60;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(25,53,91,.85), rgba(11,35,55,.8));
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; color: #f4f8ff; font-size: 2.1rem; }
    .hero p { color: #a9bdd8; margin: .4rem 0 0 0; }
    div[data-testid="stMetric"] {
        background: rgba(17, 37, 63, .82);
        border: 1px solid #1b3b5f;
        border-radius: 14px;
        padding: 14px 16px;
    }
    div[data-testid="stForm"] {
        background: rgba(15, 32, 55, .65);
        border: 1px solid #1b3b5f;
        border-radius: 16px;
        padding: 1rem;
    }
    .section-label {
        color: #9fc4ff;
        font-size: 1.05rem;
        font-weight: 700;
        margin: .5rem 0 .75rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model and data helpers
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path, modified_time):
    return joblib.load(model_path)


def engineer_features(data):
    """Create the six features expected by the example app."""
    result = data.copy()

    if "AVG_MONTHLY_SPENDING" not in result.columns:
        # Assumption: monthly spending is purchases divided by tenure.
        result["AVG_MONTHLY_SPENDING"] = (
            result["PURCHASES"] / result["TENURE"].replace(0, np.nan)
        )

    if "CREDIT_UTILIZATION" not in result.columns:
        result["CREDIT_UTILIZATION"] = (
            result["BALANCE"] / result["CREDIT_LIMIT"].replace(0, np.nan)
        )

    return result


FEATURES = [
    "BALANCE",
    "PURCHASES_FREQUENCY",
    "CREDIT_LIMIT",
    "PRC_FULL_PAYMENT",
    "AVG_MONTHLY_SPENDING",
    "CREDIT_UTILIZATION",
]

CLUSTER_NAMES = {
    0: "Average Customers",
    1: "Frequent Buyers & Full Payers",
    2: "High-Balance, High-Spending Customers",
}


def cluster_name(value):
    try:
        return CLUSTER_NAMES.get(int(value), f"Cluster {value}")
    except (TypeError, ValueError):
        return f"Cluster {value}"


def get_model_prediction(model, feature_frame):
    return model.predict(feature_frame[FEATURES])


def add_cluster_names(data):
    data = data.copy()
    data["Cluster_Name"] = data["Cluster"].apply(cluster_name)
    return data


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 💳 Customer Segmentation")
    st.caption("Understand customer behavior. Find useful segments.")
    page = st.radio(
        "NAVIGATION",
        ["Predict Customer", "Visualizations", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Project status")
    st.caption("Model-driven predictions and dataset-based insights")
    st.caption("Better insights. Happier customers. Smarter decisions.")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>💳 Customer Segmentation</h1>
      <p>Predict a customer's segment and explore patterns across your customer base.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
model_path = "best_pipeline.joblib"

try:
    import os
    if not os.path.exists(model_path):
        st.error(
            "Couldn't find `best_pipeline.joblib`. Put it in the same folder as `app.py`."
        )
        st.stop()
    model = load_model(model_path, os.path.getmtime(model_path))
except Exception as exc:
    st.error(f"Couldn't load the model: {exc}")
    st.stop()

# ---------------------------------------------------------
# Dataset upload and preparation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Customer dataset")
    uploaded_file = st.file_uploader("Upload Customer Data.csv", type=["csv"])

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
elif os.path.exists("Customer Data.csv"):
    raw_data = pd.read_csv("Customer Data.csv")
else:
    raw_data = None

data = None
data_error = None

if raw_data is not None:
    try:
        data = engineer_features(raw_data)
        missing_columns = [col for col in FEATURES if col not in data.columns]
        if missing_columns:
            data_error = "Missing required feature columns: " + ", ".join(missing_columns)
            data = None
        else:
            data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
            data["Cluster"] = get_model_prediction(model, data)
            data = add_cluster_names(data)
    except Exception as exc:
        data_error = str(exc)

# ---------------------------------------------------------
# Predict customer page
# ---------------------------------------------------------
if page == "Predict Customer":
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown('<div class="section-label">👤 Customer information</div>', unsafe_allow_html=True)
        with st.form("customer_form"):
            col1, col2 = st.columns(2)
            with col1:
                balance = st.number_input("Balance", min_value=0.0, value=2500.0, step=100.0)
                credit_limit = st.number_input("Credit limit", min_value=0.0, value=8000.0, step=500.0)
                monthly_spending = st.number_input(
                    "Average monthly spending", min_value=0.0, value=450.0, step=50.0
                )
            with col2:
                purchases_frequency = st.slider("Purchases frequency", 0.0, 1.0, 0.65, 0.01)
                full_payment = st.slider("Full payment ratio", 0.0, 1.0, 0.40, 0.01)
                credit_utilization = st.number_input(
                    "Credit utilization", min_value=0.0, value=0.25, step=0.05, format="%.2f"
                )
            submitted = st.form_submit_button("✨ Predict customer segment", use_container_width=True)

    with right:
        st.markdown('<div class="section-label">🎯 Prediction result</div>', unsafe_allow_html=True)
        if submitted:
            input_data = pd.DataFrame([{
                "BALANCE": balance,
                "PURCHASES_FREQUENCY": purchases_frequency,
                "CREDIT_LIMIT": credit_limit,
                "PRC_FULL_PAYMENT": full_payment,
                "AVG_MONTHLY_SPENDING": monthly_spending,
                "CREDIT_UTILIZATION": credit_utilization,
            }])

            try:
                prediction = model.predict(input_data[FEATURES])[0]
                name = cluster_name(prediction)
                st.success(f"### {name}")
                st.metric("Predicted cluster ID", str(prediction))
                st.caption(
                    "The segment name is a human-readable label. Confirm that the mapping "
                    "matches the cluster profiles you observed during training."
                )
                result_df = pd.DataFrame({
                    "Feature": FEATURES,
                    "Customer input": input_data[FEATURES].iloc[0].values,
                })
                fig = px.bar(
                    result_df,
                    x="Feature",
                    y="Customer input",
                    title="Your customer's feature values",
                    template="plotly_dark",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=50, b=80),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        else:
            st.info("Fill in the customer details, then click **Predict customer segment**.")
            st.markdown(
                """
                **What you'll get**
                - Predicted customer cluster
                - A readable segment label
                - A quick view of the submitted feature values
                """
            )

    st.markdown("---")
    st.markdown('<div class="section-label">📌 Dataset snapshot</div>', unsafe_allow_html=True)
    if data is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers in dataset", f"{len(data):,}")
        c2.metric("Customer segments", f"{data['Cluster'].nunique():,}")
        c3.metric("Average balance", f"{data['BALANCE'].mean():,.0f}")
    elif data_error:
        st.warning(f"Dataset couldn't be prepared for visualizations: {data_error}")
    else:
        st.warning("Upload `Customer Data.csv` in the sidebar to enable dataset insights.")

# ---------------------------------------------------------
# Visualizations page
# ---------------------------------------------------------
elif page == "Visualizations":
    if data is None:
        st.warning(
            "Upload `Customer Data.csv` using the sidebar. The CSV must contain the original "
            "credit-card customer columns used by your project."
        )
        if data_error:
            st.error(data_error)
    else:
        counts = data["Cluster_Name"].value_counts().rename_axis("Segment").reset_index(name="Customers")
        counts["Share (%)"] = 100 * counts["Customers"] / len(data)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total customers", f"{len(data):,}")
        m2.metric("Segments found by pipeline", f"{data['Cluster'].nunique()}")
        m3.metric("Largest segment", counts.iloc[0]["Segment"])

        chart1, chart2 = st.columns(2, gap="large")
        with chart1:
            st.markdown('<div class="section-label">🥧 Cluster overview</div>', unsafe_allow_html=True)
            fig = px.pie(
                counts,
                names="Segment",
                values="Customers",
                hole=0.58,
                color_discrete_sequence=["#34d399", "#3b82f6", "#a855f7", "#f59e0b"],
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart2:
            st.markdown('<div class="section-label">📊 Customer count by segment</div>', unsafe_allow_html=True)
            fig = px.bar(
                counts,
                x="Segment",
                y="Customers",
                text="Customers",
                color="Segment",
                color_discrete_sequence=["#34d399", "#3b82f6", "#a855f7", "#f59e0b"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=80),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">📈 Feature comparison by cluster</div>', unsafe_allow_html=True)
        compare_features = [
            "BALANCE",
            "CREDIT_LIMIT",
            "AVG_MONTHLY_SPENDING",
            "PURCHASES_FREQUENCY",
            "PRC_FULL_PAYMENT",
            "CREDIT_UTILIZATION",
        ]
        summary = data.groupby("Cluster_Name", dropna=False)[compare_features].mean().reset_index()
        chosen_feature = st.selectbox("Choose a feature to compare", compare_features)
        fig = px.bar(
            summary,
            x="Cluster_Name",
            y=chosen_feature,
            color="Cluster_Name",
            text_auto=".2f",
            color_discrete_sequence=["#34d399", "#3b82f6", "#a855f7", "#f59e0b"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=70),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">🧭 Customer segments (PCA view)</div>', unsafe_allow_html=True)
        try:
            pca_input = data[FEATURES].copy()
            pca_input = pca_input.fillna(pca_input.median(numeric_only=True))
            scaled = StandardScaler().fit_transform(pca_input)
            coordinates = PCA(n_components=2, random_state=42).fit_transform(scaled)
            pca_data = pd.DataFrame({
                "PCA 1": coordinates[:, 0],
                "PCA 2": coordinates[:, 1],
                "Segment": data["Cluster_Name"].values,
            })
            fig = px.scatter(
                pca_data,
                x="PCA 1",
                y="PCA 2",
                color="Segment",
                opacity=0.65,
                color_discrete_sequence=["#34d399", "#3b82f6", "#a855f7", "#f59e0b"],
                title="Each point represents one customer; PCA compresses six features into two dimensions.",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"PCA visualization couldn't be generated: {exc}")

        st.markdown('<div class="section-label">💡 Key insights</div>', unsafe_allow_html=True)
        insight_cols = st.columns(min(3, len(summary)))
        for idx, (_, row) in enumerate(summary.iterrows()):
            segment = row["Cluster_Name"]
            with insight_cols[idx % len(insight_cols)]:
                st.markdown(f"**{segment}**")
                st.write(f"Average balance: **{row['BALANCE']:,.0f}**")
                st.write(f"Average credit limit: **{row['CREDIT_LIMIT']:,.0f}**")
                st.write(f"Purchase frequency: **{row['PURCHASES_FREQUENCY']:.2f}**")
                st.write(f"Share of customers: **{100 * (data['Cluster_Name'] == segment).mean():.1f}%**")

        st.markdown('<div class="section-label">🔎 Customer records</div>', unsafe_allow_html=True)
        selected_segment = st.selectbox(
            "Filter by segment",
            ["All segments"] + sorted(data["Cluster_Name"].dropna().unique().tolist()),
        )
        view = data if selected_segment == "All segments" else data[data["Cluster_Name"] == selected_segment]
        display_cols = [c for c in ["CUST_ID"] + FEATURES + ["Cluster", "Cluster_Name"] if c in view.columns]
        st.dataframe(view[display_cols].head(500), use_container_width=True, hide_index=True)
        st.caption("Showing up to 500 rows for readability.")

# ---------------------------------------------------------
# About page
# ---------------------------------------------------------
else:
    st.markdown("### About this project")
    st.write(
        "This dashboard uses a saved scikit-learn pipeline to predict customer segments. "
        "The pipeline should include the same imputation, scaling, and Decision Tree steps "
        "that were fitted during training."
    )
    st.info(
        "Feature-engineering assumption: AVG_MONTHLY_SPENDING = PURCHASES / TENURE, and "
        "CREDIT_UTILIZATION = BALANCE / CREDIT_LIMIT. If your training notebook calculated "
        "either feature differently, update `engineer_features()` so the website matches training."
    )
    st.warning(
        "Cluster names are labels for presentation only. Confirm the mapping in `CLUSTER_NAMES` "
        "against your actual cluster profile table."
    )
