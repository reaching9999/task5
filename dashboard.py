import streamlit as st
import pandas as pd
import requests
import io
import plotly.graph_objects as go
import numpy as np
import pingouin as pg
from fpdf import FPDF

# url for the gsheet, pretty important stuff. easy to find up here
G_SHEET_URL71 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSq8vervIKDmR0OLpampNd6BIay2_a3Lo0yBcVTRjrvpvVi61QKElSydzP_uk2-CXbNrznOmUenDqfn/pub?gid=0&single=true&output=csv"

@st.cache_data(ttl=600) # Caching this thing for 10 mins, so we dont keep hitting the sheet
def load_data71(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        # this whole timedelta thing is the only way to get the dates right from sheets
        time_deltas = pd.to_timedelta(df['Date'], unit='d')
        origin_date = pd.Timestamp('1899-12-30')
        df['Date'] = origin_date + time_deltas
        
        return df
    except Exception as e:
        st.error(f"An error occured while processing the data: {e}")
        return None

def find_anomalies71(series, k, thresh, win, pct_thresh, alpha):
    iqr_anomalies = pd.Series(False, index=series.index)
    z_anomalies = pd.Series(False, index=series.index)
    ma_anomalies = pd.Series(False, index=series.index)
    grubbs_anomalies = pd.Series(False, index=series.index)

    if not series.empty:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            iqr_anomalies = (series < (q1 - k * iqr)) | (series > (q3 + k * iqr))

        if series.std() > 0:
            z_scores = np.abs((series - series.mean()) / series.std())
            z_anomalies = z_scores > thresh

        moving_avg = series.rolling(window=win, center=True, min_periods=1).mean()
        ma_dev = np.abs(series - moving_avg) / moving_avg * 100
        ma_anomalies = ma_dev > pct_thresh

        try:
            grubbs_result = pg.grubbs(series, alpha=alpha)
            if grubbs_result['outlier'].any():
                outlier_index = grubbs_result.index[grubbs_result['outlier']].values[0]
                grubbs_anomalies.loc[outlier_index] = True
        except Exception:
            pass
            
    # the | means 'or', so if any test finds it, its an anomaly
    all_anomalies = iqr_anomalies | z_anomalies | ma_anomalies | grubbs_anomalies
    return all_anomalies

def create_plot71(df, col, chart_type, trend_deg, anomalies):
    fig = go.Figure()
    
    if chart_type == 'Line':
        fig.add_trace(go.Scatter(x=df['Date'], y=df[col], mode='lines', name=col))
    else:
        fig.add_trace(go.Bar(x=df['Date'], y=df[col], name=col))

    anomaly_pts = df[anomalies]
    fig.add_trace(go.Scatter(
        x=anomaly_pts['Date'], y=anomaly_pts[col],
        mode='markers', marker=dict(color='red', size=10, symbol='x'), name='Anomaly'
    ))

    if trend_deg is not None:
        x_numeric = (df['Date'] - df['Date'].min()).dt.days
        coeffs = np.polyfit(x_numeric, df[col].fillna(0), trend_deg)
        trendline = np.poly1d(coeffs)(x_numeric)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=trendline, mode='lines', line=dict(color='orange', dash='dash'), name=f'Trend (Deg {trend_deg})'
        ))

    fig.update_layout(
        title=f'Daily Production for {col}', xaxis_title='Date', yaxis_title='Resource Units',
        barmode='stack' if chart_type == 'Stacked Bar' else 'group'
    )
    return fig

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Weyland-Yutani Mining Operations Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report71(stats_df, anomaly_df, fig, mine):
    pdf = PDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Descriptive Statistics', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    header = ['Metric'] + list(stats_df.columns)
    pdf.cell(40, 6, header[0], 1)
    for h in header[1:]:
        pdf.cell(30, 6, h, 1)
    pdf.ln()
    for index, row in stats_df.iterrows():
        pdf.cell(40, 6, index, 1)
        for item in row:
            pdf.cell(30, 6, f'{item:.2f}', 1)
        pdf.ln()
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f'Production Chart for {mine}', 0, 1)
    
    chart_path = "temp_chart.png"
    try:
        fig.write_image(chart_path, engine="kaleido")
        pdf.image(chart_path, x=10, y=None, w=190)
    except Exception as e:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, f"(Chart could not be generated: {e})", 0, 1)

    if not anomaly_df.empty:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Detected Anomaly Details', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        anomaly_header = list(anomaly_df.columns)
        col_widths = [35, 30, 30, 30, 30]
        for i, h in enumerate(anomaly_header):
            pdf.cell(col_widths[i], 6, str(h), 1)
        pdf.ln()
        for _, row in anomaly_df.iterrows():
            for i, item in enumerate(row):
                if isinstance(item, pd.Timestamp):
                    pdf.cell(col_widths[i], 6, item.strftime('%Y-%m-%d'), 1)
                else:
                    pdf.cell(col_widths[i], 6, f'{item:.2f}', 1)
            pdf.ln()

    return bytes(pdf.output())

def main():
    st.set_page_config(layout="wide")
    st.title("Weyland-Yutani Mining Ops Dashbord 71")

    st.sidebar.title('Dashboard Controlls')

    st.sidebar.header('Anomaly Detection Settings')
    iqr_k71 = st.sidebar.slider("IQR Multiplier (k)", 1.0, 3.0, 1.5, 0.1)
    z_thresh71 = st.sidebar.slider("Z-Score Threshold", 1.0, 4.0, 3.0, 0.1)
    ma_win71 = st.sidebar.slider("Moving Avg Window (days)", 3, 21, 7, 1)
    ma_pct_thresh71 = st.sidebar.slider("Moving Avg Deviation (%)", 5, 50, 20, 1)
    grubbs_a71 = st.sidebar.selectbox("Grubbs' Test Alpha", [0.01, 0.05, 0.10], index=1)

    st.sidebar.header('Chart Options')
    chart_sel71 = st.sidebar.selectbox("Chart Type", ["Line", "Bar", "Stacked Bar"])
    trend_deg71 = st.sidebar.selectbox("Polynomial Trendline Degree", [None, 1, 2, 3, 4])

    df71 = load_data71(G_SHEET_URL71)

    if df71 is not None:
        st.success("Live data feed from generator sheet loaded successfuly.")

        final_cols71 = ['Date', 'LV-426', 'Origae-6', 'Florina 151']
        if all(col in df71.columns for col in final_cols71):
            df_clean71 = df71[final_cols71].copy()
            mine_cols71 = ['LV-426', 'Origae-6', 'Florina 151']
            for col in mine_cols71:
                df_clean71[col] = pd.to_numeric(df_clean71[col], errors='coerce')

            df_clean71['Total'] = df_clean71[mine_cols71].sum(axis=1)
            analysis_cols71 = mine_cols71 + ['Total']

            st.header("Descriptive Statistiks")
            stats_data = {col: {"Mean": df_clean71[col].mean(), "Std Dev": df_clean71[col].std(), "Median": df_clean71[col].median(), "IQR": df_clean71[col].quantile(0.75) - df_clean71[col].quantile(0.25)} for col in analysis_cols71}
            stats_df71 = pd.DataFrame(stats_data)
            st.dataframe(stats_df71.style.format("{:.2f}"))

            st.header("Production Analaysis Chart")
            mine_sel71 = st.selectbox("Select Mine/Total to Analyze", analysis_cols71)
            series_to_check = df_clean71[mine_sel71].dropna()
            anomalies71 = find_anomalies71(series_to_check, iqr_k71, z_thresh71, ma_win71, ma_pct_thresh71, grubbs_a71)
            fig71 = create_plot71(df_clean71, mine_sel71, chart_sel71, trend_deg71, anomalies71)
            st.plotly_chart(fig71, use_container_width=True)

            st.header("Detected Anomalys")
            anomaly_df = df_clean71.loc[anomalies71]
            if anomaly_df.empty:
                st.info("No anomalies detected with the current settings.")
            else:
                with st.expander("Show Anomaly Details"):
                    st.dataframe(anomaly_df)
            
            st.header("Reporting")
            if st.button("Generate PDF Report"):
                with st.spinner("Generating Report..."):
                    pdf_data = generate_pdf_report71(stats_df71, anomaly_df, fig71, mine_sel71)
                    st.download_button(
                        label="Download Report as PDF",
                        data=pdf_data,
                        file_name="mining_report_71.pdf",
                        mime="application/pdf"
                    )
        
        else:
            st.error("The loaded data is missing one or more required columns.")
    else:
        st.warning("Could not load or process data. Please check the data source url.")

if __name__ == "__main__":

    main()
