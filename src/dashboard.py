#!/usr/bin/env python3
"""
Simplified Airport Data Quality Dashboard
Clean, interactive web interface for data quality monitoring.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Import our simplified monitor
try:
    from data_quality_monitor import AirportDataMonitor
except ImportError:
    st.error("❌ Please make sure 'data_quality_monitor.py' is in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Airport Data Quality Monitor",
    page_icon="✈️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .metric-container {
        background-color: #2a5033;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_cache(data_source, file_path=None):
    """Cached data loading function"""
    monitor = AirportDataMonitor()
    
    if data_source == "database":
        data = monitor.connect_to_database()
    
    else:
        return None, None
    
    return data, monitor

def display_metrics_cards(info):
    """Display key metrics in cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Total Records</h3>
            <h2>{info['total_records']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📋 Columns</h3>
            <h2>{info['total_columns']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        memory_mb = info.get('memory_usage', 0)
        st.markdown(f"""
        <div class="metric-container">
            <h3>💾 Dataset Size</h3>
            <h2>{memory_mb:.1f} MB</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🔢 Data Types</h3>
            <h2>{len(set(info['data_types'].values()))}</h2>
        </div>
        """, unsafe_allow_html=True)

def show_missing_values_analysis(monitor):
    """Display missing values analysis"""
    st.subheader("🔍 Missing Values Analysis")
    
    missing_results = monitor.check_missing_values()
    
    if missing_results is None:
        st.error("No data available for analysis")
        return
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Missing %", f"{missing_results['total_missing_percent']:.2f}%")
    
    with col2:
        st.metric("Critical Columns", len(missing_results['critical_columns']))
    
    with col3:
        status = "Good" if missing_results['total_missing_percent'] < 5 else "Needs Attention"
        st.metric("Data Quality", status)
    
    # Missing values chart
    missing_df = missing_results['missing_summary']
    
    if not missing_df.empty:
        fig = px.bar(
            missing_df,
            x='Column',
            y='Missing_Percent',
            color='Missing_Percent',
            color_continuous_scale=['green', 'yellow', 'red'],
            title="Missing Values by Column (%)"
        )
        # Fixed: Use update_layout
        fig.update_layout(xaxis_tickangle=-45)
        fig.add_hline(y=10, line_dash="dash", line_color="red", 
                     annotation_text="Critical Threshold (10%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.subheader("Detailed Missing Values Summary")
        st.dataframe(missing_df, use_container_width=True)
        
        # Warnings for critical columns
        if missing_results['critical_columns']:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ Critical Issues Found</h4>
                <p>The following columns have more than 10% missing values:</p>
                <ul>
                    {''.join([f'<li>{col}</li>' for col in missing_results['critical_columns']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

def show_outlier_analysis(monitor):
    """Display outlier analysis"""
    st.subheader("📊 Outlier Detection")
    
    outlier_results = monitor.detect_outliers()
    
    if outlier_results is None or 'message' in outlier_results:
        st.info("No numeric columns found for outlier analysis")
        return
    
    outlier_df = outlier_results['outlier_summary']
    
    if outlier_df.empty:
        st.info("No outliers detected in the dataset")
        return
    
    # Summary metrics
    total_outliers = outlier_df['IQR_Outliers'].sum()
    total_records = len(monitor.data)
    outlier_percentage = (total_outliers / total_records) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Outliers", f"{total_outliers:,}")
    
    with col2:
        st.metric("Outlier Percentage", f"{outlier_percentage:.2f}%")
    
    with col3:
        status = "Good" if outlier_percentage < 5 else "Review Needed"
        st.metric("Status", status)
    

    
    # Detailed table
    st.subheader("Outlier Details")
    st.dataframe(outlier_df, use_container_width=True)
    
    # Column selection for detailed analysis
    if not outlier_df.empty:
        selected_column = st.selectbox("Select column for detailed analysis:", outlier_df['Column'].tolist())
        
        if selected_column and selected_column in monitor.data.columns:
            col_data = monitor.data[selected_column].dropna()
            
            # Box plot
            fig_box = px.box(y=col_data, title=f"Box Plot: {selected_column}")
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Histogram
            fig_hist = px.histogram(x=col_data, title=f"Distribution: {selected_column}", nbins=30)
            st.plotly_chart(fig_hist, use_container_width=True)

def show_duplicate_analysis(monitor):
    """Display duplicate analysis"""
    st.subheader("🔄 Duplicate Records Analysis")
    
    duplicate_results = monitor.find_duplicates()
    
    if duplicate_results is None:
        st.error("No data available for analysis")
        return
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Duplicate Rows", f"{duplicate_results['total_duplicates']:,}")
    
    with col2:
        st.metric("Duplicate %", f"{duplicate_results['duplicate_percent']:.2f}%")
    
    with col3:
        status = "Clean" if duplicate_results['total_duplicates'] == 0 else "Action Needed"
        st.metric("Status", status)
    
    # Key column duplicates
    if duplicate_results['key_column_duplicates']:
        st.subheader("Key Column Duplicates")
        
        key_data = []
        for col, count in duplicate_results['key_column_duplicates'].items():
            key_data.append({'Column': col, 'Duplicates': count})
        
        if key_data:
            key_df = pd.DataFrame(key_data)
            fig = px.bar(key_df, x='Column', y='Duplicates', title="Duplicates in Key Columns")
            st.plotly_chart(fig, use_container_width=True)
    
    # Show sample duplicates
    if duplicate_results['total_duplicates'] > 0 and duplicate_results['sample_duplicates'] is not None:
        st.subheader("Sample Duplicate Records")
        st.dataframe(duplicate_results['sample_duplicates'], use_container_width=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Duplicate Records Found</h4>
            <p>Consider removing duplicate records to improve data quality and analysis accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            <h4>✅ No Duplicate Records</h4>
            <p>Great! Your dataset is clean from duplicate records.</p>
        </div>
        """, unsafe_allow_html=True)


def show_data_sample(monitor):
    """Display sample of the dataset"""
    st.subheader("📋 Data Sample")
    
    # Display basic info
    info = monitor.get_basic_info()
    display_metrics_cards(info)
    
    # Show column information
    st.subheader("Column Information")
    col_info = []
    for col, dtype in info['data_types'].items():
        unique_count = monitor.data[col].nunique()
        null_count = monitor.data[col].isnull().sum()
        
        col_info.append({
            'Column': col,
            'Data Type': str(dtype),
            'Unique Values': unique_count,
            'Null Values': null_count,
            'Sample Value': str(monitor.data[col].dropna().iloc[0]) if len(monitor.data[col].dropna()) > 0 else 'N/A'
        })
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True)
    
    # Show sample data
    st.subheader("Sample Records")
    sample_size = st.slider("Number of records to display", 5, 50, 10)
    st.dataframe(monitor.data.head(sample_size), use_container_width=True)

def generate_summary_report(monitor):
    """Generate and display summary report"""
    st.subheader("📝 Quality Summary Report")
    
    # Run all analyses
    basic_info = monitor.get_basic_info()
    missing_results = monitor.check_missing_values()
    outlier_results = monitor.detect_outliers()
    duplicate_results = monitor.find_duplicates()
    
    # Calculate overall quality score
    quality_score = 100
    
    # Deduct points for missing values
    if missing_results:
        quality_score -= missing_results['total_missing_percent']
    
    # Deduct points for duplicates
    if duplicate_results:
        quality_score -= duplicate_results['duplicate_percent']
    
    # Deduct points for high outlier percentage
    if outlier_results and 'outlier_summary' in outlier_results:
        outlier_df = outlier_results['outlier_summary']
        if not outlier_df.empty:
            avg_outlier_pct = outlier_df['Outlier_Percent'].mean()
            if avg_outlier_pct > 5:
                quality_score -= (avg_outlier_pct - 5)
    
    quality_score = max(0, min(100, quality_score))  # Keep between 0-100
    
    # Display quality score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if quality_score >= 90:
            st.success(f"🟢 Overall Quality Score: {quality_score:.1f}%")
        elif quality_score >= 70:
            st.warning(f"🟡 Overall Quality Score: {quality_score:.1f}%")
        else:
            st.error(f"🔴 Overall Quality Score: {quality_score:.1f}%")
    
    with col2:
        issues_count = len([x for x in [
            missing_results['critical_columns'] if missing_results else [],
            [1] if duplicate_results and duplicate_results['total_duplicates'] > 0 else [],
        ] if x])
        st.metric("Critical Issues", issues_count)
    
    with col3:
        st.metric("Analysis Date", datetime.now().strftime('%Y-%m-%d'))
    
    # Issues and recommendations
    st.subheader("🎯 Recommendations")
    
    recommendations = []
    
    if missing_results and missing_results['critical_columns']:
        recommendations.append("🔴 **High Priority:** Address columns with >10% missing values")
        for col in missing_results['critical_columns'][:3]:  # Show top 3
            recommendations.append(f"   - Review data collection process for '{col}'")
    
    if duplicate_results and duplicate_results['total_duplicates'] > 0:
        recommendations.append("🟡 **Medium Priority:** Remove duplicate records")
        recommendations.append(f"   - Found {duplicate_results['total_duplicates']} duplicate rows")
    
    if outlier_results and 'outlier_summary' in outlier_results:
        outlier_df = outlier_results['outlier_summary']
        if not outlier_df.empty:
            high_outlier_cols = outlier_df[outlier_df['Outlier_Percent'] > 10]
            if not high_outlier_cols.empty:
                recommendations.append("🟡 **Review Needed:** High outlier percentages detected")
                for _, row in high_outlier_cols.iterrows():
                    recommendations.append(f"   - '{row['Column']}': {row['Outlier_Percent']:.1f}% outliers")
    
    if not recommendations:
        recommendations.append("🟢 **Excellent:** No critical data quality issues detected!")
        recommendations.append("   - Dataset appears to be in good condition")
        recommendations.append("   - Continue monitoring for ongoing data quality")
    
    for rec in recommendations:
        st.markdown(rec)

def main():
    """Main application function"""
    
    # Header
    st.title("✈️ Airport Data Quality Monitor")
    st.markdown("**Professional data quality analysis for airport datasets**")
    st.markdown("---")
    
   

    # Data source 
    data_source = st.sidebar.radio(
	 "data source:",
	 ["database"],
	 format_func=lambda x: "MySQL Database"
    )

    uploaded_file = None
    file_path = None
    
   
        
    if uploaded_file:
       # Save uploaded file temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Load data button
    if st.sidebar.button("🚀 Load Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                data, monitor = load_data_cache(data_source, file_path)
                
                if data is not None:
                    st.session_state.data = data
                    st.session_state.monitor = monitor
                    st.session_state.data_loaded = True
                    st.sidebar.success(f"✅ Loaded {len(data):,} records")
                else:
                    st.sidebar.error("❌ Failed to load data")
                    st.session_state.data_loaded = False
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
                st.session_state.data_loaded = False
    
    # Clean up temporary file
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            pass
    
    # Display database configuration info
    if data_source == "database":
        with st.sidebar.expander("📋 Database Configuration"):
            st.write("Set these environment variables:")
            st.code("""
DB_HOST=localhost
DB_USER=airport_user  
DB_PASS=your_password
DB_NAME=airport_db
            """)
    
    # Main content area
    if not st.session_state.get('data_loaded', False):
        st.info("👆 Please load your data using the sidebar to begin analysis")
        
        # Instructions
        st.markdown("""
        ### 📋 How to Use
        
        1. **Data Source**: MySQL Database 
        2. **Load Data**: Click "Load Data" to import your dataset  
        3. **Analyze**: Explore the analysis tabs to review data quality
        4. **Generate Report**: View comprehensive quality assessment
        
        ### 📊 Analysis Features
        
        - **Missing Values**: Identify incomplete data and critical gaps
        - **Outlier Detection**: Find unusual values using statistical methods
        - **Duplicate Records**: Detect and analyze duplicate entries
        - **Quality Score**: Overall data quality assessment
        """)
        
        return
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5  = st.tabs([
        "📊 Overview",
        "🔍 Missing Values", 
        "📈 Outliers",
        "🔄 Duplicates",
        "📝 Summary Report"
    ])
    
    monitor = st.session_state.monitor
    
    with tab1:
        show_data_sample(monitor)
    
    with tab2:
        show_missing_values_analysis(monitor)
    
    with tab3:
        show_outlier_analysis(monitor)
    
    with tab4:
        show_duplicate_analysis(monitor)
    
    with tab5:
        generate_summary_report(monitor)
    
    # Sidebar quick stats
    if st.session_state.get('data_loaded', False):
        st.sidebar.header("📊 Quick Stats")
        
        info = monitor.get_basic_info()
        st.sidebar.metric("Records", f"{info['total_records']:,}")
        st.sidebar.metric("Columns", info['total_columns'])
        
        # Quick quality indicators
        missing_results = monitor.check_missing_values()
        if missing_results:
            missing_pct = missing_results['total_missing_percent']
            if missing_pct < 5:
                st.sidebar.success(f"Missing: {missing_pct:.1f}%")
            elif missing_pct < 15:
                st.sidebar.warning(f"Missing: {missing_pct:.1f}%")
            else:
                st.sidebar.error(f"Missing: {missing_pct:.1f}%")
        
        duplicate_results = monitor.find_duplicates()
        if duplicate_results:
            dup_pct = duplicate_results['duplicate_percent']
            if dup_pct == 0:
                st.sidebar.success("No Duplicates")
            else:
                st.sidebar.warning(f"Duplicates: {dup_pct:.1f}%")

if __name__ == "__main__":
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    main()
