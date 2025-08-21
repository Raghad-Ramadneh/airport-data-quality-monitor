#!/usr/bin/env python3
"""
Simplified Airport Data Quality Monitor
A clean, easy-to-understand tool for analyzing airport data quality.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
import pymysql
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

class AirportDataMonitor:
    """Simplified Airport Data Quality Monitor"""
    
    def __init__(self):
        """Initialize the monitor with default settings"""
        self.data = None
        self.results = {}
        self.report_content = []
        
        # Database configuration from environment variables
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'airport_user'),
            'password': os.getenv('DB_PASS', 'AirportSecure123!'),
            'database': os.getenv('DB_NAME', 'airport_db')
        }
    
    def connect_to_database(self, connection_type='mysql'):
        """
        Connect to database and load data
        
        Args:
            connection_type (str): 'mysql' or 'sqlite'
        
        Returns:
            pandas.DataFrame: Loaded data or None if failed
        """
        print("🔌 Connecting to database...")
        
        try:
            if connection_type == 'mysql':
                # MySQL connection
                engine = create_engine(
                    f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@"
                    f"{self.db_config['host']}/{self.db_config['database']}"
                )
                query = "SELECT * FROM MOCK_DATA"
                self.data = pd.read_sql(query, engine)
                
            elif connection_type == 'sqlite':
                # SQLite connection (for testing)
                conn = sqlite3.connect('airport_data.db')
                self.data = pd.read_sql("SELECT * FROM MOCK_DATA", conn)
                conn.close()
            
            print(f"✅ Successfully loaded {len(self.data)} records")
            return self.data
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return None
    
    
    
    def get_basic_info(self):
        """Get basic dataset information"""
        if self.data is None:
            return None
        
        info = {
            'total_records': len(self.data),
            'total_columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'data_types': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        self.results['basic_info'] = info
        return info
    
    def check_missing_values(self):
        """
        Check for missing values in the dataset
        
        Returns:
            dict: Missing values analysis results
        """
        print("🔍 Checking for missing values...")
        
        if self.data is None:
            return None
        
        # Calculate missing values for each column
        missing_data = []
        for column in self.data.columns:
            missing_count = self.data[column].isnull().sum()
            missing_percent = (missing_count / len(self.data)) * 100
            
            missing_data.append({
                'Column': column,
                'Missing_Count': missing_count,
                'Missing_Percent': round(missing_percent, 2),
                'Status': 'Critical' if missing_percent > 10 else 'Good'
            })
        
        missing_df = pd.DataFrame(missing_data)
        
        # Overall missing data percentage
        total_missing = (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        
        results = {
            'missing_summary': missing_df,
            'total_missing_percent': round(total_missing, 2),
            'critical_columns': missing_df[missing_df['Missing_Percent'] > 10]['Column'].tolist()
        }
        
        self.results['missing_values'] = results
        
        print(f"   Total missing data: {total_missing:.2f}%")
        if results['critical_columns']:
            print(f"   ⚠️  Critical columns (>10% missing): {results['critical_columns']}")
        
        return results
    
    def detect_outliers(self):
        """
        Detect outliers using Z-score and IQR methods
        
        Returns:
            dict: Outlier detection results
        """
        print("📊 Detecting outliers...")
        
        if self.data is None:
            return None
        
        # Get numeric columns only
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            print("   No numeric columns found for outlier detection")
            return {'message': 'No numeric columns found'}
        
        outlier_results = []
        
        for column in numeric_columns:
            # Remove missing values for analysis
            clean_data = self.data[column].dropna()
            
            if len(clean_data) < 10:  # Skip if too few data points
                continue
            
            # Method 1: Z-score (values beyond 3 standard deviations)
            z_scores = np.abs((clean_data - clean_data.mean()) / clean_data.std())
            z_outliers = clean_data[z_scores > 3]
            
            # Method 2: IQR (Interquartile Range)
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            
            outlier_results.append({
                'Column': column,
                'Z_Score_Outliers': len(z_outliers),
                'IQR_Outliers': len(iqr_outliers),
                'Outlier_Percent': round((len(iqr_outliers) / len(clean_data)) * 100, 2),
                'Min_Value': clean_data.min(),
                'Max_Value': clean_data.max(),
                'Mean': round(clean_data.mean(), 2),
                'Std_Dev': round(clean_data.std(), 2)
            })
        
        results = {
            'outlier_summary': pd.DataFrame(outlier_results),
            'numeric_columns': numeric_columns
        }
        
        self.results['outliers'] = results
        
        total_outliers = sum([r['IQR_Outliers'] for r in outlier_results])
        print(f"   Found {total_outliers} potential outliers across {len(numeric_columns)} numeric columns")
        
        return results
    
    def find_duplicates(self):
        """
        Find duplicate records in the dataset
        
        Returns:
            dict: Duplicate analysis results
        """
        print("🔄 Checking for duplicate records...")
        
        if self.data is None:
            return None
        
        # Check for complete duplicate rows
        total_duplicates = self.data.duplicated().sum()
        duplicate_percent = (total_duplicates / len(self.data)) * 100
        
        # Check for duplicates in key identifier columns
        key_columns = ['Flight Number', 'Airport Code', 'Flight_ID']  # Adjust based on your data
        key_duplicates = {}
        
        for col in key_columns:
            if col in self.data.columns:
                col_duplicates = self.data[col].duplicated().sum()
                key_duplicates[col] = col_duplicates
        
        # Get sample duplicate rows
        duplicate_rows = None
        if total_duplicates > 0:
            duplicate_rows = self.data[self.data.duplicated(keep=False)].head(10)
        
        results = {
            'total_duplicates': total_duplicates,
            'duplicate_percent': round(duplicate_percent, 2),
            'key_column_duplicates': key_duplicates,
            'sample_duplicates': duplicate_rows
        }
        
        self.results['duplicates'] = results
        
        print(f"   Found {total_duplicates} complete duplicate rows ({duplicate_percent:.2f}%)")
        
        return results
    
    
    
    def create_visualizations(self, output_dir='charts'):
        """
        Create visualization charts for the analysis
        
        Args:
            output_dir (str): Directory to save chart images
        """
        print("📊 Creating visualizations...")
        
        if self.data is None:
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Missing Values Chart
        if 'missing_values' in self.results:
            missing_df = self.results['missing_values']['missing_summary']
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(missing_df['Column'], missing_df['Missing_Percent'], 
                          color=['red' if x > 10 else 'skyblue' for x in missing_df['Missing_Percent']])
            plt.title('Missing Values by Column')
            plt.xlabel('Columns')
            plt.ylabel('Missing Percentage (%)')
            plt.xticks(rotation=45, ha='right')
            plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% Threshold')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/missing_values.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Outliers Chart
        if 'outliers' in self.results and not self.results['outliers']['outlier_summary'].empty:
            outlier_df = self.results['outliers']['outlier_summary']
            
            plt.figure(figsize=(12, 6))
            plt.bar(outlier_df['Column'], outlier_df['Outlier_Percent'], color='orange')
            plt.title('Outlier Percentage by Column')
            plt.xlabel('Columns')
            plt.ylabel('Outlier Percentage (%)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/outliers.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        
    
    def generate_markdown_report(self, output_file='data_quality_report.md'):
        """
        Generate a comprehensive Markdown report
        
        Args:
            output_file (str): Output file path for the report
        """
        print("📝 Generating Markdown report...")
        
        report = []
        report.append("# Airport Data Quality Analysis Report")
        report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        if 'basic_info' in self.results:
            info = self.results['basic_info']
            report.append(f"- **Total Records:** {info['total_records']:,}")
            report.append(f"- **Total Columns:** {info['total_columns']}")
            report.append(f"- **Dataset Size:** {info['memory_usage']:.2f} MB")
        
        if 'missing_values' in self.results:
            missing = self.results['missing_values']
            report.append(f"- **Overall Missing Data:** {missing['total_missing_percent']}%")
            report.append(f"- **Critical Columns (>10% missing):** {len(missing['critical_columns'])}")
        
        if 'duplicates' in self.results:
            duplicates = self.results['duplicates']
            report.append(f"- **Duplicate Records:** {duplicates['total_duplicates']} ({duplicates['duplicate_percent']}%)")
        
        report.append("")
        
        # Missing Values Analysis
        if 'missing_values' in self.results:
            report.append("## Missing Values Analysis")
            report.append("")
            
            missing = self.results['missing_values']
            missing_df = missing['missing_summary']
            
            report.append("### Summary Table")
            report.append("")
            report.append("| Column | Missing Count | Missing % | Status |")
            report.append("|--------|---------------|-----------|---------|")
            
            for _, row in missing_df.iterrows():
                status_icon = "🔴" if row['Status'] == 'Critical' else "🟢"
                report.append(f"| {row['Column']} | {row['Missing_Count']} | {row['Missing_Percent']}% | {status_icon} {row['Status']} |")
            
            report.append("")
            report.append("![Missing Values Chart](charts/missing_values.png)")
            report.append("")
        
        # Outlier Detection
        if 'outliers' in self.results and not self.results['outliers']['outlier_summary'].empty:
            report.append("## Outlier Detection")
            report.append("")
            
            outliers = self.results['outliers']
            outlier_df = outliers['outlier_summary']
            
            report.append("### Outlier Summary")
            report.append("")
            report.append("| Column | Z-Score Outliers | IQR Outliers | Outlier % | Min | Max | Mean |")
            report.append("|--------|------------------|--------------|-----------|-----|-----|------|")
            
            for _, row in outlier_df.iterrows():
                report.append(f"| {row['Column']} | {row['Z_Score_Outliers']} | {row['IQR_Outliers']} | {row['Outlier_Percent']}% | {row['Min_Value']} | {row['Max_Value']} | {row['Mean']} |")
            
            report.append("")
            report.append("![Outliers Chart](charts/outliers.png)")
            report.append("")
        
        # Duplicate Analysis
        if 'duplicates' in self.results:
            report.append("## Duplicate Records Analysis")
            report.append("")
            
            duplicates = self.results['duplicates']
            report.append(f"- **Total Duplicate Rows:** {duplicates['total_duplicates']}")
            report.append(f"- **Duplicate Percentage:** {duplicates['duplicate_percent']}%")
            report.append("")
            
            if duplicates['key_column_duplicates']:
                report.append("### Key Column Duplicates")
                for col, count in duplicates['key_column_duplicates'].items():
                    report.append(f"- **{col}:** {count} duplicates")
                report.append("")
        
        
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        recommendations = []
        
        if 'missing_values' in self.results and self.results['missing_values']['critical_columns']:
            recommendations.append("🔴 **Critical:** Address columns with >10% missing values")
        
        if 'duplicates' in self.results and self.results['duplicates']['total_duplicates'] > 0:
            recommendations.append("🟡 **Important:** Remove duplicate records to ensure data integrity")
        
        if 'outliers' in self.results:
            total_outliers = sum([row['IQR_Outliers'] for _, row in self.results['outliers']['outlier_summary'].iterrows()])
            if total_outliers > len(self.data) * 0.05:  # More than 5% outliers
                recommendations.append("🟡 **Review:** High number of outliers detected - verify data accuracy")
        
        if not recommendations:
            recommendations.append("🟢 **Good:** No critical data quality issues detected")
        
        for rec in recommendations:
            report.append(f"- {rec}")
        
        report.append("")
        
        # Footer
        report.append("---")
        report.append("*Report generated by Airport Data Quality Monitor*")
        
        # Write report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Report saved as '{output_file}'")
    
    def run_complete_analysis(self, data_source='database', file_path=None, date_column=None):
        """
        Run the complete data quality analysis pipeline
        
        Args:
            data_source (str): 'database'
            

        """
        print("🚀 Starting Airport Data Quality Analysis")
        print("=" * 50)
        
        # Step 1: Load Data
        if data_source == 'database':
            self.connect_to_database()
        
        else:
            print("❌ Invalid data source or missing file path")
            return
        
        if self.data is None:
            print("❌ Failed to load data. Exiting.")
            return
        
        # Step 2: Basic Information
        self.get_basic_info()
        
        # Step 3: Quality Checks
        self.check_missing_values()
        self.detect_outliers()
        self.find_duplicates()

        # Step 4: Create Visualizations
        self.create_visualizations()
        
        # Step 5: Generate Report
        self.generate_markdown_report()
        
        print("=" * 50)
        print("✅ Analysis completed successfully!")
        print("📁 Check the following files:")
        print("   - data_quality_report.md (Main report)")
        print("   - charts/ directory (Visualization images)")


# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = AirportDataMonitor()
    
    # Run complete analysis
    monitor.run_complete_analysis(data_source='database')
 
