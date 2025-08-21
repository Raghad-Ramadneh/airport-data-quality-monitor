# ✈️ Airport Data Quality Monitor

A simplified, professional tool for monitoring and analyzing the quality of airport-related datasets. This tool provides comprehensive data quality assessment including missing values detection, outlier analysis, duplicate identification, and trend analysis.

## 🎯 Features

- **Database Connectivity**: Direct connection to MySQL databases
- **Missing Values Analysis**: Identify incomplete data and critical gaps
- **Outlier Detection**: Statistical methods (Z-score, IQR) to find unusual values
- **Duplicate Detection**: Find exact duplicates and key field duplicates
- **Trend Analysis**: Time-series analysis for patterns and insights
- **Interactive Dashboard**: Real-time web interface using Streamlit
- **Professional Reports**: Automated Markdown report generation
- **Visualization**: Charts and graphs for better data understanding

## 📁 Project Structure

```
airport-data-quality-monitor/
├── src/
│   ├── data_quality_monitor.py    # Core analysis logic
│   ├── streamlit_dashboard.py     # Interactive web dashboard
│                   
├── task1/                         # Python virtual environment
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── data_quality_report.md         # Markdown Report
└── .gitignore                     # Git ignore rules
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/raghad-ramadneh/airport-data-quality-monitor.git
cd airport-data-quality-monitor
```

### 2. Create Virtual Environment

```bash
# Create virtual environment named 'task1' as required
python -m venv task1

# Activate virtual environment
# On Windows:
task1\Scripts\activate
# On macOS/Linux:
source task1/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Database (Optional)

If using MySQL database, set these environment variables:

```bash
export DB_HOST=localhost
export DB_USER=airport_user
export DB_PASS=your_password
export DB_NAME=airport_db
```

Or on Windows:
```cmd
set DB_HOST=localhost
set DB_USER=airport_user
set DB_PASS=your_password
set DB_NAME=airport_db
```

## 📊 Usage

### Option 1: Command Line Analysis

```bash
cd src
python data_quality_monitor.py

```

### Option 2: Interactive Dashboard

```bash
cd src
streamlit run streamlit_dashboard.py
```

Then open your browser 

## 📈 Analysis Components

### 1. Missing Values Analysis
- **Completeness Check**: Calculate missing data percentage for each column
- **Critical Identification**: Highlight columns with >10% missing values
- **Visual Reports**: Bar charts showing missing data distribution

### 2. Outlier Detection
- **Z-Score Method**: Identify values beyond 3 standard deviations
- **IQR Method**: Use Interquartile Range to find outliers
- **Statistical Summary**: Mean, median, min, max for numeric columns
- **Visualizations**: Box plots and histograms

### 3. Duplicate Analysis
- **Exact Duplicates**: Find completely identical rows
- **Key Field Duplicates**: Check unique identifiers (Flight Number, Airport Code)
- **Sample Display**: Show examples of duplicate records

### 4. Trend Analysis
- **Time Series**: Daily and monthly flight patterns
- **Delay Analysis**: Average delays over time (if available)
- **Anomaly Detection**: Unusual spikes or drops in data


### 3. Interactive Dashboard
- Real-time analysis interface
- Interactive charts and filters
- Quality score calculation
- Export capabilities

## 🔧 Configuration

### Database Configuration
The tool uses environment variables for database connection:

- `DB_HOST`: Database host (default: localhost)
- `DB_USER`: Database username (default: airport_user)
- `DB_PASS`: Database password (default: AirportSecure123!)
- `DB_NAME`: Database name (default: airport_db)

### Expected Data Schema
The tool expects airport data with columns like:
- Flight Number
- Airport Code
- Flight Arrival Country
- Flight Arrival City
- Flight Duration
- Airport GPS Code
- Airport Region Code
- Flight Departure Airport
- Flight Airline Code




### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m "Add feature"`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request


## 🙏 Acknowledgments

- Pandas and NumPy communities for excellent data processing libraries
- Streamlit team for the amazing web framework
- Plotly for interactive visualizations
