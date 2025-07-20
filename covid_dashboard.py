#!/usr/bin/env python3
"""
COVID-19 Data Visualizer for India
==================================

This project visualizes the spread of COVID-19 in India using real-world time-series data.
It uses Python, Pandas, and Matplotlib to load, filter, and plot COVID-19 case trends,
including confirmed, recovered, and death counts.

Author: COVID Data Analysis Team
Date: 2024
Version: 1.0
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import requests
import io
import warnings
warnings.filterwarnings('ignore')

class CovidDataVisualizer:
    """Main class for COVID-19 data visualization and analysis"""
    
    def __init__(self):
        """Initialize the visualizer"""
        self.data = None
        self.india_data = None
        print("🦠 COVID-19 Data Visualizer for India")
        print("=" * 50)
        
    def load_data_from_url(self):
        """
        Load COVID-19 data from Johns Hopkins CSSE GitHub repository
        
        Returns:
            tuple: (confirmed_df, deaths_df, recovered_df) DataFrames
        """
        try:
            print("📡 Attempting to load real-time data from Johns Hopkins CSSE...")
            
            # URLs for Johns Hopkins CSSE COVID-19 data
            base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
            
            urls = {
                'confirmed': f"{base_url}time_series_covid19_confirmed_global.csv",
                'deaths': f"{base_url}time_series_covid19_deaths_global.csv",
                'recovered': f"{base_url}time_series_covid19_recovered_global.csv"
            }
            
            dataframes = {}
            for key, url in urls.items():
                print(f"   Loading {key} data...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                dataframes[key] = pd.read_csv(io.StringIO(response.text))
            
            print("✅ Successfully loaded real-time data!")
            return dataframes['confirmed'], dataframes['deaths'], dataframes['recovered']
            
        except Exception as e:
            print(f"❌ Error loading real-time data: {e}")
            print("🔄 Falling back to sample data...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """
        Create realistic sample COVID-19 data for India for demonstration
        
        Returns:
            tuple: (confirmed_df, deaths_df, recovered_df) DataFrames with sample data
        """
        print("🔧 Creating realistic sample data for demonstration...")
        
        # Create date range from March 2020 to December 2023
        dates = pd.date_range(start='2020-03-01', end='2023-12-31', freq='D')
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Generate realistic COVID-19 data patterns for India
        confirmed = []
        daily_cases = 0
        
        for i, date in enumerate(dates):
            # Simulate different phases of the pandemic
            if i < 30:  # Initial phase (March 2020)
                daily_new = np.random.poisson(max(1, i * 3))
            elif i < 200:  # First wave (April-September 2020)
                wave1_peak = 100
                wave1_intensity = 1000 * np.sin((i-30) * np.pi / 170) + np.random.normal(0, 200)
                daily_new = max(0, int(wave1_intensity))
            elif i < 300:  # Between waves (October-December 2020)
                daily_new = np.random.poisson(max(100, 45000 - (i-200) * 200))
            elif i < 500:  # Second wave (January-July 2021)
                wave2_peak = 400
                wave2_intensity = 2500 * np.sin((i-300) * np.pi / 200) + np.random.normal(0, 500)
                daily_new = max(0, int(wave2_intensity))
            elif i < 650:  # Post second wave (August-December 2021)
                daily_new = np.random.poisson(max(50, 200000 - (i-500) * 300))
            elif i < 750:  # Omicron wave (January-March 2022)
                wave3_intensity = 1200 * np.sin((i-650) * np.pi / 100) + np.random.normal(0, 300)
                daily_new = max(0, int(wave3_intensity))
            else:  # Endemic phase (April 2022 onwards)
                daily_new = np.random.poisson(max(10, 5000 - (i-750) * 2))
            
            daily_cases += daily_new
            confirmed.append(daily_cases)
        
        # Generate deaths data (approximately 1.2% case fatality rate with delays)
        deaths = []
        for i, c in enumerate(confirmed):
            if i < 14:  # Initial delay
                death_count = int(c * 0.001)
            else:
                # Deaths follow confirmed cases with ~2 week delay and varying rates
                delayed_cases = confirmed[max(0, i-14)]
                if i < 400:  # Higher mortality in first waves
                    death_rate = 0.018
                else:  # Lower mortality with better treatment
                    death_rate = 0.008
                death_count = int(delayed_cases * death_rate)
            deaths.append(death_count)
        
        # Generate recovered data (approximately 97% recovery rate with delays)
        recovered = []
        for i, c in enumerate(confirmed):
            if i < 21:  # Initial delay for recovery
                recovery_count = int(c * 0.3)
            else:
                # Recoveries follow confirmed cases with ~3 week delay
                delayed_cases = confirmed[max(0, i-21)]
                recovery_count = int(delayed_cases * 0.97)
            recovered.append(min(recovery_count, c - deaths[i]))  # Can't recover more than (confirmed - deaths)
        
        # Create base DataFrame structure
        base_df = pd.DataFrame({
            'Province/State': [''],
            'Country/Region': ['India'],
            'Lat': [20.5937],
            'Long': [78.9629]
        })
        
        # Create separate DataFrames for each metric
        confirmed_df = base_df.copy()
        deaths_df = base_df.copy()
        recovered_df = base_df.copy()
        
        # Add time series data as columns
        for i, date in enumerate(dates):
            date_str = date.strftime('%-m/%-d/%y')  # Format: M/D/YY
            confirmed_df[date_str] = confirmed[i]
            deaths_df[date_str] = deaths[i]
            recovered_df[date_str] = recovered[i]
        
        print("✅ Sample data created successfully!")
        return confirmed_df, deaths_df, recovered_df
    
    def filter_india_data(self, confirmed_df, deaths_df, recovered_df):
        """
        Filter and process data specifically for India
        
        Args:
            confirmed_df: DataFrame with confirmed cases
            deaths_df: DataFrame with deaths
            recovered_df: DataFrame with recovered cases
            
        Returns:
            DataFrame: Processed India-specific COVID-19 data
        """
        print("🇮🇳 Filtering and processing data for India...")
        
        # Filter for India data
        india_confirmed = confirmed_df[confirmed_df['Country/Region'] == 'India']
        india_deaths = deaths_df[deaths_df['Country/Region'] == 'India']
        india_recovered = recovered_df[recovered_df['Country/Region'] == 'India']
        
        if india_confirmed.empty:
            print("❌ No India data found in the dataset")
            return None
        
        # Get date columns (skip the first 4 metadata columns)
        date_columns = list(confirmed_df.columns[4:])
        
        # Extract and sum time series data for India (in case of multiple provinces)
        confirmed_series = india_confirmed[date_columns].sum()
        deaths_series = india_deaths[date_columns].sum()
        recovered_series = india_recovered[date_columns].sum()
        
        # Convert date strings to datetime objects
        try:
            dates = pd.to_datetime(date_columns, format='%m/%d/%y')
        except:
            # Try alternative format
            dates = pd.to_datetime(date_columns, format='%#m/%#d/%y')
        
        # Create comprehensive India dataset
        india_data = pd.DataFrame({
            'Date': dates,
            'Confirmed': confirmed_series.values,
            'Deaths': deaths_series.values,
            'Recovered': recovered_series.values
        })
        
        # Calculate derived metrics
        india_data['Active'] = india_data['Confirmed'] - india_data['Deaths'] - india_data['Recovered']
        india_data['Active'] = india_data['Active'].clip(lower=0)  # Ensure no negative values
        
        # Calculate daily changes
        india_data['Daily_Confirmed'] = india_data['Confirmed'].diff().fillna(0).clip(lower=0)
        india_data['Daily_Deaths'] = india_data['Deaths'].diff().fillna(0).clip(lower=0)
        india_data['Daily_Recovered'] = india_data['Recovered'].diff().fillna(0).clip(lower=0)
        
        # Calculate moving averages
        india_data['MA_7_Confirmed'] = india_data['Daily_Confirmed'].rolling(window=7, center=True).mean()
        india_data['MA_14_Confirmed'] = india_data['Daily_Confirmed'].rolling(window=14, center=True).mean()
        
        # Calculate rates
        india_data['Mortality_Rate'] = (india_data['Deaths'] / india_data['Confirmed'] * 100).fillna(0)
        india_data['Recovery_Rate'] = (india_data['Recovered'] / india_data['Confirmed'] * 100).fillna(0)
        india_data['Active_Rate'] = (india_data['Active'] / india_data['Confirmed'] * 100).fillna(0)
        
        print(f"✅ Processed {len(india_data)} days of India COVID-19 data")
        print(f"📅 Date range: {india_data['Date'].min().strftime('%Y-%m-%d')} to {india_data['Date'].max().strftime('%Y-%m-%d')}")
        
        return india_data
    
    def create_visualizations(self, india_data):
        """
        Create comprehensive COVID-19 visualizations for India
        
        Args:
            india_data: DataFrame with processed India COVID-19 data
        """
        print("📊 Creating comprehensive visualizations...")
        
        # Set up plotting style and parameters
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 10
        
        # Define color scheme
        colors = {
            'confirmed': '#FF6B6B',    # Red
            'recovered': '#4ECDC4',    # Teal
            'deaths': '#45B7D1',       # Blue
            'active': '#96CEB4',       # Light Green
            'daily': '#FFA07A',        # Light Salmon
            'ma': '#8B0000'            # Dark Red
        }
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        
        # 1. CUMULATIVE CASES OVER TIME
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(india_data['Date'], india_data['Confirmed'], 
                color=colors['confirmed'], linewidth=2.5, label='Confirmed', alpha=0.9)
        ax1.plot(india_data['Date'], india_data['Recovered'], 
                color=colors['recovered'], linewidth=2.5, label='Recovered', alpha=0.9)
        ax1.plot(india_data['Date'], india_data['Deaths'], 
                color=colors['deaths'], linewidth=2.5, label='Deaths', alpha=0.9)
        ax1.plot(india_data['Date'], india_data['Active'], 
                color=colors['active'], linewidth=2.5, label='Active', alpha=0.9)
        
        ax1.set_title('📈 COVID-19 Cumulative Cases in India', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('Number of Cases', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Format y-axis with Indian number system
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_indian_number(x)))
        
        # 2. DAILY NEW CASES WITH MOVING AVERAGES
        ax2 = plt.subplot(2, 3, 2)
        ax2.fill_between(india_data['Date'], india_data['Daily_Confirmed'], 
                        color=colors['daily'], alpha=0.5, label='Daily Cases')
        ax2.plot(india_data['Date'], india_data['MA_7_Confirmed'], 
                color=colors['ma'], linewidth=2.5, label='7-day Average')
        ax2.plot(india_data['Date'], india_data['MA_14_Confirmed'], 
                color='darkblue', linewidth=2, label='14-day Average', alpha=0.8)
        
        ax2.set_title('📊 Daily New COVID-19 Cases in India', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Date', fontweight='bold')
        ax2.set_ylabel('Daily New Cases', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_indian_number(x)))
        
        # 3. CURRENT CASE DISTRIBUTION (PIE CHART)
        ax3 = plt.subplot(2, 3, 3)
        latest_data = india_data.iloc[-1]
        sizes = [latest_data['Active'], latest_data['Recovered'], latest_data['Deaths']]
        labels = ['Active', 'Recovered', 'Deaths']
        colors_pie = [colors['active'], colors['recovered'], colors['deaths']]
        explode = (0.05, 0, 0)  # Slightly separate the active slice
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, 
                                          autopct='%1.1f%%', startangle=90, explode=explode,
                                          shadow=True)
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax3.set_title(f'🥧 Case Distribution\n(as of {latest_data["Date"].strftime("%Y-%m-%d")})', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 4. MORTALITY AND RECOVERY RATES
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(india_data['Date'], india_data['Mortality_Rate'], 
                color=colors['deaths'], linewidth=2.5, label='Mortality Rate (%)')
        ax4.plot(india_data['Date'], india_data['Recovery_Rate'], 
                color=colors['recovered'], linewidth=2.5, label='Recovery Rate (%)')
        
        # Add horizontal reference lines
        ax4.axhline(y=india_data['Mortality_Rate'].iloc[-1], color=colors['deaths'], 
                   linestyle='--', alpha=0.5)
        ax4.axhline(y=india_data['Recovery_Rate'].iloc[-1], color=colors['recovered'], 
                   linestyle='--', alpha=0.5)
        
        ax4.set_title('📋 Mortality and Recovery Rates in India', fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel('Date', fontweight='bold')
        ax4.set_ylabel('Rate (%)', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. WEEKLY COMPARISON (BAR CHART)
        ax5 = plt.subplot(2, 3, 5)
        
        # Get last 8 weeks of data
        weekly_data = india_data.tail(56).groupby(india_data.tail(56)['Date'].dt.isocalendar().week).agg({
            'Daily_Confirmed': 'sum',
            'Daily_Deaths': 'sum',
            'Daily_Recovered': 'sum'
        }).tail(8)
        
        x = range(len(weekly_data))
        width = 0.25
        
        bars1 = ax5.bar([i - width for i in x], weekly_data['Daily_Confirmed'], 
                       width, label='New Cases', color=colors['confirmed'], alpha=0.8)
        bars2 = ax5.bar(x, weekly_data['Daily_Deaths'], 
                       width, label='Deaths', color=colors['deaths'], alpha=0.8)
        bars3 = ax5.bar([i + width for i in x], weekly_data['Daily_Recovered'], 
                       width, label='Recovered', color=colors['recovered'], alpha=0.8)
        
        ax5.set_title('📅 Weekly COVID-19 Summary (Last 8 Weeks)', fontsize=14, fontweight='bold', pad=20)
        ax5.set_xlabel('Week', fontweight='bold')
        ax5.set_ylabel('Number of Cases', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'W{i+1}' for i in range(len(weekly_data))])
        
        # 6. GROWTH RATE ANALYSIS
        ax6 = plt.subplot(2, 3, 6)
        
        # Calculate growth rates
        india_data['Growth_Rate'] = india_data['Confirmed'].pct_change(periods=7) * 100
        india_data['Growth_Rate_MA'] = india_data['Growth_Rate'].rolling(window=7).mean()
        
        ax6.fill_between(india_data['Date'], india_data['Growth_Rate'], 
                        color=colors['confirmed'], alpha=0.3, label='Weekly Growth Rate')
        ax6.plot(india_data['Date'], india_data['Growth_Rate_MA'], 
                color=colors['ma'], linewidth=2.5, label='7-day Average')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax6.set_title('📈 Weekly Growth Rate (%)', fontsize=14, fontweight='bold', pad=20)
        ax6.set_xlabel('Date', fontweight='bold')
        ax6.set_ylabel('Growth Rate (%)', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
        
        # Add overall title and adjust layout
        plt.tight_layout()
        plt.suptitle('🦠 COVID-19 Data Visualizer for India - Comprehensive Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Add footer with data source
        fig.text(0.5, 0.02, 'Data Source: Johns Hopkins CSSE COVID-19 Dataset | Generated by COVID-19 Data Visualizer', 
                ha='center', fontsize=10, style='italic')
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(top=0.94, bottom=0.08, hspace=0.3, wspace=0.3)
        
        # Show the plot
        plt.show()
        
        # Print summary statistics
        self.print_summary(india_data)
    
    def print_summary(self, india_data):
        """
        Print comprehensive summary statistics
        
        Args:
            india_data: DataFrame with processed India COVID-19 data
        """
        latest = india_data.iloc[-1]
        peak_daily = india_data['Daily_Confirmed'].max()
        peak_date = india_data.loc[india_data['Daily_Confirmed'].idxmax(), 'Date']
        
        # Calculate additional statistics
        total_days = len(india_data)
        avg_daily_cases = india_data['Daily_Confirmed'].mean()
        max_active = india_data['Active'].max()
        max_active_date = india_data.loc[india_data['Active'].idxmax(), 'Date']
        
        print("\n" + "🦠" + "="*60 + "🦠")
        print("           COVID-19 INDIA COMPREHENSIVE SUMMARY")
        print("🦠" + "="*60 + "🦠")
        
        print(f"\n📅 DATA PERIOD:")
        print(f"   From: {india_data['Date'].min().strftime('%B %d, %Y')}")
        print(f"   To:   {india_data['Date'].max().strftime('%B %d, %Y')}")
        print(f"   Total Days Analyzed: {total_days:,}")
        
        print(f"\n📊 LATEST STATISTICS (as of {latest['Date'].strftime('%B %d, %Y')}):")
        print(f"   ├── 🔴 Total Confirmed Cases: {format_indian_number(latest['Confirmed'])}")
        print(f"   ├── 🟢 Total Recovered:       {format_indian_number(latest['Recovered'])}")
        print(f"   ├── 🔵 Total Deaths:          {format_indian_number(latest['Deaths'])}")
        print(f"   └── 🟡 Active Cases:          {format_indian_number(latest['Active'])}")
        
        print(f"\n📈 KEY METRICS:")
        print(f"   ├── 💀 Mortality Rate:        {latest['Mortality_Rate']:.2f}%")
        print(f"   ├── 💚 Recovery Rate:         {latest['Recovery_Rate']:.2f}%")
        print(f"   ├── 🟡 Active Case Rate:      {latest['Active_Rate']:.2f}%")
        print(f"   └── 📊 Avg Daily Cases:       {format_indian_number(avg_daily_cases)}")
        
        print(f"\n🎯 PEAK STATISTICS:")
        print(f"   ├── 🔥 Peak Daily Cases:      {format_indian_number(peak_daily)}")
        print(f"   ├── 📅 Peak Date:             {peak_date.strftime('%B %d, %Y')}")
        print(f"   ├── 🏔️ Max Active Cases:      {format_indian_number(max_active)}")
        print(f"   └── 📅 Max Active Date:       {max_active_date.strftime('%B %d, %Y')}")
        
        # Wave Analysis
        print(f"\n🌊 WAVE ANALYSIS:")
        self.analyze_waves(india_data)
        
        print("\n" + "🦠" + "="*60 + "🦠")
        print("         Analysis Complete - Stay Safe, Stay Healthy! 🏥")
        print("🦠" + "="*60 + "🦠")
    
    def analyze_waves(self, india_data):
        """
        Analyze COVID-19 waves in India using peak detection
        
        Args:
            india_data: DataFrame with processed India COVID-19 data
        """
        # Use 7-day moving average for smoother peak detection
        daily_cases_smooth = india_data['MA_7_Confirmed'].fillna(0)
        
        # Find significant peaks
        peaks = []
        min_peak_height = daily_cases_smooth.max() * 0.15  # Peaks must be at least 15% of max
        min_prominence = 30  # Minimum days between peaks
        
        for i in range(min_prominence, len(daily_cases_smooth) - min_prominence):
            current_value = daily_cases_smooth.iloc[i]
            
            if (current_value > min_peak_height and
                current_value > daily_cases_smooth.iloc[i-min_prominence:i].max() and 
                current_value > daily_cases_smooth.iloc[i+1:i+min_prominence+1].max()):
                
                peak_date = india_data.iloc[i]['Date']
                peaks.append((peak_date, current_value))
        
        # Display wave information
        if peaks:
            for i, (date, cases) in enumerate(peaks, 1):
                print(f"   ├── 🌊 Wave {i}: {date.strftime('%B %Y')} - {format_indian_number(cases)} daily cases")
        else:
            print("   └── 📊 No distinct waves detected in the data")
    
    def export_data(self, india_data, filename='covid_india_data.csv'):
        """
        Export processed data to CSV file
        
        Args:
            india_data: DataFrame with processed India COVID-19 data
            filename: Output filename
        """
        try:
            india_data.to_csv(filename, index=False)
            print(f"\n💾 Data exported successfully to '{filename}'")
            print(f"📁 File contains {len(india_data)} rows and {len(india_data.columns)} columns")
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
    
    def run_analysis(self):
        """
        Main function to run the complete COVID-19 analysis
        
        Returns:
            DataFrame: Processed India COVID-19 data or None if failed
        """
        try:
            print("🚀 Starting COVID-19 analysis for India...")
            
            # Load data
            confirmed_df, deaths_df, recovered_df = self.load_data_from_url()
            
            # Filter and process India data
            india_data = self.filter_india_data(confirmed_df, deaths_df, recovered_df)
            
            if india_data is None:
                print("❌ Could not process India data")
                return None
            
            # Store data in instance
            self.india_data = india_data
            
            # Create visualizations
            self.create_visualizations(india_data)
            
            return india_data
            
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            return None


def format_indian_number(num):
    """
    Format numbers in Indian numbering system (Lakhs and Crores)
    
    Args:
        num: Number to format
        
    Returns:
        str: Formatted number string
    """
    if pd.isna(num):
        return "0"
    
    num = int(num)
    if num >= 10000000:  # 1 Crore
        return f"{num/10000000:.1f}Cr"
    elif num >= 100000:  # 1 Lakh
        return f"{num/100000:.1f}L"
    elif num >= 1000:  # 1 Thousand
        return f"{num/1000:.1f}K"
    else:
        return str(num)


def save_plots_as_images():
    """Save the current plot as high-quality images"""
    try:
        import os
        
        # Create screenshots directory if it doesn't exist
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')
        
        # Save as PNG and PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_filename = f"screenshots/covid_dashboard_{timestamp}.png"
        pdf_filename = f"screenshots/covid_dashboard_{timestamp}.pdf"
        
        plt.savefig(png_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"📸 Plots saved as:")
        print(f"   - {png_filename}")
        print(f"   - {pdf_filename}")
        
    except Exception as e:
        print(f"❌ Error saving plots: {e}")


# Main execution
if __name__ == "__main__":
    try:
        print("🌟" + "="*60 + "🌟")
        print("        COVID-19 DATA VISUALIZER FOR INDIA")
        print("     Comprehensive Analysis & Visualization Tool")
        print("🌟" + "="*60 + "🌟")
        
        # Create visualizer instance
        visualizer = CovidDataVisualizer()
        
        # Run the complete analysis
        data = visualizer.run_analysis()
        
        if data is not None:
            # Export processed data
            visualizer.export_data(data)
            
            # Save plots as images
            save_plots_as_images()
            
            print("\n🎉 Analysis Complete! Dashboard Features:")
            print("   ✅ 1. Cumulative cases timeline")
            print("   ✅ 2. Daily new cases with moving averages")
            print("   ✅ 3. Current case distribution pie chart")
            print("   ✅ 4. Mortality and recovery rates")
            print("   ✅ 5. Weekly summary comparison")
            print("   ✅ 6. Growth rate analysis")
            print("   ✅ 7. Wave detection and peak analysis")
            print("   ✅ 8. Data export to CSV")
            print("   ✅ 9. High-quality plot exports")
            
            print("\n📂 Generated Files:")
            print("   📊 covid_india_data.csv - Processed data")
            print("   📸 screenshots/ - Dashboard images")
            
            print("\n🔧 Technical Features:")
            print("   🌐 Real-time data loading from Johns Hopkins CSSE")
            print("   🔄 Fallback to sample data if online unavailable")
            print("   📈 Advanced statistical analysis")
            print("   🎨 Professional visualization styling")
            print("   🔢 Indian number formatting (Lakhs/Crores)")
            print("   📅 Comprehensive date handling")
            
        else:
            print("\n❌ Analysis failed. Possible issues:")
            print("   🌐 Check internet connection")
            print("   📊 Verify data source availability")
            print("   🔧 Ensure all required packages are installed")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        print("👋 Thank you for using COVID-19 Data Visualizer!")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("🔧 Please check your Python environment and dependencies")
        
    finally:
        print("\n" + "🏥" + "="*60 + "🏥")
        print("  Remember: Wear masks, maintain distance, get vaccinated!")
        print("           Stay safe and stay healthy! 💉🏥")
        print("🏥" + "="*60 + "🏥")


# Additional utility functions for enhanced functionality
class DataAnalyzer:
    """Additional analysis utilities for COVID-19 data"""
    
    @staticmethod
    def calculate_doubling_time(data, cases_column='Confirmed'):
        """
        Calculate the doubling time of COVID-19 cases
        
        Args:
            data: DataFrame with COVID-19 data
            cases_column: Column name for cases data
            
        Returns:
            Series: Doubling times in days
        """
        growth_rate = data[cases_column].pct_change()
        doubling_time = np.log(2) / np.log(1 + growth_rate)
        return doubling_time.replace([np.inf, -np.inf], np.nan)
    
    @staticmethod
    def detect_outbreak_periods(data, threshold_multiplier=2):
        """
        Detect periods of rapid outbreak based on daily cases
        
        Args:
            data: DataFrame with COVID-19 data
            threshold_multiplier: Multiplier for average to determine outbreak
            
        Returns:
            list: List of outbreak periods
        """
        daily_cases = data['Daily_Confirmed']
        avg_cases = daily_cases.mean()
        threshold = avg_cases * threshold_multiplier
        
        outbreak_periods = []
        in_outbreak = False
        outbreak_start = None
        
        for i, cases in enumerate(daily_cases):
            if cases > threshold and not in_outbreak:
                outbreak_start = data.iloc[i]['Date']
                in_outbreak = True
            elif cases <= threshold and in_outbreak:
                outbreak_end = data.iloc[i]['Date']
                outbreak_periods.append((outbreak_start, outbreak_end))
                in_outbreak = False
        
        return outbreak_periods
    
    @staticmethod
    def calculate_attack_rate(data, population=1380000000):
        """
        Calculate attack rate (percentage of population infected)
        
        Args:
            data: DataFrame with COVID-19 data
            population: Total population (default: India ~1.38 billion)
            
        Returns:
            Series: Attack rates as percentage
        """
        return (data['Confirmed'] / population) * 100


class ReportGenerator:
    """Generate detailed reports from COVID-19 data"""
    
    @staticmethod
    def generate_markdown_report(data, filename='covid_report.md'):
        """
        Generate a comprehensive markdown report
        
        Args:
            data: DataFrame with COVID-19 data
            filename: Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# COVID-19 India Analysis Report\n\n")
                f.write(f"**Report Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n\n")
                
                # Executive Summary
                latest = data.iloc[-1]
                f.write("## Executive Summary\n\n")
                f.write(f"- **Total Confirmed Cases:** {format_indian_number(latest['Confirmed'])}\n")
                f.write(f"- **Total Recovered:** {format_indian_number(latest['Recovered'])}\n")
                f.write(f"- **Total Deaths:** {format_indian_number(latest['Deaths'])}\n")
                f.write(f"- **Active Cases:** {format_indian_number(latest['Active'])}\n")
                f.write(f"- **Mortality Rate:** {latest['Mortality_Rate']:.2f}%\n")
                f.write(f"- **Recovery Rate:** {latest['Recovery_Rate']:.2f}%\n\n")
                
                # Data Period
                f.write("## Data Period\n\n")
                f.write(f"**From:** {data['Date'].min().strftime('%B %d, %Y')}\n")
                f.write(f"**To:** {data['Date'].max().strftime('%B %d, %Y')}\n")
                f.write(f"**Total Days:** {len(data):,}\n\n")
                
                # Key Statistics
                f.write("## Key Statistics\n\n")
                peak_daily = data['Daily_Confirmed'].max()
                peak_date = data.loc[data['Daily_Confirmed'].idxmax(), 'Date']
                f.write(f"- **Peak Daily Cases:** {format_indian_number(peak_daily)} on {peak_date.strftime('%B %d, %Y')}\n")
                f.write(f"- **Average Daily Cases:** {format_indian_number(data['Daily_Confirmed'].mean())}\n")
                f.write(f"- **Maximum Active Cases:** {format_indian_number(data['Active'].max())}\n\n")
                
                # Methodology
                f.write("## Methodology\n\n")
                f.write("This analysis uses data from the Johns Hopkins CSSE COVID-19 Dataset. ")
                f.write("The following metrics are calculated:\n\n")
                f.write("- **Active Cases:** Confirmed - Recovered - Deaths\n")
                f.write("- **Daily Changes:** Day-over-day differences\n")
                f.write("- **Moving Averages:** 7-day and 14-day rolling averages\n")
                f.write("- **Rates:** Mortality, recovery, and active case percentages\n\n")
                
                # Data Quality Notes
                f.write("## Data Quality Notes\n\n")
                f.write("- Data may include reporting delays and revisions\n")
                f.write("- Moving averages help smooth daily reporting variations\n")
                f.write("- Numbers are formatted using Indian numbering system (Lakhs/Crores)\n\n")
                
                f.write("---\n")
                f.write("*Report generated by COVID-19 Data Visualizer for India*\n")
                
            print(f"📄 Markdown report generated: {filename}")
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")


# Configuration and settings
CONFIG = {
    'DATA_SOURCES': {
        'johns_hopkins_base': "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/",
        'backup_sources': [
            # Add backup data sources here if needed
        ]
    },
    'VISUALIZATION': {
        'figure_size': (20, 14),
        'dpi': 300,
        'style': 'seaborn-v0_8',
        'color_scheme': {
            'confirmed': '#FF6B6B',
            'recovered': '#4ECDC4', 
            'deaths': '#45B7D1',
            'active': '#96CEB4'
        }
    },
    'ANALYSIS': {
        'moving_average_windows': [7, 14],
        'wave_detection_threshold': 0.15,
        'outbreak_threshold_multiplier': 2,
        'india_population': 1380000000
    }
}


# Version and metadata
__version__ = "1.0.0"
__author__ = "COVID Data Analysis Team"
__description__ = "Comprehensive COVID-19 data visualization and analysis tool for India"
__license__ = "MIT"

if __name__ == "__main__":
    print(f"\n🔖 COVID-19 Data Visualizer v{__version__}")
    print(f"📝 {__description__}")
    print(f"👥 {__author__}")
    print(f"📄 License: {__license__}")
    print("-" * 60)