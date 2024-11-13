import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import pathlib
from property_risk_data_generator import KenyaPropertyDataGenerator
warnings.filterwarnings('ignore')

class RiskVisualizer:
    def __init__(self, property_data, weather_data):
        self.property_data = property_data
        self.weather_data = weather_data
        # Merge datasets for combined analysis
        self.combined_data = self._merge_datasets()

        # Create visualizations directory
        os.makedirs('visualizations', exist_ok=True)
        
    def _merge_datasets(self):
        """Merge property and weather data with aggregated weather events"""
        weather_agg = self.weather_data.groupby('property_id').agg({
            'damage_value': ['count', 'sum', 'mean'],
            'severity': lambda x: (x == 'Severe').sum()
        }).reset_index()
        weather_agg.columns = ['property_id', 'event_count', 'total_damage', 
                             'avg_damage', 'severe_events']
        
        return self.property_data.merge(weather_agg, on='property_id')

    def create_risk_heatmap(self):
        """Create an animated heatmap of property risks across Kenya"""
        fig = px.density_mapbox(self.combined_data,
                              lat='latitude',
                              lon='longitude',
                              z='total_damage',
                              radius=20,
                              center=dict(lat=-1.286389, lon=36.817223),
                              zoom=5,
                              mapbox_style="carto-positron",
                              title='Property Risk Heatmap: Total Damage Distribution',
                              color_continuous_scale="Viridis")
        
        fig.update_layout(
            title_x=0.5,
            title_font_size=20,
            margin=dict(t=60, b=20, l=20, r=20),
            width=1920,
            height=1080
        )
        return fig

    def create_risk_bubble_map(self):
        """Create a bubble map showing property values and risk levels"""
        fig = px.scatter_mapbox(self.combined_data,
                               lat='latitude',
                               lon='longitude',
                               size='property_value',
                               color='event_count',
                               hover_name='property_id',
                               hover_data=['property_type', 'zone_type', 
                                         'total_damage', 'flood_zone'],
                               color_continuous_scale='RdYlBu_r',
                               zoom=5,
                               title='Property Values vs Risk Events',
                               mapbox_style="carto-positron")
        
        fig.update_layout(
            title_x=0.5,
            title_font_size=20,
            margin=dict(t=60, b=20, l=20, r=20)
        )
        return fig

    def create_zone_comparison(self):
        """Create a comparison of risk metrics across different zones"""
        zone_metrics = self.combined_data.groupby('zone_type').agg({
            'total_damage': 'mean',
            'event_count': 'mean',
            'severe_events': 'mean',
            'property_value': 'mean'
        }).reset_index()
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Average Total Damage',
                                         'Average Event Count',
                                         'Average Severe Events',
                                         'Average Property Value'))
        
        # Create four bar charts
        metrics = ['total_damage', 'event_count', 'severe_events', 'property_value']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for metric, pos in zip(metrics, positions):
            fig.add_trace(
                go.Bar(x=zone_metrics['zone_type'],
                      y=zone_metrics[metric],
                      name=metric.replace('_', ' ').title(),
                      marker_color=['#FF9999', '#66B2FF', '#99FF99']),
                row=pos[0], col=pos[1]
            )
        
        fig.update_layout(height=800, 
                         showlegend=False,
                         title_text="Risk Metrics by Zone Type",
                         title_x=0.5)
        return fig

    def create_damage_trends(self):
        """Create a timeline of damage events"""
        monthly_damage = self.weather_data.groupby(
            [pd.Grouper(key='date', freq='M'), 'event_type']
        )['damage_value'].sum().reset_index()
        
        fig = px.line(monthly_damage, 
                     x='date', 
                     y='damage_value',
                     color='event_type',
                     title='Monthly Damage Trends by Event Type')
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Total Damage Value (KES)",
            title_x=0.5
        )
        return fig

    def generate_risk_summary(self):
        """Generate summary statistics for the video"""
        summary = {
            'total_properties': len(self.property_data),
            'total_events': len(self.weather_data),
            'total_damage': self.weather_data['damage_value'].sum(),
            'highest_risk_zone': self.combined_data.groupby('zone_type')['total_damage'].mean().idxmax(),
            'avg_events_per_property': self.combined_data['event_count'].mean(),
            'severe_event_percentage': (self.weather_data['severity'] == 'Severe').mean() * 100
        }
        return summary
    
    def save_visualizations(self):
        """Save all visualizations to HTML files"""
        try:
            print("Generating visualizations...")
            
            print("Creating risk heatmap...")
            heatmap = self.create_risk_heatmap()
            heatmap.write_html("visualizations/risk_heatmap.html")
            
            print("Creating bubble map...")
            bubble_map = self.create_risk_bubble_map()
            bubble_map.write_html("visualizations/risk_bubble_map.html")
            
            print("Creating zone comparison...")
            zone_comparison = self.create_zone_comparison()
            zone_comparison.write_html("visualizations/zone_comparison.html")
            
            print("Creating damage trends...")
            damage_trends = self.create_damage_trends()
            damage_trends.write_html("visualizations/damage_trends.html")
            
            print("All visualizations saved successfully!")
            
        except Exception as e:
            print(f"Error saving visualizations: {str(e)}")

# Test
# if __name__ == "__main__":
#     # First generate dummy data using our previous generator
#     generator = KenyaPropertyDataGenerator()
#     property_data = generator.generate_property_data(1000)
#     weather_data = generator.generate_weather_events(property_data)
    
#     # Create visualizations
#     visualizer = RiskVisualizer(property_data, weather_data)
    
#     # Generate all plots
#     heatmap = visualizer.create_risk_heatmap()
#     bubble_map = visualizer.create_risk_bubble_map()
#     zone_comparison = visualizer.create_zone_comparison()
#     damage_trends = visualizer.create_damage_trends()
    
#     # Get summary statistics
#     summary = visualizer.generate_risk_summary()
#     visualizer = visualizer.save_visualizations()

#     # Save plots (optional)
#     # heatmap.write_html("visualizations/risk_heatmap.html")
#     # bubble_map.write_html("visualizations/risk_bubble_map.html")
#     # zone_comparison.write_html("visualizations/zone_comparison.html")
#     # damage_trends.write_html("visualizations/damage_trends.html")