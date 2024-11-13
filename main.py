from property_risk_data_generator import KenyaPropertyDataGenerator
from property_risk_visualizations import RiskVisualizer
from property_risk_ml_pipeline import PropertyRiskModel

def main():
    try:
        print("Starting property risk analysis...")
        
        # Generate data
        print("Generating dummy data...")
        generator = KenyaPropertyDataGenerator()
        property_data = generator.generate_property_data(1000)
        weather_data = generator.generate_weather_events(property_data)
        
        # Create and save visualizations
        print("Creating and saving visualizations...")
        visualizer = RiskVisualizer(property_data, weather_data)
        visualizer.save_visualizations()
        
        # Display summary
        summary = visualizer.generate_risk_summary()
        print("\nAnalysis Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
            
        print("\nAnalysis completed successfully!")

        # Add cross validation score, uncomment this function in the ml pipeline file to use it
        # risk_model = PropertyRiskModel()
        # cv_score_results = risk_model.cross_val_score_evaluation()
        # print("\nCross Validation Score Results:")
        # for metric, values in cv_score_results.items():
        #     print(f"{metric}:")
        #     print(f"Mean: {values['mean']:.4f} (±{values['std']:.4f})")
        
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()