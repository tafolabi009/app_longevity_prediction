#!/usr/bin/env python3

import sys
import os
import argparse
from scripts.model import AppLongevityModel
from joblib import load
import json
import pandas as pd
import numpy as np

def main():
    """
    Command-line interface for predicting app longevity based on app name.
    Wraps the App Longevity Prediction project in a simple, easy-to-use interface.
    """
    parser = argparse.ArgumentParser(
        description="App Longevity Prediction - Analyze any app's potential lifespan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_app.py "Spotify"
  python predict_app.py "Instagram" --compare-competitors
  python predict_app.py "Candy Crush" --no-visualize
  
Report bugs to: your-email@example.com
App Longevity Prediction home page: <https://github.com/yourusername/app_longevity_prediction>
"""
    )
    
    parser.add_argument("app_name", 
                        help="Name of the app to analyze", 
                        type=str)
    
    parser.add_argument("--compare-competitors", 
                        action="store_true",
                        help="Compare with competitor apps in the same category")
    
    parser.add_argument("--no-visualize", 
                        action="store_true",
                        help="Don't generate visualization graphics")
    
    parser.add_argument("--simple", 
                        action="store_true",
                        help="Simple output without detailed explanations")
    
    parser.add_argument("--save-results", 
                        action="store_true",
                        help="Save results to a JSON file")
    
    args = parser.parse_args()
    
    try:
        # Ensure directories exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        # Load or train model
        print("Loading app longevity prediction model...")
        model = AppLongevityModel()
        
        # Check if models exist
        model_files = os.listdir('models') if os.path.exists('models') else []
        if not model_files or not any(f.endswith('.joblib') for f in model_files):
            print("No trained models found. Please run scripts/model.py to train models first.")
            return 1
            
        # Load best model
        try:
            with open('models/metrics.json', 'r') as f:
                metrics = json.load(f)
            best_model_name = min(metrics.items(), key=lambda x: x[1]['rmse'])[0]
            print(f"Using {best_model_name} model for prediction")
            
            if best_model_name == 'nn':
                from tensorflow.keras.models import load_model as load_keras_model
                model_obj = load_keras_model(f'models/{best_model_name}_model.h5')
            else:
                model_obj = load(f'models/{best_model_name}_model.joblib')
            
            model.best_model = best_model_name
            model.models = {best_model_name: model_obj}
            model.scaler = load('models/scaler.joblib')
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Please run scripts/model.py to train models first.")
            return 1
        
        # Analyze the app
        print(f"\n{'='*60}")
        print(f"📱 Analyzing app: {args.app_name}")
        print(f"{'='*60}")
        
        # Start analysis
        results = model.predict_app_longevity(
            args.app_name,
            compare_with_competitors=args.compare_competitors,
            visualize=not args.no_visualize,
            explain=not args.simple
        )
        
        if not results:
            print("❌ Failed to analyze app. Please check the app name or try a different app.")
            return 1
            
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return 1
        
        # Display results in a nice format
        print(f"\n{'='*60}")
        print(f"📊 APP LONGEVITY PREDICTION RESULTS")
        print(f"{'='*60}")
        print(f"App Name: {results['app_name']}")
        print(f"Platform: {results['platform']}")
        
        # Longevity score with interpretation
        longevity = results['predicted_longevity']
        interpretation = results['longevity_interpretation']
        
        print(f"\n📈 Longevity Score: {longevity:.4f}")
        print(f"Category: {interpretation['category']}")
        print(f"Expected Lifespan: {interpretation['expected_lifespan']}")
        print(f"Success Probability: {interpretation['success_probability']}")
        print(f"\nInterpretation: {interpretation['description']}")
        
        # Key metrics section
        print(f"\n📱 Key App Metrics:")
        print(f"{'-'*40}")
        for feature, value in results['key_metrics'].items():
            if value not in [None, 'Unknown']:
                print(f"• {feature.replace('_', ' ').title()}: {value}")
        
        # Contributing factors
        if 'contributing_factors' in results:
            print(f"\n🔍 Top Contributing Factors:")
            print(f"{'-'*40}")
            for factor in results['contributing_factors']:
                print(f"• {factor['description']}")
        
        # Recommendations
        if 'recommendations' in results:
            print(f"\n💡 Recommendations:")
            print(f"{'-'*40}")
            for rec in results['recommendations']:
                print(f"• {rec['area']} ({rec['issue']}): {rec['recommendation']}")
        
        # Competitor analysis
        if 'competitor_analysis' in results:
            print(f"\n🥇 Market Position Analysis:")
            print(f"{'-'*40}")
            competition = results['competitor_analysis']
            
            if 'market_position_percentile' in competition and competition['market_position_percentile'] is not None:
                print(f"Market Position: {competition['market_position_percentile']}th percentile in {competition['category']}")
            
            print("\nTop Competitors:")
            for comp in competition['competitors'][:3]:  # Show top 3
                print(f"• {comp['name']} - Rating: {comp['rating']}, Downloads: {comp['downloads']}")
        
        # Visualization
        if 'visualization_path' in results:
            print(f"\n📊 Visualization saved to: {results['visualization_path']}")
        
        print(f"\n{'='*60}")
        print(f"Analysis Date: {results['date_analyzed']}")
        print(f"{'='*60}")
        
        # Save results to file if requested
        if args.save_results:
            # Create a custom encoder for NumPy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            # Save to file
            safe_name = args.app_name.replace(" ", "_").replace("/", "_")
            results_file = f"reports/{safe_name}_analysis.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
                
            print(f"\n💾 Results saved to: {results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        return 130
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
        return 1
        
if __name__ == "__main__":
    sys.exit(main()) 
