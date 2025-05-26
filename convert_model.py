import pickle
import xgboost as xgb
import os
import numpy as np

def convert_model():
    try:
        print("Loading old model...")
        # Try to load the model with different methods
        try:
            # Method 1: Direct pickle load
            with open('xgb_model.pkl', 'rb') as f:
                model = pickle.load(f)
                print("Model loaded successfully with pickle")
        except Exception as e1:
            print(f"Pickle load failed: {str(e1)}")
            try:
                # Method 2: XGBoost native load
                model = xgb.Booster()
                model.load_model('xgb_model.pkl')
                print("Model loaded successfully with XGBoost native format")
            except Exception as e2:
                print(f"XGBoost native load failed: {str(e2)}")
                raise Exception("Both loading methods failed")

        # Save in multiple formats
        print("Saving model in multiple formats...")
        
        # Save as JSON
        model.save_model('xgb_model.json')
        print("Saved as JSON format")
        
        # Save as binary
        model.save_model('xgb_model.bin')
        print("Saved as binary format")
        
        # Save as pickle with protocol 4
        with open('xgb_model_new.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=4)
        print("Saved as new pickle format")
        
        print("Model conversion completed successfully!")
        
    except Exception as e:
        print(f"Error in conversion process: {str(e)}")
        print("Please check if the model file is corrupted or in an incompatible format.")

if __name__ == "__main__":
    convert_model() 