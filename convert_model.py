import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

def convert_model():
    try:
        # Load the ONNX model
        onnx_file_path = "model.onnx"
        print("Loading ONNX model...")
        onnx_model = onnx.load(onnx_file_path)
        print("ONNX model loaded successfully.")

        # Create directory for SavedModel if it doesn't exist
        saved_model_path = "saved_model"
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)

        # Convert ONNX to TensorFlow SavedModel
        print("Converting to TensorFlow SavedModel format...")
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(saved_model_path)
        print(f"Model successfully converted and saved at {saved_model_path}")
        
        # Test loading the saved model
        print("Testing model loading...")
        model = tf.saved_model.load(saved_model_path)
        print("Model loaded successfully for verification.")
        
        return True
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = convert_model()
    
    if success:
        print("\nModel conversion completed successfully!")
        print("\nNext steps:")
        print("1. Your model is saved in the 'saved_model' directory")
        print("2. Run the Streamlit app using: streamlit run app.py")
    else:
        print("\nModel conversion failed. Please check the error messages above.")
