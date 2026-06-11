import os

def main():
    """
    Inference script that reads INPUT_TEXT and HF_TOKEN from environment variables.
    This script is designed to be run by GitHub Actions workflow.
    """
    input_text = os.getenv("INPUT_TEXT")
    hf_token = os.getenv("HF_TOKEN")
    
    if not input_text:
        print("Error: INPUT_TEXT environment variable not set")
        return
    
    print(f"Input: {input_text}")
    print(f"HF Token loaded: {hf_token is not None}")
    
    # Add your actual inference logic here
    # Example:
    # from transformers import pipeline
    # classifier = pipeline("sentiment-analysis", model="your-model", use_auth_token=hf_token)
    # result = classifier(input_text)
    # print(f"Result: {result}")

if __name__ == "__main__":
    main()
