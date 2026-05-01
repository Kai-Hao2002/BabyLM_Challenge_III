import os
import logging
from datasets import load_dataset

# Setup logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_babybabellm(output_dir="data/raw"):
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the three languages to download
    languages = ["eng", "zho", "nld"]
    
    # Loop through each language
    for lang in languages:
        logging.info(f"Starting download for {lang} data...")
        
        # Define dataset name (This line MUST be indented inside the for loop)
        dataset_name = f"BabyLM-community/babylm-{lang}"
        
        try:
            # Download and save
            dataset = load_dataset(dataset_name)
            save_path = os.path.join(output_dir, f"{lang}_dataset")
            
            dataset.save_to_disk(save_path)
            logging.info(f"Successfully saved {lang} data to {save_path}")
            
        except Exception as e:
            logging.error(f"Error downloading {lang}: {e}")

if __name__ == "__main__":
    download_babybabellm()


# Step 1: Accept the terms on the website

## You must manually visit the Hugging Face pages for these three datasets and click the agreement button 
#  (usually saying "Agree and access repository" or asking you to fill out a brief form).

#https://huggingface.co/datasets/BabyLM-community/babylm-eng
#https://huggingface.co/datasets/BabyLM-community/babylm-zho
#https://huggingface.co/datasets/BabyLM-community/babylm-nld


# Step 2: Get your Access Token

#On the Hugging Face website, click your profile picture in the top right corner and select "Settings".
#Find "Access Tokens" in the left menu.
#Click "Create new token".
#Select "Read" for the token type (that's sufficient), name it (e.g., babylm_download), and create it.
#Copy this Token (it will look like a long string starting with hf_...).


# Step 3: Login in your Terminal

# hf auth login

# python src/data_pipeline/download.py




