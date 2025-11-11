import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MeetingAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.models = {}
        self.tokenizers = {}
        self.categories = ['Goals', 'Achievements', 'Challenges', 'Feedback']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_models(self):
        """Load the trained models"""
        logger.info(f"Loading models from: {self.model_path}")
        
        # Load metadata
        with open(f"{self.model_path}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Model Version: {metadata['version']}")
        logger.info(f"Trained on: {metadata['training_timestamp']}")
        
        # Load each model
        model_names = metadata['models']
        for model_name in model_names:
            safe_name = model_name.replace('/', '_')
            model_dir = f"{self.model_path}/{safe_name}"
            
            logger.info(f"Loading: {model_name}")
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_dir)
            self.models[model_name] = AutoModelForSequenceClassification.from_pretrained(
                model_dir
            ).to(self.device)
            self.models[model_name].eval()
        
        logger.info("✓ All models loaded successfully!")
        
    def predict(self, text):
        """Predict categories for a single text"""
        all_predictions = []
        
        for model_name, model in self.models.items():
            tokenizer = self.tokenizers[model_name]
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, 
                             truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.sigmoid(outputs.logits).cpu().numpy()[0]
                all_predictions.append(predictions)
        
        # Average predictions from all models
        final_pred = np.mean(all_predictions, axis=0)
        
        # Get categories above threshold
        threshold = 0.45
        results = {}
        for i, category in enumerate(self.categories):
            if final_pred[i] > threshold:
                results[category] = float(final_pred[i])
        
        return results
    
    def analyze_csv(self, csv_path, output_path="analysis_output.csv"):
        """Analyze entire CSV file"""
        logger.info(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        logger.info(f"Processing {len(df)} rows...")
        
        # Add columns for each category
        for category in self.categories:
            df[f'{category}_Score'] = 0.0
            df[f'{category}_Found'] = 'No'
        
        # Process each row
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{len(df)}")
            
            text = str(row.get('Text', ''))
            predictions = self.predict(text)
            
            # Fill in predictions
            for category, score in predictions.items():
                df.at[idx, f'{category}_Score'] = score
                df.at[idx, f'{category}_Found'] = 'Yes'
        
        # Save results
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"✓ Results saved to: {output_path}")
        
        # Print summary
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        for category in self.categories:
            found_count = (df[f'{category}_Found'] == 'Yes').sum()
            if found_count > 0:
                avg_score = df[df[f'{category}_Found'] == 'Yes'][f'{category}_Score'].mean()
                print(f"{category}: {found_count} segments (avg: {avg_score:.2%})")
        
        print("="*60 + "\n")

# ==========================
# HOW TO USE
# ==========================

if __name__ == "__main__":
    # Your model path
    MODEL_PATH = r"C:\\Users\\shath\\OneDrive\\Documents\\opsht\\UseModel\\saved_models"
    
    # Your CSV file path
    CSV_PATH = "C:\\Users\\shath\\OneDrive\\Documents\\opsht\\diarization\\meeting_transcript.csv"  # Update this!
    
    # Create analyzer
    analyzer = MeetingAnalyzer(MODEL_PATH)
    
    # Load models
    analyzer.load_models()
    
    # Analyze your CSV
    results = analyzer.analyze_csv(CSV_PATH, "meeting_analysis_results.csv")
    
    print("\n✓ Done! Check 'meeting_analysis_results.csv' for results")