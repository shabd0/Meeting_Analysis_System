from transformers import pipeline
import torch
import pandas as pd
from huggingface_hub import login

# Login to Hugging Face (do this once at startup)
def initialize_hf():
    login("add_yours_key")

# Initialize generator (global, load once)
generator = None

def load_model():
    """Load the LLM model once"""
    global generator
    if generator is None:
        generator = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            device=0 if torch.cuda.is_available() else -1
        )
    return generator

def generate_recommendations(df: pd.DataFrame, custom_prompt: str = None) -> str:
    """
    Generate AI recommendations from meeting analysis DataFrame
    
    Args:
        df: DataFrame with meeting analysis results
        custom_prompt: Optional custom prompt for specific recommendations
        
    Returns:
        str: Generated recommendations text
    """
    # Load model if not already loaded
    gen = load_model()
    
    # Prepare meeting summary
    meeting_summary = ""
    for _, row in df.iterrows():
        meeting_summary += (
            f"[{row['Speaker']}] {row['Text']} "
            f"| Goals: {row.get('Goals_Found', 'N/A')} "
            f"| Achievements: {row.get('Achievements_Found', 'N/A')} "
            f"| Challenges: {row.get('Challenges_Found', 'N/A')} "
            f"| Feedback: {row.get('Feedback_Found', 'N/A')}\n"
        )
    
    # Default prompt or custom
    if custom_prompt:
        prompt = f"""
You are an AI meeting coach. Here is the analyzed meeting data:

{meeting_summary}

{custom_prompt}
"""
    else:
        prompt = f"""
You are an AI meeting coach. Here is the analyzed meeting data:

{meeting_summary}

Based on this, please give structured RECOMMENDATIONS with these sections:
1. Key action items
2. Communication improvements
3. Challenges that need follow-up
4. Suggestions for better productivity
"""
    
    # Generate recommendations
    result = gen(
        prompt,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7
    )
    
    recommendations = result[0]["generated_text"]
    
    return recommendations