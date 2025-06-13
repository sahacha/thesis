import json
import uuid
import random
import os
import requests
from typing import List, Dict
api_key = os.environ.get("XAI_API_KEY")
# Call xAI Grok API with the given prompt and return the response text
def call_llm(prompt: str) -> str:
    if not api_key:
        raise RuntimeError("XAI_API_KEY environment variable not set.")
    url = "https://api.grok.x.ai/v1/chat/completions"  # Replace with actual endpoint if different
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-1",  # Replace with actual model name if needed
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    # Adjust this extraction based on actual API response structure
    return result["choices"][0]["message"]["content"]

# Context data for Chiang Mai (simplified for demo)
context_data = [
    "Chiang Mai is known for its rich culture, ancient temples, and vibrant night markets. Popular attractions include Doi Suthep, Wat Chedi Luang, and the Elephant Nature Park.",
    "Accommodation in Chiang Mai ranges from luxury resorts like Anantara to budget hostels like Hostel Backpackers. Many options offer unique cultural experiences.",
    "Chiang Mai's food scene includes Northern Thai cuisine like Khao Soi at Khun Yai and international options at The Riverside Bar & Restaurant.",
    "A typical 3-day itinerary in Chiang Mai could include temple visits, ethical elephant tours, and exploring local markets like the Night Bazaar."
]

# Function to generate dataset
def generate_dataset(num_samples: int) -> List[Dict]:
    dataset = []
    
    question_templates = [
        "What is a must-visit place in Chiang Mai?",
        "Where should I stay in Chiang Mai?",
        "Which restaurant in Chiang Mai serves great food?",
        "Can you suggest a 3-day travel itinerary for Chiang Mai?"
    ]
    
    for _ in range(num_samples):
        # Randomly select a question type
        question = random.choice(question_templates)
        
        # Generate ground truth and answer using LLM
        answer = call_llm(question)
        ground_truth = [answer]  # Single ground truth for simplicity
        
        # Select relevant context
        if "place" in question.lower():
            context = [context_data[0]]
        elif "stay" in question.lower():
            context = [context_data[1]]
        elif "restaurant" in question.lower():
            context = [context_data[2]]
        else:  # Itinerary
            context = [context_data[3]]
        
        # Create dataset entry
        entry = {
            "question": question,
            "ground_truths": ground_truth,
            "answer": answer,
            "contexts": context
        }
        
        dataset.append(entry)
    
    return dataset

# Generate 5 sample entries
dataset = generate_dataset(5)

# Save to JSON file
with open("chiangmai_tourism_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

# Print sample output
print("Generated Dataset Sample:")
for entry in dataset[:2]:  # Show first 2 entries for brevity
    print(json.dumps(entry, ensure_ascii=False, indent=2))