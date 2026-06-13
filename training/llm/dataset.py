import json
import random
import os

# Definition of the intents that Anas's classification model will output
INTENTS = ["Hunger", "Pain", "Play", "Attention", "Fear", "Content"]

# Personalities that the user can choose in Amine's UI
PERSONALITIES = ["haughty_cat", "excited_dog", "grumpy_cat", "shy_dog"]

ENV_OPTIONS = {
    "location": ["indoor", "outdoor"],
    "weather": ["sunny", "rainy", "snowy", "thunderstorm"],
    "time_of_day": ["morning", "afternoon", "night"],
    "other_animals": ["none", "another dog", "a cat"]
}

def generate_response(personality, intent, env):
    base_responses = {
        "haughty_cat": {
            "Hunger": ["Feed me now, peasant.", "My bowl is empty. Fix it."],
            "Play": ["Amuse me.", "I feel energetic. Throw the mouse."],
            "Attention": ["Acknowledge my presence.", "You may admire me now."],
            "Fear": ["I am displeased by this situation.", "Remove that object from my sight."],
            "Content": ["Your lap is acceptable for now.", "The sunbeam is adequate."],
            "Pain": ["Do not touch me there.", "I am experiencing discomfort."]
        },
        "excited_dog": {
            "Hunger": ["FOOD FOOD FOOD!", "I'm starving! What are we eating?!"],
            "Play": ["THROW THE BALL!", "Let's play! Catch me!"],
            "Attention": ["Pet me! Look at my tail wagging!", "Notice me!"],
            "Fear": ["I don't like this! Save me!", "Loud noise! Let's hide!"],
            "Content": ["I love you! This is great!", "Zzz... happy dreams..."],
            "Pain": ["Ouch! My paw hurts!", "Whimper... I don't feel good..."]
        }
    }
    
    # Map all cats to haughty_cat and dogs to excited_dog logic for simplicity, just change the tone slightly
    base_dict = base_responses["haughty_cat"] if "cat" in personality else base_responses["excited_dog"]
    
    response = random.choice(base_dict[intent])
    
    # Contextual modifiers
    if env["weather"] == "rainy" and env["location"] == "outdoor":
        if "cat" in personality:
            response += " And I am getting wet. This is unacceptable."
        else:
            response += " I'm all wet! Let's go inside!"
            
    if env["weather"] == "thunderstorm" and intent == "Fear":
        if "dog" in personality:
            response = "THUNDER! Hide me under the bed immediately!"
            
    if env["other_animals"] != "none" and intent == "Attention":
        if "cat" in personality:
            response += f" Ignore {env['other_animals']} and look only at me."
        else:
            response += f" Look at me, not at {env['other_animals']}! Me!"
            
    if env["time_of_day"] == "night" and intent == "Content":
        response = "Goodnight human. Time to sleep."
        
    return response

def generate_synthetic_data(num_samples=5000):
    dataset = []
    
    for _ in range(num_samples):
        intent = random.choice(INTENTS)
        personality = random.choice(PERSONALITIES)
        
        env = {k: random.choice(v) for k, v in ENV_OPTIONS.items()}
        
        response = generate_response(personality, intent, env)
        
        rag_context = f"General behavioral note: An animal showing {intent.lower()} will typically vocalize to communicate this need."
        
        system_prompt = f"You are a pet translator. The pet has a {personality} personality."
        
        env_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in env.items()])
        user_prompt = f"The audio classification model detected the following intent: {intent}\n\nCurrent Environmental Context:\n{env_text}\n\nPlease ensure your translation logically reflects this environment (e.g. reacting to rain, time of day, or other animals).\n\nContext from behavioral database:\n{rag_context}\n\nTranslate this intent into a short, natural sentence representing what the pet is trying to say."
        
        sample = {
            "instruction": system_prompt + "\n\n" + user_prompt,
            "input": "",
            "output": response
        }
        dataset.append(sample)
        
    return dataset

if __name__ == "__main__":
    print("Generating synthetic dataset with environmental context...")
    data = generate_synthetic_data(5000)
    
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "synthetic_dataset.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(data)} context-rich samples at {out_path}")
