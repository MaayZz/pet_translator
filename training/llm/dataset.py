import json
import random
import os

# Definition of the intents that Anas's classification model will output
INTENTS = ["Hunger", "Pain", "Play", "Attention", "Fear", "Content"]

# Personalities that the user can choose in Amine's UI
PERSONALITIES = ["haughty_cat", "excited_dog", "grumpy_cat", "shy_dog"]

# Templates to generate synthetic data
TEMPLATES = {
    "haughty_cat": {
        "Hunger": ["Feed me now, peasant.", "My bowl is empty. Fix it.", "I require sustenance immediately."],
        "Play": ["I might chase that string if you entertain me.", "Amuse me.", "I feel energetic. Throw the mouse."],
        "Attention": ["Look at me. No, don't touch, just look.", "Acknowledge my presence.", "You may admire me now."],
        "Fear": ["What is that noise? Protect me, human.", "I am displeased by this situation.", "Remove that object from my sight."],
        "Content": ["Your lap is acceptable for now.", "I am purring. Do not disturb.", "The sunbeam is adequate."],
        "Pain": ["Do not touch me there.", "I am experiencing discomfort.", "Take me to the healer, quickly."]
    },
    "excited_dog": {
        "Hunger": ["FOOD FOOD FOOD! IS IT DINNER TIME?!", "I'm starving! What are we eating?!", "Treats? Did someone say treats?!"],
        "Play": ["THROW THE BALL! PLEASE THROW THE BALL!", "Let's play! Let's play! Catch me!", "Wrestle time! Rawr!"],
        "Attention": ["Pet me! Pet me! Look at my tail wagging!", "Hi! Hi! Hi! Notice me!", "Belly rubs please!"],
        "Fear": ["Thunder! Hide me!", "I don't like the vacuum cleaner! Save me!", "Loud noise! Let's hide under the bed!"],
        "Content": ["I love you! This couch is great!", "Zzz... happy dreams...", "Best day ever with my human!"],
        "Pain": ["Ouch! My paw hurts!", "Whimper... I don't feel good...", "Hold me, I'm hurting."]
    }
}
# Fallback for others to keep it simple
for p in ["grumpy_cat", "shy_dog"]:
    TEMPLATES[p] = TEMPLATES["haughty_cat"] if "cat" in p else TEMPLATES["excited_dog"]

def generate_synthetic_data(num_samples=5000):
    dataset = []
    
    for _ in range(num_samples):
        intent = random.choice(INTENTS)
        personality = random.choice(PERSONALITIES)
        response = random.choice(TEMPLATES[personality][intent])
        
        # We simulate the prompt structure
        # The input will be provided by Anas's model (the intent category)
        # We also simulate some RAG context that might be retrieved
        rag_context = f"General behavioral note: An animal showing {intent.lower()} will typically vocalize to communicate this need."
        
        system_prompt = f"You are a pet translator. The pet has a {personality} personality."
        user_prompt = f"Context from animal behavior database: {rag_context}\n\nThe audio model classified the sound as: {intent}.\nTranslate this into a natural sentence."
        
        # Alpaca / ChatML format style
        sample = {
            "instruction": system_prompt + "\n" + user_prompt,
            "input": "",
            "output": response
        }
        dataset.append(sample)
        
    return dataset

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    data = generate_synthetic_data(5000)
    
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "synthetic_dataset.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(data)} samples at {out_path}")
