"""
LLM text generator.

When the fine-tuned Llama model is ready, this module will load the GGUF file
via llama-cpp-python and generate responses.

For now, it returns mock responses so the frontend can be developed independently.
"""

import random

MOCK_RESPONSES = {
    "cat": {
        "cat_snobbish": {
            "hunger": "Finally, you decide to feed your king.",
            "play": "That piece of string moves. Interesting.",
            "attention": "You may pet me, I grant you this favor.",
            "fear": "There is an intruder in my kingdom.",
            "pain": "Something is wrong in my domain.",
            "content": "You have my favor for today.",
        },
        "cat_timid": {
            "hunger": "Um... sorry... could I have some food?",
            "play": "If you want to play... that's okay...",
            "attention": "You're here... I'm happy...",
            "fear": "I heard a noise... I'm scared...",
            "pain": "It hurts... I don't know what to do...",
            "content": "I'm cozy right here... nice and soft...",
        },
        "cat_grumpy": {
            "hunger": "Late for my meal again. Pathetic.",
            "play": "You want to play? You have 30 seconds.",
            "attention": "What now? Hurry up.",
            "fear": "There's a noise. Go check. Now.",
            "pain": "Something's wrong. Move it.",
            "content": "Silence. I'm sleeping. Finally.",
        },
    },
    "dog": {
        "dog_excited": {
            "hunger": "FOOD! FOOD! YES YES YES!",
            "play": "PLAY WITH ME! Ball ball ball ball!",
            "attention": "LOOK AT ME! I'm HERE!",
            "fear": "I'm scared... STAY WITH ME!",
            "pain": "Ouch ouch ouch... that hurts...",
            "content": "I love you. You're the best. Life is beautiful.",
        },
        "dog_playful": {
            "hunger": "A snack? To play after? Yes?",
            "play": "Another ball? Again? PLEASE!",
            "attention": "Hey hey hey! Did you see me? I did something cool!",
            "fear": "What's that noise? Should we check? Together?",
            "pain": "I hurt myself while playing...",
            "content": "Everything is good when we're together!",
        },
        "dog_gentle": {
            "hunger": "I'd love a little food, if you have time.",
            "play": "I'd like to play with you, gently.",
            "attention": "I'm here, next to you. I love you.",
            "fear": "I'm a bit scared. Will you protect me?",
            "pain": "I don't feel well. Stay with me.",
            "content": "Everything is fine. I'm happy with you.",
        },
    },
}

def generate_response(prompt: str) -> str:
    try:
        from llama_cpp import Llama
        llm = Llama(model_path="models/llama.gguf")
        output = llm(
            prompt,
            max_tokens=30,
            temperature=0.8,
            stop=["\n", ".", "!"],
        )
        return output["choices"][0]["text"].strip()
    except (ImportError, FileNotFoundError, Exception):
        pass

    return _mock_response(prompt)

def _mock_response(prompt: str) -> str:
    is_dog = "dog" in prompt
    pet_key = "dog" if is_dog else "cat"
    responses = MOCK_RESPONSES.get(pet_key, {})
    category_map = {"hungry": "hunger", "play": "play", "attention": "attention",
                    "scared": "fear", "pain": "pain", "happy": "content", "relaxed": "content"}
    for word, cat_key in category_map.items():
        if word in prompt:
            for personality_key, categories in responses.items():
                if cat_key in categories:
                    return categories[cat_key]
    return "Woof."
