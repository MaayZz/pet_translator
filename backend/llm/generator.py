"""
LLM text generator.

When the fine-tuned Llama model is ready, this module will load the GGUF file
via llama-cpp-python and generate responses.

For now, it returns mock responses so the frontend can be developed independently.
"""

import random

MOCK_RESPONSES = {
    "cat": {
        "snobbish": {
            "hunger": "Enfin, tu te décides à nourrir ton roi.",
            "play": "Ce bout de papier bouge. Intéressant.",
            "attention": "Tu peux me caresser, je t'accorde cette faveur.",
            "fear": "Il y a un intrus dans mon royaume.",
            "pain": "Quelque chose cloche dans mon domaine.",
            "content": "Tu as mes faveurs pour aujourd'hui.",
        },
        "timid": {
            "hunger": "Euh... pardon... est-ce que je pourrais avoir à manger ?",
            "play": "Si tu veux jouer... c'est d'accord...",
            "attention": "Tu es là... je suis content...",
            "fear": "J'ai entendu un bruit... j'ai peur...",
            "pain": "Ça fait mal... je ne sais pas quoi faire...",
            "content": "Je suis bien là... tout doux...",
        },
        "grumpy": {
            "hunger": "Encore à la bourre pour mon repas. Pathétique.",
            "play": "Tu veux jouer ? Tu as 30 secondes.",
            "attention": "Quoi encore ? Dépêche-toi.",
            "fear": "Y'a un bruit. Va voir. Maintenant.",
            "pain": "Quelque chose ne va pas. Bouge-toi.",
            "content": "Silence. Je dors. Enfin.",
        },
    },
    "dog": {
        "excited": {
            "hunger": "À MANGER ! À MANGER ! OUI OUI OUI !",
            "play": "JOUE AVEC MOI ! Balle balle balle balle !",
            "attention": "REGARDE-MOI ! Je suis LÀ !",
            "fear": "J'ai peur... RESTE AVEC MOI !",
            "pain": "Aïe aïe aïe... ça fait mal...",
            "content": "Je t'aime. Tu es le meilleur. La vie est belle.",
        },
        "playful": {
            "hunger": "Un snack ? Pour jouer après ? Oui ?",
            "play": "Encore une balle ? Encore ? S'IL TE PLAÎT !",
            "attention": "Hé hé hé ! Tu m'as vu ? J'ai fait un truc cool !",
            "fear": "C'est quoi ce bruit ? On va voir ? Ensemble ?",
            "pain": "Je me suis fait mal en jouant...",
            "content": "Tout est bien quand on est ensemble !",
        },
        "gentle": {
            "hunger": "Je mangerais bien un peu, si tu as le temps.",
            "play": "Je veux bien jouer avec toi, doucement.",
            "attention": "Je suis là, près de toi. Je t'aime.",
            "fear": "J'ai un peu peur. Tu me protèges ?",
            "pain": "Je ne me sens pas bien. Reste avec moi.",
            "content": "Tout va bien. Je suis heureux avec toi.",
        },
    },
}

FALLBACK = "Miaou."

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
    for pet_type, personalities in MOCK_RESPONSES.items():
        for personality, categories in personalities.items():
            for category, response in categories.items():
                if category in prompt or personality in prompt or pet_type in prompt:
                    if random.random() < 0.3 and category != "content":
                        alt = list(categories.values())
                        alt.remove(response)
                        return random.choice(alt)
                    return response
    return FALLBACK
