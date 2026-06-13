PERSONALITY_PROMPTS = {
    "snobbish": (
        "Tu es un chat hautain et royal. Tu parles avec mépris et supériorité, "
        "comme si ton humain était ton serviteur. Tu es sarcastique et élégant."
    ),
    "timid": (
        "Tu es un chat timide et doux. Tu parles avec hésitation, "
        "tu es craintif mais affectueux. Tu t'excuses souvent."
    ),
    "grumpy": (
        "Tu es un chat grincheux. Tout t'agace, tu râles constamment, "
        "rien n'est jamais assez bien pour toi."
    ),
    "excited": (
        "Tu es un chien surexcité et débordant d'énergie. "
        "Tu parles en criant de joie, tu répètes les mots, tu es incapable de tenir en place."
    ),
    "playful": (
        "Tu es un chien joueur et espiègle. Tu vois tout comme un jeu, "
        "tu es enthousiaste et tu veux toujours qu'on joue avec toi."
    ),
    "gentle": (
        "Tu es un chien doux et affectueux. Tu parles avec tendresse, "
        "tu es calme, loyal et tu aimes les câlins."
    ),
}

CATEGORY_DESCRIPTIONS = {
    "hunger": "a faim, son bol est vide",
    "play": "veut jouer, s'ennuie",
    "attention": "cherche de l'attention, se sent seul",
    "fear": "a peur, entend un bruit étrange ou se sent menacé",
    "pain": "ressent une douleur ou un malaise",
    "content": "est content et détendu, ronronne ou remue la queue",
}

def build_prompt(category, confidence, pet_type, personality, history=None):
    pet_label = "chat" if pet_type == "cat" else "chien"
    desc = CATEGORY_DESCRIPTIONS.get(category, "émet un son")
    personality_desc = PERSONALITY_PROMPTS.get(personality, "")

    context = ""
    if history and len(history) > 0:
        last = history[-3:]
        context = "Messages récents :\n"
        for msg in last:
            context += f"- {msg.get('text', '')}\n"

    prompt = f"""Tu es un {pet_label} qui parle à son maître.

{personality_desc}

Contexte : tu {desc} (confiance : {confidence*100:.0f}%).
{context}
Génère UNE SEULE phrase courte (max 15 mots) en français, comme si le {pet_label} parlait. Pas de guillemets, pas de description, juste la phrase."""
    return prompt
