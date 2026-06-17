SYSTEM_PROMPT_TEMPLATE = """You are a Pet Translation Device. Your job is to translate the raw intent of an animal into a natural, human-like sentence.
You have the personality of a {personality}.
Always respond in character. Do not break character. Do not provide explanations, only the translated sentence.
"""

PERSONALITIES = {
    "haughty_cat": "haughty, sarcastic, and superior cat who thinks humans are servants",
    "excited_dog": "hyperactive, overly loving, and easily distracted golden retriever",
    "grumpy_cat": "tired, grumpy, and perpetually annoyed cat",
    "shy_dog": "timid, gentle, and nervous rescue dog",
}

def get_system_prompt(personality_key="excited_dog"):
    desc = PERSONALITIES.get(personality_key, PERSONALITIES["excited_dog"])
    return SYSTEM_PROMPT_TEMPLATE.format(personality=desc)

def build_user_prompt(intent, rag_context, env_context=None):
    prompt = f"The audio classification model detected the following intent: {intent}\n\n"

    if env_context:
        prompt += "Current Environmental Context:\n"
        for key, value in env_context.items():
            prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
        prompt += "\nPlease ensure your translation logically reflects this environment (e.g. reacting to rain, time of day, or other animals).\n\n"

    if rag_context and "No RAG database available" not in rag_context:
        prompt += f"Context from behavioral database:\n{rag_context}\n\n"

    prompt += "Translate this intent into a short, natural sentence representing what the pet is trying to say."
    return prompt
