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

def build_user_prompt(intent, rag_context, env_context=None, confidence=None, probabilities=None):
    prompt = f"The audio classification model detected the primary intent: {intent}\n"

    if confidence is not None:
        prompt += f"Model Confidence: {confidence:.2f}\n"

    if probabilities:
        prompt += "Detailed class probabilities:\n"
        for k, v in probabilities.items():
            prompt += f"- {k}: {v:.2f}\n"

    prompt += "\nTONE INSTRUCTIONS BASED ON CONFIDENCE:\n"
    if intent == "uncertain":
        prompt += "- The model is unsure (uncertain). Give a fun, generic response acknowledging the pet has something to say, without claiming exactly what.\n"
    elif probabilities and len(probabilities) > 1:
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        margin = sorted_probs[0][1] - sorted_probs[1][1]
        if margin >= 0.2:
            prompt += f"- The model is highly confident. Be assertive and direct about the intent '{intent}'.\n"
        else:
            prompt += f"- The model is torn between '{sorted_probs[0][0]}' and '{sorted_probs[1][0]}'. You must nuance your response (e.g. 'Maybe X, or perhaps Y'). Keep a playful/cautious tone, avoid categorical statements.\n"
    prompt += "\n"

    if env_context:
        prompt += "Current Environmental Context:\n"
        for key, value in env_context.items():
            prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
        prompt += "\nPlease ensure your translation logically reflects this environment (e.g. reacting to rain, time of day, or other animals).\n\n"

    if rag_context and "No RAG database available" not in rag_context:
        prompt += f"Context from behavioral database:\n{rag_context}\n\n"

    prompt += "Translate this intent into a short, natural sentence representing what the pet is trying to say."
    return prompt
