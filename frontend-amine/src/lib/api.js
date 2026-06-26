const API_BASE = "http://localhost:8000";

const MOCK_RESPONSES = {
  dog: {
    bark: "WOOF! WOOF! Listen to me!",
    growl: "Stay back... I mean it.",
    grunt: "Hmmph. Fine. Whatever.",
    uncertain: "I'm trying to say something...",
  },
  cat: {
    brushing: "Mmm yes, right there. Don't stop.",
    food: "Feed me. Now. You know the rules.",
    isolation: "Why am I alone in this room?",
    uncertain: "I have feelings too, you know.",
  },
};

export async function translate({ animal, label, confidence, probabilities, history }) {
  try {
    const res = await fetch(`${API_BASE}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        animal,
        label,
        confidence,
        probabilities: probabilities || {},
        history: (history || []).map(m => ({
          text: m.text,
          emotion: m.emotion,
          confidence: m.confidence,
          timestamp: m.timestamp,
        })),
      }),
    });
    if (!res.ok) throw new Error("API error");
    return await res.json();
  } catch {
    await new Promise((r) => setTimeout(r, 600 + Math.random() * 400));
    const fallback = (MOCK_RESPONSES[animal] || {})[label] || "🐾?";
    return {
      text: fallback,
      emotion: label,
      animal,
      confidence,
      timestamp: new Date().toISOString(),
    };
  }
}

export async function getHistory() {
  try {
    const res = await fetch(`${API_BASE}/history`);
    if (!res.ok) throw new Error("API error");
    return await res.json();
  } catch {
    return [];
  }
}
