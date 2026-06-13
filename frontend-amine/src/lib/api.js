const API_BASE = "http://localhost:8000";

const MOCK_RESPONSES = {
  hunger: "I'm hungry! Feed me!",
  play: "Play with me! Come on!",
  attention: "Look at me, I'm here!",
  fear: "I'm scared... stay with me.",
  pain: "Something doesn't feel right...",
  content: "I'm happy. Everything is fine.",
};

export async function translate({ category, confidence, petType, history }) {
  try {
    const res = await fetch(`${API_BASE}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category, confidence, pet_type: petType, history }),
    });
    if (!res.ok) throw new Error("API error");
    return await res.json();
  } catch {
    await new Promise((r) => setTimeout(r, 600 + Math.random() * 400));
    return {
      text: MOCK_RESPONSES[category] || "🐾?",
      emotion: category,
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
