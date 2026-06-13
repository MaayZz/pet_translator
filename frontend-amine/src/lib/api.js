const API_BASE = "http://localhost:8000";

const MOCK_RESPONSES = {
  hunger: "J'ai faim ! Donne-moi à manger !",
  play: "Joue avec moi ! Allez !",
  attention: "Regarde-moi, je suis là !",
  fear: "J'ai peur... viens me rassurer.",
  pain: "Quelque chose ne va pas...",
  content: "Je suis bien, tout va bien.",
};

export async function translate({ category, confidence, petType, personality, history }) {
  try {
    const res = await fetch(`${API_BASE}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category, confidence, pet_type: petType, personality, history }),
    });
    if (!res.ok) throw new Error("API error");
    return await res.json();
  } catch {
    await new Promise((r) => setTimeout(r, 600 + Math.random() * 400));
    return {
      text: MOCK_RESPONSES[category] || "Miaou ?",
      emotion: category,
      confidence: confidence,
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
