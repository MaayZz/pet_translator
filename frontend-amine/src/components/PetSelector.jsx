import { useState } from "react";

const PETS = {
  cat: { emoji: "🐱", label: "Chat", personalities: ["snobbish", "timid", "grumpy"] },
  dog: { emoji: "🐶", label: "Chien", personalities: ["excited", "playful", "gentle"] },
};

const PERSONALITY_LABELS = {
  snobbish: "Hautain",
  timid: "Timide",
  grumpy: "Grumpy",
  excited: "Surexcité",
  playful: "Joueur",
  gentle: "Doux",
};

export default function PetSelector({ petType, personality, onPetChange, onPersonalityChange }) {
  return (
    <div className="pet-selector">
      <div className="selector-group">
        <label>Animal</label>
        <div className="pet-buttons">
          {Object.entries(PETS).map(([key, pet]) => (
            <button
              key={key}
              className={`pet-btn ${petType === key ? "active" : ""}`}
              onClick={() => onPetChange(key)}
            >
              <span className="pet-emoji">{pet.emoji}</span>
              <span>{pet.label}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="selector-group">
        <label>Personnalité</label>
        <select
          value={personality}
          onChange={(e) => onPersonalityChange(e.target.value)}
          className="personality-select"
        >
          {PETS[petType]?.personalities.map((p) => (
            <option key={p} value={p}>
              {PERSONALITY_LABELS[p]}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
