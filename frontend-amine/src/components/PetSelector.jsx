const PETS = { cat: "🐱", dog: "🐶" };

export default function PetSelector({ petType, onPetChange }) {
  return (
    <div className="pet-selector">
      {Object.entries(PETS).map(([key, emoji]) => (
        <button
          key={key}
          className={`pet-btn ${petType === key ? "active" : ""}`}
          onClick={() => onPetChange(key)}
        >
          {emoji}
        </button>
      ))}
    </div>
  );
}
