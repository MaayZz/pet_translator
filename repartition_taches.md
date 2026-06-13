# Répartition des Tâches (Équipe de 4 étudiants)

Ce document détaille les responsabilités spécifiques pour chaque membre de l'équipe concernant le projet **Pet Translation Device**.

## Étudiant 1 : Traitement du Signal Audio et VAD (Voice Activity Detection)
**Rôle :** Responsable de la capture et de la préparation des données audio brutes.
*   **Tâches :**
    *   Mise en place du pipeline d'acquisition audio (datasets existants sur les animaux ou enregistrements réels).
    *   Développement de la réduction de bruit ambiant (denoising).
    *   Implémentation d'un VAD pour isoler exactement le moment où l'animal émet un son.
*   **Charge de travail estimée :** 25%

## Étudiant 2 : Modèles de Classification & Interprétation (Core ML)
**Rôle :** Responsable du "cerveau" analytique qui donne un sens au son.
*   **Tâches :**
    *   Sélection et entraînement ou fine-tuning d'un modèle de classification audio.
    *   Création de catégories d'intentions (ex: Faim, Douleur, Jeu, Menace).
    *   Évaluation des performances du modèle (Précision, Rappel) et gestion des faux positifs.
*   **Charge de travail estimée :** 25%

## Étudiant 3 : Interaction LLM & Génération de personnalité
**Rôle :** Responsable de la "traduction" humaine.
*   **Tâches :**
    *   Intégration d'un LLM (ex: via API ou modèle local) pour convertir la catégorie de classification en une phrase naturelle et amusante.
    *   Prompt engineering pour donner une personnalité spécifique à l'animal (ex: chat hautain, chien surexcité).
    *   Génération de l'historique de chat pour donner l'illusion d'une vraie conversation.
*   **Charge de travail estimée :** 25%

## Étudiant 4 : Déploiement Edge AI & Application Mobile
**Rôle :** Responsable de l'intégration finale et de l'expérience utilisateur.
*   **Tâches :**
    *   Optimisation des modèles pour les faire tourner au maximum en local (quantization, conversion en formats mobiles).
    *   Création d'une interface utilisateur simple (Application ou mock-up interactif) simulant l'écran du smartphone (le "chat" avec le pet).
    *   Gestion de la communication entre le flux audio, le modèle local et l'appel LLM.
*   **Charge de travail estimée :** 25%
