# Problème : Mock Classifier non fonctionnel

## Symptôme

Quand on ouvre l'app et qu'on enregistre un son (ou qu'on upload un fichier), le
classifieur renvoie des prédictions avec une confiance de 80-99% même pour du
silence, de la musique ambiante, ou n'importe quel bruit aléatoire.

L'utilisateur voit :

> 🔊 Bark 95%

alors qu'il n'y a pas de chien.

## Cause racine

Le modèle TensorFlow.js ne peut pas être chargé sur ce Mac (TensorFlow plante
avec une erreur `mutex lock failed` sur macOS). Du coup le code tombe en
fallback sur `mockClassify()` dans `modelLoader.js`.

### Pourquoi le mock actuel est mauvais

La fonction `mockClassify()` utilise un **hash du fichier audio brut** (les
premiers bytes) pour déterminer la classe de façon déterministe :

```js
const hash = simpleHash(buf);   // hash basé sur les bytes du fichier
const idx = hash % classes.length;
// confiance = 0.82 + (hash % 15) / 100 → entre 82% et 96%
```

Ça ne regarde **pas du tout** le contenu audio :
- Pas de décodage du signal
- Pas d'analyse fréquentielle
- Pas d'énergie RMS
- Pas de zero-crossing rate

Du coup un fichier de silence pur donne exactement la même prédiction qu'un vrai
aboiement, du moment que le hash tombe sur le même index.

### Pourquoi le vrai modèle n'est pas utilisé

Le pipeline complet devrait être :

1. Audio → preprocessing JS (resample 16kHz, Hann 1024, 64 mel bands,
   10\*log10, min-max, 3 channels, resize 96x96)
2. Tensor normalisé [-1, 1] → MobileNetV2 backbone (via
   `@tensorflow-models/mobilenet`)
3. Embedding 1280-d → trained head (Dense 64 ReLU → Dense 3 Softmax) → classes

Mais le chargement du backbone MobileNetV2 via TF.js échoue silencieusement ou
n'est pas correctement initialisé, donc on tombe sur le mock.

## Ce qu'il faudrait (TODO Anas)

1. **Débugger le chargement TFJS** — Vérifier pourquoi `mobilenet.load()` ou
   `loadHeadWeights()` échoue. Console F12 → voir les erreurs.

2. **Remplacer `mockClassify()`** par une vraie inférence TFJS.

3. **En attendant** : améliorer le mock pour qu'il analyse au moins le
   contenu audio (énergie RMS, zero-crossing, bandes de fréquences) et qu'il
   mette une confiance basse (< 50%) sur le silence ou les sons non-vocaux.

## Fichiers concernés

| Fichier | Rôle |
|---------|------|
| `frontend-amine/src/lib/modelLoader.js` | Contient `mockClassify()` + preprocessing + inférence |
| `frontend-amine/public/model/{dog,cat}/head_weights.bin` | Poids du head (ok, à utiliser) |
| `frontend-amine/public/model/{dog,cat}/head_shapes.json` | Shapes du head (ok) |
| `frontend-amine/node_modules/@tensorflow-models/mobilenet` | Backbone MobileNetV2 (installé) |

## Pour reproduire

1. Lancer l'app (`npm run dev`)
2. Ouvrir la console navigateur (F12)
3. Uploader un fichier `.wav` quelconque, même du silence
4. Voir la confiance > 80% et la classe attribuée au hasard
5. Vérifier la console pour les erreurs TFJS
