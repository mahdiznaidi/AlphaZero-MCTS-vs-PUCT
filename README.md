# Breakthrough 5x5 : MCTS (UCT/RAVE) vs PUCT + Réseau (style AlphaZero)

## 1. Objectif du projet

Implémenter et étudier l’impact de l’introduction d’un réseau de neurones (policy + value) dans une recherche Monte-Carlo Tree Search via PUCT, sur le jeu Breakthrough 5x5.

Le projet vise une approche inspirée d’AlphaZero, mais à échelle réaliste (solo M2) :

- **Baselines** : Random, Flat Monte-Carlo, UCT, (option : RAVE)
- **Modèle** : CNN léger policy/value
- **Recherche** : PUCT
- **Apprentissage** : self-play itératif (2–5 itérations)
- **Évaluation** : tournois + courbes de performance à budget de playout fixé
- **Objectif scientifique** : comparer la performance en fonction du budget de simulations et analyser le gain apporté par le réseau (guidage + évaluation).

## 2. Livrables

- ✅ Code complet (jeu + IA + entraînement + évaluation)
- ✅ Reproductibilité (seed, configs, logs)
- ✅ Expériences + graphiques
- ✅ Rapport (ou notebook) : analyse + discussion
- ✅ (Optionnel) petite UI console / ASCII, ou export PGN-like

## 3. Jeu : Breakthrough 5x5

### Règles (résumé)

- Plateau 5x5
- Chaque joueur contrôle des pions.
- Déplacement :
  - Avancer d’une case tout droit si vide
  - Capturer en diagonale avant-gauche / avant-droite
- Condition de victoire :
  - Atteindre la dernière rangée adverse ou
  - Capturer tous les pions adverses
- (Option) gérer situations sans coups → défaite

### Représentation recommandée

- `board`: matrice 5x5 avec valeurs `{0, +1, -1}`
- `player`: +1 ou -1 (joueur courant)

## 4. Méthodes à implémenter

### 4.1 Baselines

- Random : coup légal uniforme
- Flat Monte Carlo : choisir le coup avec meilleur score moyen via rollouts

### 4.2 MCTS classique : UCT

- Sélection : UCB1
- UCT = Q(s,a) + c * sqrt( ln(N(s)) / (1 + N(s,a)) )
- Expansion : ajout du premier enfant non exploré
- Simulation : rollout aléatoire (option : rollout biaisé “captures”)
- Backprop : mise à jour visites + valeurs

### 4.3 (Optionnel mais très utile) RAVE

Ajoute une estimation AMAF pour accélérer l’apprentissage des coups “bons”. Très pertinent à faible budget de simulations.

### 4.4 PUCT (style AlphaZero)

- Sélection :
  - PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
- `P(s,a)` provient du réseau (policy head)
- Valeur terminale ou réseau (value head) pour l’évaluation

## 5. Réseau de neurones (pragmatique)

### 5.1 Entrées

Tensor 5x5xC :

- Canal 1 : positions des pions du joueur courant
- Canal 2 : positions des pions adverses
- Canal 3 (option) : plan constant “joueur courant” (tout à 1)

### 5.2 Sorties

- **Policy head** : distribution sur un espace d’actions fixe
- **Value head** : scalaire `v ∈ [-1, 1]` (estimation de victoire)

### 5.3 Espace d’actions (important)

Pour rester simple, définir un mapping fixe :

- Pour chaque case (r,c) et pour chaque direction possible {forward, diagL, diagR}
- Total max ≈ 25 * 3 = 75 actions

Les actions illégales seront masquées (probabilité 0 après softmax masqué).

### 5.4 Architecture conseillée (simple mais efficace)

- 3 à 5 blocs Conv2D (petit nombre de filtres)
- 2 têtes :
  - Policy : Conv1x1 -> Flatten -> Dense(75)
  - Value : Conv1x1 -> Flatten -> Dense -> Dense(1) + tanh

### 5.5 Fonction de perte

- Policy loss : cross-entropy entre `π_target` (proportions de visites MCTS) et `π_pred`
- Value loss : MSE entre `z` (résultat final ±1) et `v_pred`
- Total : `L = L_policy + λ * L_value + β * L2`

## 6. Self-play : boucle d’entraînement (AlphaZero simplifié)

### 6.1 Boucle globale (2–5 itérations)

Pour `iter = 1..K` :

1. Générer `G` parties en self-play avec PUCT + réseau courant
2. Collecter dataset : `(state_t, π_t, z)` pour chaque position
3. Entraîner le réseau sur le dataset (quelques epochs)
4. Évaluer le nouveau réseau vs baselines et vs modèle précédent
5. Sauver modèle + stats

### 6.2 Paramètres réalistes (solo)

- K = 3
- G = 200 à 1000 parties / itération (selon temps)
- Simulations MCTS par coup : 50 / 100 / 200 (selon budget)
- Température :
  - haute au début (exploration)
  - basse à la fin (exploitation)
- Replay buffer : garder N dernières positions (ex : 50k)

## 7. Évaluation expérimentale (indispensable)

### 7.1 Questions de recherche

- À budget fixe de simulations, PUCT + NN bat-il UCT ?
- À quel point le NN aide-t-il :
  - la policy (guidage)
  - la value (moins de rollouts nécessaires)
- Quel est l’effet de :
  - `c_uct / c_puct`
  - nombre de simulations
  - rollouts aléatoires vs value head

### 7.2 Protocole

- Jouer des matches en miroir (couleurs alternées)
- Report :
  - Winrate
  - Intervalle de confiance (option)
  - Temps moyen par coup

### 7.3 Matrice de tournois minimale

- Random vs Flat MC
- Flat MC vs UCT(200)
- UCT(200) vs RAVE(200) (option)
- UCT(200) vs PUCT-NN(200)
- PUCT-NN(50) vs UCT(200) (pour montrer l’efficacité au faible budget)

### 7.4 Graphiques attendus

- Winrate vs nombre de simulations (courbe)
- Elo approximatif (option)
- Temps de calcul vs performance

## 8. Plan B (anti-risque)

Si l’entraînement self-play est trop long / instable :

- **Plan B1** : PUCT + NN entraîné sur données heuristiques
  - Générer dataset via UCT fort
  - Entraîner policy/value en supervision
  - Tester PUCT avec ce NN
- **Plan B2** : PUCT avec policy prior “handcrafted”
  - Prioriser captures / avancée centrale
  - Montrer que PUCT > UCT même avec un prior simple

Le projet reste valide car l’étude principale = PUCT + priors, et comparaison expérimentale.

## 9. Structure du dépôt

```
breakthrough-puct/
  README.md
  requirements.txt
  configs/
    default.yaml
    uct.yaml
    puct.yaml
    train.yaml
  src/
    game/
      breakthrough.py        # règles, coups, terminal, winner
      encoding.py            # state -> tensor
      action_space.py        # mapping action<->index, mask
    mcts/
      node.py
      uct.py
      rave.py                # optionnel
      puct.py
      rollout.py
    nn/
      model.py               # CNN policy/value
      train.py               # training loop
      replay_buffer.py
    selfplay/
      generate.py            # self-play workers
      arena.py               # matches et tournois
    utils/
      seed.py
      logger.py
      metrics.py
  scripts/
    run_selfplay.sh
    run_tournament.sh
  outputs/
    models/
    logs/
    plots/
```

## 10. Commandes

### Installation

```bash
pip install -r requirements.txt
```

### Lancer un tournoi baseline

```bash
python -m src.selfplay.arena --config configs/uct.yaml
```

### Générer du self-play

```bash
python -m src.selfplay.generate --config configs/train.yaml
```

### Entraîner le réseau

```bash
python -m src.nn.train --config configs/train.yaml
```

## 11. Checklist de progression (milestones)

### Semaine 1 — Jeu + baselines

- [ ] Breakthrough 5x5 complet
- [ ] Random + Flat MC
- [ ] Tests unitaires (coups légaux, terminal)

### Semaine 2 — MCTS UCT + évaluation

- [ ] UCT stable + tournois
- [ ] Logging + plots

### Semaine 3 — PUCT (sans NN)

- [ ] PUCT avec priors uniformes
- [ ] Comparaison UCT vs PUCT (doit être similaire)

### Semaine 4 — Réseau + intégration

- [ ] Policy/value network
- [ ] Masquage actions illégales
- [ ] PUCT + NN inference

### Semaine 5 — Self-play itératif + analyse

- [ ] 2–5 itérations
- [ ] courbes / tables
- [ ] rapport final

## 12. Rapport : structure conseillée

- Introduction (jeu, MCTS, motivation PUCT)
- Méthodes (UCT, PUCT, NN, self-play)
- Implémentation (action space, encoding, paramètres)
- Expériences (protocole, résultats, figures)
- Discussion (ce qui marche/pas, biais, limites, coût calcul)
- Conclusion + perspectives (RAVE/GRAVE, plus grand plateau, meilleurs rollouts)

## 13. Paramètres initiaux (bons defaults)

- UCT : `c_uct = 1.4` (à tuner)
- PUCT : `c_puct = 1.0 à 2.5` (à tuner)
- Simulations : 100 / 200
- Rollout max depth : 50 (pour éviter boucles)
- Température : `1.0` (début), `0.1` (fin)
- Optim : Adam
- LR : `1e-3` (descendre si instable)
- Batch : 128
- Epochs : 5–10 / itération

## 14. Exemples rapides

- **Jouer un match Random vs UCT (20 parties)**

```bash
python -m src.selfplay.arena --config configs/uct.yaml --matches 20
```

- **Générer un dataset de self-play (50 parties)**

```bash
python -m src.selfplay.generate --config configs/train.yaml --games 50
```

- **Entraîner le réseau sur un dataset existant**

```bash
python -m src.nn.train --config configs/train.yaml --dataset outputs/logs/selfplay_dataset.npz
```

---

Pour toute question sur le pipeline, consultez les fichiers `configs/*.yaml` et les modules `src/`.
