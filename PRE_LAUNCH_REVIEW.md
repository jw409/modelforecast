# 🚀 Pre-Launch Review: modelforecast

**Status**: Staged & Ready
**Last Updated**: Monday, Dec 29, 2025

## 🛠️ Changes Staged for Review
1.  **News Refresh**: `NEWS.md` updated for Dec 29 ("Dominance and Evolution" edition).
2.  **Visualizer Sync**:
    -   1,054 battle recordings copied to `docs/corewars/recordings/tournament_001/`.
    -   Tournament JSONs (including Dec 28) copied to `docs/corewars/tournaments/`.
    -   Evolved warriors copied to `docs/corewars/warriors/tournament_001/`.
3.  **Path Normalization**:
    -   `index.html` updated: `gpu_mars/warriors/` -> `warriors/`.
    -   `tournament_loader.js` updated: `var/tournaments/` -> `tournaments/`.
    -   `manifest.json` updated to remove `var/` prefixes.
4.  **Landing Page**: Created `docs/index.html` to auto-redirect to the visualizer.
5.  **LLM Registry**: Added `Obsidian Breaker v4` to the competition array.

## 🔍 Verification Checklist for Tomorrow
- [ ] **Static Serving**: Run `python3 -m http.server -d external/modelforecast/docs` and visit `http://localhost:8000`.
- [ ] **Recording Load**: Click a battle in the 3D grid and verify the recording actually plays.
- [ ] **Warrior Code**: Open the "Show Code" panel and verify `metacog_obsidian_breaker_v4.red` loads.
- [ ] **Leaderboard**: Verify DeepSeek V3 is correctly showing at the top.

## 📡 Deployment Instructions
When you are ready to go live:
```bash
cd external/modelforecast
git push origin main
```
This will trigger GitHub Pages to serve the `/docs` folder.

---
*Note: This file is for review purposes and should be deleted before the final push if you want a clean repo root.*
