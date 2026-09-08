# Project_pKY_166 - Polished

This repository is a polished copy of the original Project_pKY_166 with frontend and backend fixes applied to make the app more robust and ready for local production testing.

What I changed

- Frontend
  - Configurable API base / WebSocket base via Vite env (VITE_API_BASE / VITE_API_PORT).
  - Robust API client that safely handles empty responses and surfaces errors.
  - Fixed repeated fetch issues and wired core UI buttons (runs, schedules, approvals).

- Backend
  - WebSocket robustness: validated messages, supports sync and async approval resolution, sends termination event, and cleans up active runs.
  - Model registry reads Ollama host from settings/env and returns friendly errors when unreachable.
  - Session & background task hardening.

How to run locally

Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend

```bash
cd frontend
npm install
# optionally create a .env with VITE_API_PORT=8000
npm run dev
```

Docker (example)

```bash
# from repo root
docker-compose up --build
```

Smoke tests

- Health: `curl http://localhost:8000/api/health`
- List projects: `curl http://localhost:8000/api/projects`

If you want me to change anything (make private, tweak Docker, add CI), tell me and I will update the copy.
