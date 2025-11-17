# Intelligent NPC — Desktop Launcher (pywebview)

This repository contains a small Python launcher that opens the existing frontend demo in a desktop window using `pywebview`.

Two modes are supported:

- Development: start your frontend dev server (e.g. Vite) with `npm run dev` and then run the Python launcher.
- Production: build the frontend into a `dist/` folder (e.g. `npm run build`), then run the Python launcher which will serve `dist/` locally and open it in a window.

Setup (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install Python deps:

```powershell
pip install -r requirements.txt
```

Development workflow

1. Start your frontend dev server (from the project root). If you use Vite, this is typically:

```powershell
npm install
npm run dev
```

2. In another terminal, run the Python launcher:

```powershell
python app.py
```

Production workflow

1. Build the frontend into `dist/` (example for Vite):

```powershell
npm install
npm run build
```

2. Run the Python launcher which will serve `dist/` and open the app:

```powershell
python app.py
```

Notes

- The launcher expects an `index.html` at `dist/index.html` for the production path.
- If no `dist/` is present, `app.py` tries to start `npm run dev` and waits for `http://127.0.0.1:5173/` to respond (Vite default). Adjust your dev server port/script if different.
- To create a standalone executable, consider packaging with `pyinstaller` or `nuitka` after verifying everything works.

If you'd like, I can:

- Add a `package.json` and Vite configuration to turn your `intelligent-npc-demo.tsx` into a buildable frontend.
- Create a simple production-ready static wrapper and a button that starts training or toggles the neural network.

Implemented enhancements

- A Vite + React TypeScript scaffold was added so you can build a production `dist/`.
- A Python bridge is exposed via `pywebview` (`app.py`) with these methods:
	- `save_training_data(json_str)` — save JSON string to `training_data.json`.
	- `load_training_data()` — return saved JSON string.
	- `python_ping()` — returns `'pong'`.
	- `start_training()` — starts a lightweight numpy-based simulated training loop (background thread).
	- `stop_training()` — stops the background training thread.
	- `get_status()` — returns current training status (episode, last/avg reward).

Frontend integration

- `src/python-bridge.ts` wires example buttons and listens for `window.onPythonMessage` events pushed from Python.
- `src/App.tsx` includes a small control panel with Start/Stop/Ping buttons and a live log/status panel.

Run instructions (dev)

```powershell
# frontend
npm install
npm run dev

# python
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Run instructions (production build)

```powershell
npm install
npm run build
python app.py
```

Next steps

- I can replace the simulated numpy training with a real model implemented in Python (NumPy -> JAX/PyTorch) to run the same training logic currently in JS and exchange weights.
- I can add endpoints to stream training logs over a WebSocket-like channel if you prefer lower-latency updates.

PyTorch notes

- The app now uses PyTorch for training. Installing `torch` via `pip` can vary by platform. On Windows with CPU-only support, a simple option is:

```powershell
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

- Alternatively, install from the official instructions at https://pytorch.org/get-started/locally/ and choose the appropriate wheel for your CUDA/CPU setup.

Tell me which direction you prefer and I'll implement it next.
