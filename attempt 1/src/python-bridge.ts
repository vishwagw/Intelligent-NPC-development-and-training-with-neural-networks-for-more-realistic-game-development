// Frontend bridge for pywebview API
// This file exposes a small helper and hooks up example buttons to the Python API.

declare global {
  interface Window {
    pywebview?: any;
  }
}

function onReady() {
  const btnSave = document.getElementById('btn-save')
  const btnLoad = document.getElementById('btn-load')
  const btnStart = document.getElementById('btn-start')
  const btnStop = document.getElementById('btn-stop')
  const btnPing = document.getElementById('btn-ping')
  const statusEl = document.getElementById('python-status')
  const logEl = document.getElementById('python-log')

  if (btnSave) {
    btnSave.addEventListener('click', async () => {
      try {
        const res = await window.pywebview.api.save_model()
        console.log('Saved model:', res)
        alert('Saved model: ' + (res.path || res.error))
      } catch (e) {
        console.error('save error', e)
        alert('Failed to save model. Are you running in the desktop app?')
      }
    })
  }

  if (btnLoad) {
    btnLoad.addEventListener('click', async () => {
      try {
        const res = await window.pywebview.api.load_model()
        console.log('Loaded model:', res)
        alert('Loaded model: ' + (res.path || res.error))
      } catch (e) {
        console.error('load error', e)
        alert('Failed to load model. Are you running in the desktop app?')
      }
    })
  }

  if (btnStart) {
    btnStart.addEventListener('click', async () => {
      try {
        const res = await window.pywebview.api.start_training()
        console.log('start_training', res)
        appendLog('Started Python training')
        refreshStatus()
      } catch (e) {
        console.error(e)
        alert('Failed to start training')
      }
    })
  }

  if (btnStop) {
    btnStop.addEventListener('click', async () => {
      try {
        const res = await window.pywebview.api.stop_training()
        console.log('stop_training', res)
        appendLog('Stopped Python training')
        refreshStatus()
      } catch (e) {
        console.error(e)
        alert('Failed to stop training')
      }
    })
  }

  if (btnPing) {
    btnPing.addEventListener('click', async () => {
      try {
        const res = await window.pywebview.api.python_ping()
        appendLog('Ping -> ' + res)
      } catch (e) {
        appendLog('Ping failed')
      }
    })
  }

  function appendLog(msg: string) {
    if (logEl) {
      const d = document.createElement('div')
      d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`
      logEl.appendChild(d)
      logEl.scrollTop = logEl.scrollHeight
    }
  }

  async function refreshStatus() {
    if (!statusEl || !window.pywebview) return
    try {
      const s = await window.pywebview.api.get_status()
      statusEl.textContent = JSON.stringify(s, null, 2)
    } catch (e) {
      statusEl.textContent = 'Error fetching status'
    }
  }

  // Handler invoked by Python via evaluate_js -> window.onPythonMessage
  // Example payloads: {type: 'training_update', episode:..., last_reward:..., avg_reward:...}
  ;(window as any).onPythonMessage = (payload: any) => {
    try {
      if (payload.type === 'training_update') {
        appendLog(`Episode ${payload.episode} | last ${payload.last_reward.toFixed(3)} | avg ${payload.avg_reward.toFixed(3)}`)
        refreshStatus()
      } else if (payload.type === 'training_stopped') {
        appendLog(`Training stopped at episode ${payload.episode}`)
        refreshStatus()
      } else {
        appendLog('Python: ' + JSON.stringify(payload))
      }
    } catch (e) {
      appendLog('Malformed payload from Python')
    }
  }

  // initial status
  refreshStatus()
}

if (window.pywebview) {
  onReady()
} else {
  // If running in browser, wait for pywebview to be ready (pywebview injects an event)
  window.addEventListener('pywebviewready', onReady)
}
