import React from 'react'
import IntelligentNPC from '../../intelligent-npc-demo'

export default function App() {
  return (
    <div style={{ padding: 20 }}>
      <IntelligentNPC />

      <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
        <button id="btn-save">Save Training Data</button>
        <button id="btn-load">Load Training Data</button>
        <button id="btn-start">Start Python Training</button>
        <button id="btn-stop">Stop Python Training</button>
        <button id="btn-ping">Ping Python</button>
      </div>

      <div style={{ marginTop: 12, background: '#111827', color: '#e5e7eb', padding: 12, borderRadius: 8 }}>
        <h3>Python Bridge</h3>
        <div style={{ display: 'flex', gap: 12 }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 12, color: '#9ca3af' }}>Status</div>
            <pre id="python-status" style={{ background: '#0b1220', padding: 8, borderRadius: 6, minHeight: 60 }}>Not connected</pre>
          </div>
          <div style={{ width: 320 }}>
            <div style={{ fontSize: 12, color: '#9ca3af' }}>Live Log</div>
            <div id="python-log" style={{ background: '#0b1220', padding: 8, height: 120, overflow: 'auto', borderRadius: 6 }} />
          </div>
        </div>
      </div>
    </div>
  )
}
