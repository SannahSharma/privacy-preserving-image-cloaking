import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8001').replace(/\/$/, '')

const transferRows = [
  ['ResNet-50', 'MobileNetV3', 18, 120, 15.00],
  ['ResNet-50', 'ViT', 8, 120, 6.67],
  ['MobileNetV3', 'ResNet-50', 10, 120, 8.33],
  ['MobileNetV3', 'ViT', 6, 120, 5.00],
  ['ViT', 'ResNet-50', 10, 120, 8.33],
  ['ViT', 'MobileNetV3', 16, 120, 13.33]
]

const robustnessRows = [
  ['ResNet-50 → MobileNetV3', 'None', 15.00],
  ['ResNet-50 → MobileNetV3', 'JPEG Q90', 6.67],
  ['ResNet-50 → MobileNetV3', 'JPEG Q70', 6.67],
  ['ResNet-50 → MobileNetV3', 'Resize 75%', 15.00],
  ['ViT → MobileNetV3', 'None', 13.33],
  ['ViT → MobileNetV3', 'JPEG Q90', 9.17],
  ['ViT → MobileNetV3', 'JPEG Q70', 6.67],
  ['ViT → MobileNetV3', 'Resize 75%', 17.50]
]

const statRows = [
  ['FGSM', 'Default', 4.17, null],
  ['FGSM', 'Stronger', 13.33, 0.011976],
  ['PGD', 'Default', 2.50, null],
  ['PGD', 'Stronger', 8.33, 0.045905]
]

function App() {
  const [status, setStatus] = useState({ online: false, text: 'Checking engine…' })
  const [models, setModels] = useState([])
  const [imageFile, setImageFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [model, setModel] = useState('resnet50')
  const [method, setMethod] = useState('fgsm')
  const [protection, setProtection] = useState('standard')
  const [strength, setStrength] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('studio')
  const [dragActive, setDragActive] = useState(false)
  const fileRef = useRef(null)

  useEffect(() => {
    checkBackend()
    const timer = setInterval(checkBackend, 15000)
    return () => clearInterval(timer)
  }, [])

  async function checkBackend() {
    try {
      const health = await fetch(`${API_BASE}/health`)
      if (!health.ok) throw new Error('offline')
      const data = await health.json()
      setStatus({ online: true, text: `Engine connected — ${data.engine_mode || 'live engine'}` })
      const modelResponse = await fetch(`${API_BASE}/models`)
      if (modelResponse.ok) {
        const modelData = await modelResponse.json()
        setModels(modelData.models || [])
      }
    } catch {
      setStatus({ online: false, text: `Engine unreachable at ${API_BASE}` })
    }
  }

  function handleFile(file) {
    if (!file) return
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
      setError('Please choose a PNG, JPEG, or WebP image.')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('Image must be 10 MB or smaller.')
      return
    }
    setError('')
    setImageFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragActive(false)
    handleFile(event.dataTransfer.files?.[0])
  }

  async function runCloak(event) {
    event.preventDefault()
    if (!imageFile) {
      setError('Upload an image first.')
      return
    }
    setError('')
    setLoading(true)
    setResult(null)
    const body = new FormData()
    body.append('image', imageFile, imageFile.name)
    body.append('strength', String(strength))
    body.append('model_name', model)
    body.append('method', method)
    body.append('protection_mode', protection)

    try {
      const response = await fetch(`${API_BASE}/cloak`, { method: 'POST', body })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.detail || `Request failed with status ${response.status}`)
      setResult(payload)
    } catch (err) {
      setError(err.message || 'Cloaking request failed.')
    } finally {
      setLoading(false)
    }
  }

  const selectedModel = models.find(m => m.name === model)
  const original = result?.original_predictions?.[0]
  const cloaked = result?.cloaked_predictions?.[0]
  const resultImage = result?.cloaked_image_base64 ? `data:image/png;base64,${result.cloaked_image_base64}` : ''
  const predictionChanged = original && cloaked && original.label !== cloaked.label

  return (
    <div className="app-shell">
      <header className="masthead">
        <div className="wordmark">
          <div className="aperture" />
          <div>
            <h1>Cloaking Studio</h1>
            <p>An adversarial perturbation bench for testing how much a classifier's confidence can be disturbed without changing what a photo looks like to a person.</p>
          </div>
        </div>
        <div className={`engine-tag ${status.online ? 'online' : 'offline'}`}>
          <span className="mark" />
          {status.text}
        </div>
      </header>

      <nav className="tabs">
        <button className={activeTab === 'studio' ? 'active' : ''} onClick={() => setActiveTab('studio')}>Studio</button>
        <button className={activeTab === 'research' ? 'active' : ''} onClick={() => setActiveTab('research')}>Research results</button>
        <button className={activeTab === 'architecture' ? 'active' : ''} onClick={() => setActiveTab('architecture')}>System flow</button>
      </nav>

      {activeTab === 'studio' && (
        <main className="content-grid">
          <aside className="panel spec-sheet">
            <p className="kicker">Run configuration</p>
            <h2>Set up a cloaking pass</h2>
            <p className="lede">Pick a target model and an attack, then send the image to the FastAPI + PyTorch engine.</p>

            <div className="field">
              <div className="field-label"><span className="num">1</span><span>Image</span></div>
              <button
                type="button"
                className={`viewfinder${dragActive ? ' drag-active' : ''}`}
                onClick={() => fileRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setDragActive(true) }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
              >
                <span className="vf-br" /><span className="vf-bl" />
                <span className="file-name">{imageFile ? imageFile.name : 'Choose or drop an image'}</span>
                <small>PNG · JPEG · WEBP, up to 10 MB</small>
              </button>
              <input ref={fileRef} hidden type="file" accept="image/png,image/jpeg,image/webp" onChange={e => handleFile(e.target.files?.[0])} />
            </div>

            <div className="field">
              <div className="field-label"><span className="num">2</span><span>Model architecture</span></div>
              <select value={model} onChange={e => setModel(e.target.value)}>
                {models.length === 0 && <option value="resnet50">ResNet-50</option>}
                {models.map(m => <option key={m.name} value={m.name} disabled={!m.implemented}>{m.display_name}{m.implemented ? '' : ' — unavailable'}</option>)}
              </select>
              <span className="field-help">{selectedModel?.implemented ? 'Implemented in the current ML engine.' : 'Waiting for backend model discovery.'}</span>
            </div>

            <div className="field">
              <div className="field-label"><span className="num">3</span><span>Perturbation method</span></div>
              <div className="pair-choice">
                <button type="button" className={method === 'fgsm' ? 'selected' : ''} onClick={() => setMethod('fgsm')}><b>FGSM</b><span>One step, fast</span></button>
                <button type="button" className={method === 'pgd' ? 'selected' : ''} onClick={() => setMethod('pgd')}><b>PGD</b><span>Iterative, stronger</span></button>
              </div>
            </div>

            <div className="field">
              <div className="field-label"><span className="num">4</span><span>Protection strength</span></div>
              <div className="strength-row"><span className="field-help" style={{ marginTop: 0 }}>Scales the perturbation within the budget below</span><span className="val">{strength.toFixed(2)}</span></div>
              <input className="range" type="range" min="0" max="1" step="0.01" value={strength} onChange={e => setStrength(Number(e.target.value))} />
              <div className="range-scale"><span>0.00</span><span>1.00</span></div>
            </div>

            <div className="field">
              <div className="field-label"><span className="num">5</span><span>Protection mode</span></div>
              <div className="pair-choice">
                <button type="button" className={protection === 'standard' ? 'selected' : ''} onClick={() => setProtection('standard')}><b>Standard</b><span>ε max 0.06</span></button>
                <button type="button" className={protection === 'strong' ? 'selected strong' : ''} onClick={() => setProtection('strong')}><b>Strong</b><span>ε max 0.125</span></button>
              </div>
            </div>

            <button className="run-btn" disabled={loading || !imageFile || !status.online} onClick={runCloak}>
              {loading ? <><span className="spinner" />Processing {method.toUpperCase()}</> : 'Apply cloak'}
            </button>

            {error && <div className="error-box">{error}</div>}

            <div className="pipeline">
              <span>Image</span><i>/</i><span>{model}</span><i>/</i><span>{method.toUpperCase()}</span><i>/</i><span>cloaked output</span>
            </div>
          </aside>

          <section className="workspace">
            <div className="readouts">
              <Reading label="SSIM" value={result ? result.ssim_score?.toFixed(4) : '—'} tone="good" />
              <Reading label="Confidence drop" value={result ? `${(result.confidence_drop * 100).toFixed(2)}%` : '—'} tone="info" />
              <Reading label="Epsilon" value={result ? result.epsilon?.toFixed(3) : '—'} tone="flag" />
              <Reading label="Runtime" value={result ? `${result.processing_time_ms?.toFixed(0)} ms` : '—'} tone="neutral" />
            </div>

            <div className="contact-sheet">
              <Frame title="Original" src={previewUrl} prediction={original} placeholder="Upload an image to begin" />
              <div className="sprocket" />
              <Frame title="Cloaked output" src={resultImage} prediction={cloaked} placeholder={loading ? 'Running the attack…' : 'The cloaked result appears here'} changed={predictionChanged} />
            </div>

            <ActivationMap points={result?.activation_map || []} />

            {result && (
              <div className="verdict">
                <div>
                  <span className="tag">Engine result</span>
                  <strong>{predictionChanged ? 'The prediction label changed after cloaking' : 'The label held; confidence shifted instead'}</strong>
                  <p>{result.model_used} · {result.method_used?.toUpperCase()} · {result.protection_mode} · strength {Number(result.strength).toFixed(2)}</p>
                </div>
                <a className="download" href={resultImage} download="cloaked_image.png">Download PNG</a>
              </div>
            )}
          </section>
        </main>
      )}

      {activeTab === 'research' && <ResearchResults />}
      {activeTab === 'architecture' && <Architecture />}
    </div>
  )
}

function Reading({ label, value, tone }) {
  return <div className={`reading tone-${tone}`}><span>{label}</span><strong>{value}</strong></div>
}

function Frame({ title, src, prediction, placeholder, changed }) {
  return <div className="frame">
    <div className="frame-head">
      <h3>{title}</h3>
      {prediction && <span className={`conf ${changed ? 'changed' : ''}`}>{(prediction.confidence * 100).toFixed(1)}% conf.</span>}
    </div>
    <div className="stage">
      <span className="vf-br" /><span className="vf-bl" />
      {src ? <img src={src} alt={title} /> : <div className="empty"><span className="ring" />{placeholder}</div>}
    </div>
    <div className="readout-line">
      <div><small>Top prediction</small><strong>{prediction?.label || '—'}</strong></div>
      {prediction && <em>{changed ? 'changed' : 'after attack'}</em>}
    </div>
  </div>
}

function ActivationMap({ points }) {
  const cells = Array.from({ length: 49 }, (_, i) => points[i]?.intensity ?? 0)
  return <div className="panel activation">
    <p className="kicker">Model interpretability</p>
    <h3>7 × 7 activation map</h3>
    <p>The backend returns 49 normalized activation points from the selected model's feature representation.</p>
    <div className="heat-grid">
      {cells.map((v, i) => <div key={i} className="heat-cell" style={{ opacity: points.length ? Math.max(.12, v) : .08 }} title={`Activation ${v.toFixed?.(3) || '0.000'}`} />)}
    </div>
  </div>
}

function ResearchResults() {
  const maxTransfer = Math.max(...transferRows.map(r => r[4]))
  return <main className="page">
    <div className="page-head">
      <div>
        <p className="kicker">Experimental evaluation</p>
        <h1>Research results</h1>
        <p>Results generated from the current project implementation and stored in <code>results/</code>.</p>
      </div>
      <div className="tally"><strong>720</strong><span>transfer experiments</span></div>
    </div>
    <div className="stat-row">
      <StatCard value="2,880" label="Robustness experiments" />
      <StatCard value="120" label="Face probes" />
      <StatCard value="119/120" label="Baseline face recognition" />
      <StatCard value="0%" label="Identity-change rate" />
    </div>
    <section className="research-grid">
      <div className="data-card">
        <p className="kicker">Transferability</p>
        <h2>Cross-model prediction changes</h2>
        <table><thead><tr><th>Source</th><th>Target</th><th>Changes</th><th>Rate</th></tr></thead>
          <tbody>{transferRows.map(r => <tr key={r.join('-')}><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}/{r[3]}</td><td><div className="bar-cell"><span className="bar" style={{ width: `${(r[4] / maxTransfer) * 100}%` }} /><b>{r[4].toFixed(2)}%</b></div></td></tr>)}</tbody>
        </table>
        <p className="table-note">70 of 720 source-target experiments showed prediction changes (9.72%).</p>
      </div>
      <div className="data-card">
        <p className="kicker">Robustness</p>
        <h2>Post-processing effects</h2>
        <table><thead><tr><th>Pair</th><th>Transform</th><th>Rate</th></tr></thead>
          <tbody>{robustnessRows.map(r => <tr key={r.join('-')}><td>{r[0]}</td><td>{r[1]}</td><td><b>{r[2].toFixed(2)}%</b></td></tr>)}</tbody>
        </table>
        <p className="table-note">JPEG generally reduced prediction-change rates; resizing produced mixed effects.</p>
      </div>
    </section>
    <section className="research-grid">
      <div className="data-card">
        <p className="kicker">Face recognition · LFW</p>
        <h2>FaceNet evaluation</h2>
        <div className="big-stat">0<span>%</span></div>
        <p className="lead">Identity-change rate after cloaking.</p>
        <div className="data-row"><span>Baseline</span><strong>119/120 · 99.17%</strong></div>
        <div className="data-row"><span>Identity changes</span><strong>0/119</strong></div>
        <div className="data-row"><span>Avg. similarity drop</span><strong>0.006068</strong></div>
        <div className="caveat">Classification-targeted cloaking did not change FaceNet identity in this experiment. That's a limitation, not a successful face-privacy result.</div>
      </div>
      <div className="data-card">
        <p className="kicker">Statistical analysis</p>
        <h2>Default vs. stronger perturbation</h2>
        <table><thead><tr><th>Method</th><th>Setting</th><th>Rate</th><th>p-value</th></tr></thead>
          <tbody>{statRows.map(r => <tr key={r.join('-')}><td>{r[0]}</td><td>{r[1]}</td><td>{r[2].toFixed(2)}%</td><td>{r[3] == null ? '—' : r[3]}</td></tr>)}</tbody>
        </table>
        <p className="table-note">FGSM p = 0.011976 and PGD p = 0.045905 for the tested comparisons.</p>
      </div>
    </section>
  </main>
}

function StatCard({ value, label }) {
  return <div className="stat-card"><strong>{value}</strong><span>{label}</span></div>
}

function Architecture() {
  return <main className="page">
    <div className="page-head">
      <div>
        <p className="kicker">System architecture</p>
        <h1>From upload to cloaked image</h1>
        <p>The live frontend talks to a FastAPI backend, which calls the PyTorch adversarial engine.</p>
      </div>
    </div>
    <div className="flow-rail">
      <Flow title="1 · Frontend" text="React + Vite" detail="Upload, model selection, attack selection, strength and result visualization" />
      <div className="flow-link">/</div>
      <Flow title="2 · Backend API" text="FastAPI" detail="/health · /models · /cloak" />
      <div className="flow-link">/</div>
      <Flow title="3 · ML engine" text="PyTorch + Torchvision" detail="ResNet-50 · MobileNetV3 · ViT · FGSM · PGD" />
      <div className="flow-link">/</div>
      <Flow title="4 · Evaluation" text="Metrics + visual feedback" detail="Prediction, confidence drop, epsilon, SSIM, activation map" />
    </div>
    <div className="glossary">
      <Concept title="FGSM" text="A single gradient-based perturbation step." />
      <Concept title="PGD" text="Iterative gradient steps constrained by the perturbation budget." />
      <Concept title="SSIM" text="Measures structural similarity between the original and cloaked images." />
      <Concept title="Activation map" text="49 normalized points representing the model's 7×7 spatial activation grid." />
    </div>
  </main>
}

function Flow({ title, text, detail }) {
  return <div className="flow-card"><p className="kicker">{title}</p><h2>{text}</h2><p>{detail}</p></div>
}
function Concept({ title, text }) {
  return <div><b>{title}</b><p>{text}</p></div>
}

createRoot(document.getElementById('root')).render(<App />)
