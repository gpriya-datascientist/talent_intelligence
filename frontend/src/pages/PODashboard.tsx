import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

const API = 'http://localhost:8000'
type Status = 'pending'|'parsing'|'awaiting_sme'|'enriching'|'matching'|'completed'|'failed'

// ── PARTICLE NETWORK (shared across pages) ────────────────────────────────
export function ParticleNetwork() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx    = canvas.getContext('2d')!
    let raf: number
    let mouse = { x: -1000, y: -1000 }
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight }
    resize()
    window.addEventListener('resize', resize)
    window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY })
    const particles = Array.from({ length: 60 }, () => ({
      x: Math.random() * canvas.width, y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.5, pulse: Math.random() * Math.PI * 2,
    }))
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      for (const p of particles) {
        p.x += p.vx; p.y += p.vy; p.pulse += 0.02
        if (p.x < 0 || p.x > canvas.width)  p.vx *= -1
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1
        const dx = mouse.x - p.x; const dy = mouse.y - p.y
        const d = Math.sqrt(dx*dx + dy*dy)
        if (d < 120) { p.x += dx * 0.002; p.y += dy * 0.002 }
        const alpha = 0.4 + Math.sin(p.pulse) * 0.15
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI*2)
        ctx.fillStyle = `rgba(59,130,246,${alpha})`; ctx.fill()
      }
      for (let i = 0; i < particles.length; i++) {
        for (let j = i+1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x; const dy = particles[i].y - particles[j].y
          const d  = Math.sqrt(dx*dx + dy*dy)
          if (d < 120) {
            ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.strokeStyle = `rgba(59,130,246,${(1-d/120)*0.25})`
            ctx.lineWidth = 0.6; ctx.stroke()
          }
        }
        const dx = particles[i].x - mouse.x; const dy = particles[i].y - mouse.y
        const d  = Math.sqrt(dx*dx + dy*dy)
        if (d < 160) {
          ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y)
          ctx.lineTo(mouse.x, mouse.y)
          ctx.strokeStyle = `rgba(99,179,246,${(1-d/160)*0.5})`
          ctx.lineWidth = 0.8; ctx.stroke()
        }
      }
      raf = requestAnimationFrame(draw)
    }
    draw()
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [])
  return <canvas ref={canvasRef} style={{ position: 'fixed', top: 0, left: 0,
    width: '100%', height: '100%', pointerEvents: 'none', zIndex: 0 }} />
}

interface Candidate {
  employee_id: string; rank: number; score: number; matched_skills: string[]
  is_backup?: boolean; full_name?: string; title?: string; seniority_level?: string
  github_stats?: { total_commits: number; top_languages: string[]; active_repos: number }
  availability?: { available_percentage: number; status: string; free_from_date?: string }
  explanation?: string
  capacity_hours?: number
  start_date_ok?: boolean
}
interface WishResult {
  id: string; status: Status; parsed_intent?: string; detected_domains?: string[]
  matched_candidates?: Candidate[]; required_sme_domains?: string[]
  sme_inputs?: Record<string, any>; ambiguities?: { field: string; question: string }[]
  role_split?: Record<string, { hours: number; headcount: number; pct: number }>
  duration_months?: number; total_hours?: number; project_start_date?: string
  additional_requirements?: Array<{
    text: string; intent: string; label: string
    candidates: Candidate[]; added_at: string
  }>
}

const PIPELINE_STEPS = [
  { key: 'parsing',      label: 'Understanding your wish',   icon: '🔍' },
  { key: 'awaiting_sme', label: 'Expert domain review',      icon: '⭐' },
  { key: 'enriching',    label: 'Building requirements',     icon: '⚙️' },
  { key: 'matching',     label: 'Searching employees',       icon: '🔎' },
  { key: 'completed',    label: 'Team assembled',            icon: '✅' },
]
const STATUS_ORDER: Status[] = ['pending','parsing','awaiting_sme','enriching','matching','completed']

function stepStatus(step: string, current: Status) {
  const curIdx  = STATUS_ORDER.indexOf(current)
  const matchIdx = STATUS_ORDER.indexOf(step as Status)
  if (current === 'completed') return 'done'
  if (curIdx > matchIdx) return 'done'
  if (step === current) return 'active'
  return 'pending'
}

function ScoreRing({ score }: { score: number }) {
  const pct   = Math.round(score * 100)
  const color = pct >= 80 ? '#10b981' : pct >= 60 ? '#f59e0b' : '#ef4444'
  return (
    <div style={{ width: 56, height: 56, borderRadius: '50%', flexShrink: 0, position: 'relative',
      background: `conic-gradient(${color} ${pct}%, rgba(255,255,255,0.05) 0)`,
      boxShadow: `0 0 16px ${color}40`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ width: 42, height: 42, borderRadius: '50%', background: 'var(--surface)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color, fontSize: 13, fontWeight: 800 }}>{pct}</div>
    </div>
  )
}

function CandidateCard({ c, index, isBackup }: { c: Candidate; index: number; isBackup?: boolean }) {
  const [flipped, setFlipped] = useState(false)
  const pct       = Math.round((c.availability?.available_percentage || 1) * 100)
  const dot       = pct >= 60 ? '🟢' : pct >= 30 ? '🟡' : '🔴'
  const score     = Math.round(c.score * 100)
  const glowColor = isBackup ? '#fb923c' : '#10b981'
  const tagColor  = isBackup ? '#fb923c' : '#10b981'
  const tagBg     = isBackup ? 'rgba(251,146,60,0.15)' : 'rgba(16,185,129,0.15)'
  const border    = isBackup ? 'rgba(251,146,60,0.45)' : index === 0 ? 'rgba(16,185,129,0.7)' : 'rgba(16,185,129,0.3)'
  const bg        = isBackup ? 'rgba(251,146,60,0.05)' : index === 0 ? 'rgba(16,185,129,0.07)' : 'rgba(16,185,129,0.03)'

  return (
    <div className="fade-up" style={{ animationDelay: `${index * 0.08}s`,
      borderRadius: 16, border: `1px solid ${border}`, background: bg,
      padding: '18px 20px', boxShadow: `0 0 20px ${glowColor}12`,
      transition: 'box-shadow 0.3s' }}>

      {!flipped ? (
        /* ── FRONT ── */
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ width: 58, height: 58, borderRadius: '50%', flexShrink: 0,
            background: `conic-gradient(${glowColor} ${score}%, rgba(255,255,255,0.05) 0)`,
            boxShadow: `0 0 18px ${glowColor}40`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ width: 44, height: 44, borderRadius: '50%', background: 'var(--surface)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: glowColor, fontSize: 13, fontWeight: 800 }}>{score}</div>
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 800, fontSize: 16 }}>{c.full_name || `Candidate #${c.rank}`}</span>
              {index === 0 && !isBackup && (
                <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 20,
                  background: 'rgba(16,185,129,0.2)', color: '#10b981' }}>BEST MATCH</span>
              )}
              {isBackup && (
                <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 20,
                  background: 'rgba(251,146,60,0.2)', color: '#fb923c' }}>BACKUP</span>
              )}
            </div>
            <p style={{ fontSize: 13, color: 'var(--muted)' }}>{c.title} · {c.seniority_level}</p>
            <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
              {c.matched_skills?.slice(0, 4).map(s => (
                <span key={s} style={{ fontSize: 11, padding: '3px 8px', borderRadius: 20,
                  background: tagBg, color: tagColor, fontWeight: 600 }}>🔧 {s}</span>
              ))}
            </div>
          </div>
          <button onClick={() => setFlipped(true)}
            style={{ background: 'none', border: `1px solid ${border}`, borderRadius: 10,
              padding: '8px 12px', cursor: 'pointer', color: 'var(--muted)', fontSize: 11,
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, flexShrink: 0 }}>
            <span style={{ fontSize: 16 }}>↻</span>details
          </button>
        </div>
      ) : (
        /* ── BACK ── */
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 36, height: 36, borderRadius: '50%',
                background: `conic-gradient(${glowColor} ${score}%, rgba(255,255,255,0.05) 0)`,
                display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ width: 26, height: 26, borderRadius: '50%', background: 'var(--surface)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: glowColor, fontSize: 11, fontWeight: 800 }}>{score}</div>
              </div>
              <span style={{ fontWeight: 800, fontSize: 15 }}>{c.full_name}</span>
            </div>
            <button onClick={() => setFlipped(false)}
              style={{ background: 'none', border: `1px solid ${border}`, borderRadius: 10,
                padding: '8px 12px', cursor: 'pointer', color: 'var(--muted)', fontSize: 11,
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
              <span style={{ fontSize: 16 }}>↻</span>back
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {c.github_stats && (
              <div style={{ background: 'rgba(0,0,0,0.25)', borderRadius: 10, padding: '10px 14px' }}>
                <p style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.08em', marginBottom: 6 }}>GITHUB ACTIVITY</p>
                <div style={{ display: 'flex', gap: 16, fontSize: 13 }}>
                  <span>📁 <strong>{c.github_stats.active_repos}</strong> repos</span>
                  <span>⚡ <strong>{c.github_stats.total_commits}</strong> commits</span>
                </div>
                {c.github_stats.top_languages?.length > 0 && (
                  <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
                    {c.github_stats.top_languages.map(l => (
                      <span key={l} style={{ fontSize: 11, padding: '2px 8px', borderRadius: 20,
                        background: 'rgba(255,255,255,0.05)', color: 'var(--muted)', fontFamily: 'monospace' }}>{l}</span>
                    ))}
                  </div>
                )}
              </div>
            )}
            {c.availability && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>
                  {dot} {pct}% available
                  {c.availability.free_from_date && (
                    <span style={{ color: 'var(--muted)', fontWeight: 400 }}>
                      {' '}· free {new Date(c.availability.free_from_date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}
                    </span>
                  )}
                </span>
                {c.capacity_hours !== undefined && (
                  <span style={{ fontSize: 12, color: c.start_date_ok ? '#10b981' : '#f59e0b' }}>
                    {c.start_date_ok ? '✅' : '⚠️'} {Math.round(c.capacity_hours)}hrs capacity
                    {!c.start_date_ok && ' · may miss start'}
                  </span>
                )}
              </div>
            )}
            {c.matched_skills?.length > 0 && (
              <div>
                <p style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.08em', marginBottom: 6 }}>ALL MATCHED SKILLS</p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {c.matched_skills.map(s => (
                    <span key={s} style={{ fontSize: 11, padding: '3px 8px', borderRadius: 20,
                      background: tagBg, color: tagColor, fontWeight: 600 }}>🔧 {s}</span>
                  ))}
                </div>
              </div>
            )}
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ fontSize: 11, padding: '3px 10px', borderRadius: 20,
                background: 'rgba(255,255,255,0.05)', color: 'var(--muted)' }}>Score: {score}%</span>
              {(c.github_stats?.total_commits || 0) > 0 && (
                <span style={{ fontSize: 11, padding: '3px 10px', borderRadius: 20,
                  background: 'rgba(16,185,129,0.1)', color: '#10b981' }}>✓ GitHub Verified</span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function SMEWaitingCard({ result }: { result: WishResult }) {
  const required = result.required_sme_domains || []
  const answered = Object.keys(result.sme_inputs || {})
  return (
    <div className="card fade-up" style={{ borderColor: 'rgba(245,158,11,0.4)', background: 'rgba(245,158,11,0.04)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
        <span style={{ fontSize: 24 }}>⭐</span>
        <div>
          <p style={{ fontWeight: 700, fontSize: 15 }}>Waiting for domain expert review</p>
          <p style={{ fontSize: 13, color: 'var(--muted)' }}>
            {answered.length}/{required.length} expert{required.length !== 1 ? 's' : ''} responded
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
        {required.map(domain => {
          const done = answered.includes(domain)
          return (
            <div key={domain} style={{ display: 'flex', alignItems: 'center', gap: 10,
              padding: '10px 14px', borderRadius: 10, background: 'var(--surface2)',
              border: `1px solid ${done ? 'var(--green)' : 'var(--border)'}` }}>
              <span style={{ fontSize: 16 }}>{done ? '✅' : '⏳'}</span>
              <div style={{ flex: 1 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>{domain}</span>
                <span style={{ fontSize: 12, color: done ? 'var(--green)' : 'var(--muted)', marginLeft: 8 }}>
                  {done ? 'Review submitted' : 'Waiting for expert...'}
                </span>
              </div>
            </div>
          )
        })}
      </div>

      {result.ambiguities && result.ambiguities.length > 0 && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 14 }}>
          <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.08em', marginBottom: 10 }}>
            QUESTIONS SENT TO EXPERTS
          </p>
          {result.ambiguities.map(a => (
            <div key={a.field} style={{ fontSize: 13, color: 'var(--muted)', padding: '4px 0', display: 'flex', gap: 8 }}>
              <span style={{ color: 'var(--accent)' }}>?</span>
              <span>{a.question}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function PODashboard() {
  const navigate = useNavigate()
  const user = JSON.parse(localStorage.getItem('ti_user') || '{}')
  const [wish, setWish]                 = useState('')
  const [startDate, setStartDate]       = useState('')
  const [durationMonths, setDuration]   = useState(3)
  const [totalHours, setTotalHours]     = useState(280)
  const [result, setResult]             = useState<WishResult | null>(null)
  const [loading, setLoading]           = useState(false)
  const [addText, setAddText]           = useState('')
  const [addLoading, setAddLoading]     = useState(false)
  const [showAddBox, setShowAddBox]     = useState(false)
  const logout = () => { localStorage.removeItem('ti_user'); navigate('/login') }

  const submitWish = async () => {
    if (!wish.trim() || !startDate || loading) return
    setLoading(true); setResult(null)
    try {
      const res  = await fetch(`${API}/wishes/`, { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          po_id:              user.id || 'po_demo',
          wish_text:          wish,
          project_start_date: startDate,
          duration_months:    durationMonths,
          total_hours:        totalHours,
        })
      })
      const data = await res.json()
      setResult(data); pollStatus(data.id)
    } catch { setLoading(false) }
  }

  const submitAdditional = async () => {
    if (!addText.trim() || !result?.id || addLoading) return
    setAddLoading(true)
    try {
      const res  = await fetch(`${API}/wishes/${result.id}/add-requirement`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ additional_text: addText })
      })
      const data = await res.json()
      setResult(data)
      setAddText('')
      setShowAddBox(false)
    } catch (e) {
      console.error(e)
    } finally {
      setAddLoading(false)
    }
  }

  const removeAdditional = async (index: number) => {
    if (!result?.id) return
    await fetch(`${API}/wishes/${result.id}/add-requirement/${index}`, { method: 'DELETE' })
    const res  = await fetch(`${API}/wishes/${result.id}`)
    const data = await res.json()
    setResult(data)
  }

  const pollStatus = (wishId: string) => {
    const iv = setInterval(async () => {
      try {
        const res  = await fetch(`${API}/wishes/${wishId}`)
        const data: WishResult = await res.json()
        setResult(data)
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(iv); setLoading(false)
        }
        // Keep polling if awaiting_sme — updates when SME answers
      } catch { clearInterval(iv); setLoading(false) }
    }, 2000)
  }

  const mainCandidates   = result?.matched_candidates?.filter(c => !c.is_backup) || []
  const backupCandidates = result?.matched_candidates?.filter(c => c.is_backup) || []
  const isAwaitingSME    = result?.status === 'awaiting_sme'

  return (
    <div className="page min-h-screen">
      <ParticleNetwork />
      <nav style={{ borderBottom: '1px solid var(--border)', padding: '16px 32px',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    background: 'rgba(8,11,16,0.8)', backdropFilter: 'blur(12px)',
                    position: 'sticky', top: 0, zIndex: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 20 }}>🧠</span>
          <span style={{ fontWeight: 800, letterSpacing: '-0.02em' }}>Talent Intelligence</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ fontSize: 13, color: 'var(--muted)' }}>🎯 {user.name || 'Product Owner'}</span>
          <button className="btn-secondary" onClick={logout} style={{ padding: '8px 16px' }}>Logout</button>
        </div>
      </nav>

      <div style={{ maxWidth: 760, margin: '0 auto', padding: '40px 24px' }}>
        <div className="fade-up" style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: 32, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 8 }}>
            What are you building?
          </h1>
          <p style={{ color: 'var(--muted)', fontSize: 15 }}>
            Describe your project — AI finds the right people from your team.
          </p>
        </div>

        <div className="card fade-up-1" style={{ marginBottom: 24 }}>
          <textarea className="input" rows={4} style={{ resize: 'none', fontSize: 15, lineHeight: 1.6, marginBottom: 16 }}
            placeholder="e.g. We need to build an AI chatbot with RAG for internal HR documents..."
            value={wish} onChange={e => setWish(e.target.value)} />

          {/* Project planning fields */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 16 }}>
            <div>
              <label style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
                START DATE *
              </label>
              <input type="date" className="input" style={{ padding: '10px 12px', fontSize: 13 }}
                value={startDate} onChange={e => setStartDate(e.target.value)}
                min={new Date().toISOString().split('T')[0]} />
            </div>
            <div>
              <label style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
                DURATION *
              </label>
              <select className="input" style={{ padding: '10px 12px', fontSize: 13 }}
                value={durationMonths} onChange={e => setDuration(Number(e.target.value))}>
                {[1,2,3,4,5,6,9,12].map(m => (
                  <option key={m} value={m}>{m} month{m > 1 ? 's' : ''}</option>
                ))}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
                TOTAL HOURS *
              </label>
              <input type="number" className="input" style={{ padding: '10px 12px', fontSize: 13 }}
                value={totalHours} onChange={e => setTotalHours(Number(e.target.value))}
                min={40} max={2000} step={40} />
            </div>
          </div>

          {/* Hours info bar */}
          <div style={{ background: 'var(--surface2)', borderRadius: 8, padding: '10px 14px',
                        fontSize: 12, color: 'var(--muted)', marginBottom: 16,
                        display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            <span>📅 {durationMonths} month{durationMonths > 1 ? 's' : ''}</span>
            <span>⏱ {totalHours} total hours</span>
            <span>👥 ~{Math.max(1, Math.round(totalHours / (durationMonths * 80)))} person{Math.max(1, Math.round(totalHours / (durationMonths * 80))) > 1 ? 's' : ''} needed</span>
            <span>📊 {Math.round(totalHours / durationMonths)} hrs/month avg</span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>{wish.length} chars</span>
            <button className="btn-primary" onClick={submitWish}
              disabled={loading || !wish.trim() || !startDate}
              style={{ width: 'auto', padding: '12px 28px' }}>
              {loading ? '⏳ Processing...' : '🔍 Find Team'}
            </button>
          </div>
        </div>

        {/* Pipeline progress */}
        {result && (
          <div className="card fade-up" style={{ marginBottom: 24 }}>
            <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em', marginBottom: 16 }}>PIPELINE</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {PIPELINE_STEPS.map(step => {
                const s = stepStatus(step.key, result.status)
                // Hide awaiting_sme step if it was skipped
                if (step.key === 'awaiting_sme' && !result.required_sme_domains?.length && s === 'pending') return null
                return (
                  <div key={step.key} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{ width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: s === 'done' ? 'rgba(16,185,129,0.15)' : s === 'active' ? 'rgba(59,130,246,0.15)' : 'var(--surface2)',
                      border: `1px solid ${s === 'done' ? 'var(--green)' : s === 'active' ? 'var(--accent)' : 'var(--border)'}`,
                      fontSize: 13 }}>
                      {s === 'done' ? '✓' : s === 'active' ? '⏳' : step.icon}
                    </div>
                    <span style={{ fontSize: 14,
                      color: s === 'done' ? 'var(--green)' : s === 'active' ? 'var(--text)' : 'var(--muted)',
                      fontWeight: s === 'active' ? 700 : 400 }}>
                      {step.label}
                      {s === 'active' && <span style={{ color: 'var(--accent)' }}> ...</span>}
                    </span>
                  </div>
                )
              })}
            </div>

            {result.parsed_intent && (
              <div style={{ marginTop: 16, padding: '12px 14px', background: 'var(--surface2)',
                            borderRadius: 10, fontSize: 13, color: 'var(--muted)', lineHeight: 1.5 }}>
                <span style={{ color: 'var(--accent)', fontWeight: 700 }}>Understood: </span>
                {result.parsed_intent}
              </div>
            )}
            {result.detected_domains?.length > 0 && (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 10 }}>
                {result.detected_domains.map(d => <span key={d} className="tag tag-amber">{d}</span>)}
              </div>
            )}
            {result.status === 'failed' && (
              <p style={{ color: 'var(--red)', fontSize: 13, marginTop: 12 }}>
                ❌ Pipeline failed. Please check backend logs.
              </p>
            )}
          </div>
        )}

        {/* SME waiting card */}
        {isAwaitingSME && result && (
          <div style={{ marginBottom: 24 }}>
            <SMEWaitingCard result={result} />
          </div>
        )}

        {/* Role split breakdown */}
        {result?.role_split && result.status === 'completed' && (
          <div className="card fade-up" style={{ marginBottom: 24, borderColor: 'rgba(59,130,246,0.3)' }}>
            <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em', marginBottom: 14 }}>
              TEAM COMPOSITION — {result.total_hours}hrs over {result.duration_months} month{result.duration_months !== 1 ? 's' : ''}
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {Object.entries(result.role_split).map(([role, data]) => (
                <div key={role} style={{ background: 'var(--surface2)', borderRadius: 10,
                  padding: '12px 16px', minWidth: 120, border: '1px solid var(--border)' }}>
                  <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent)', textTransform: 'uppercase', marginBottom: 6 }}>
                    {role}
                  </p>
                  <p style={{ fontSize: 18, fontWeight: 800, marginBottom: 2 }}>{data.headcount}</p>
                  <p style={{ fontSize: 11, color: 'var(--muted)' }}>person{data.headcount > 1 ? 's' : ''}</p>
                  <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>{data.hours}hrs · {data.pct}%</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Results */}
        {mainCandidates.length > 0 && (
          <>
            <p className="fade-up" style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)',
                            letterSpacing: '0.1em', marginBottom: 12 }}>
              RECOMMENDED TEAM ({mainCandidates.length})
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 24 }}>
              {mainCandidates.map((c, i) => <CandidateCard key={c.employee_id} c={c} index={i} isBackup={false} />)}
            </div>
          </>
        )}
        {/* Additional requirements sections */}
        {result?.additional_requirements?.map((ar, idx) => (
          <div key={idx} style={{ marginBottom: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
              <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent)', letterSpacing: '0.1em' }}>
                ADDITIONAL: {ar.label.toUpperCase()}
              </p>
              <button onClick={() => removeAdditional(idx)}
                style={{ fontSize: 11, color: 'var(--muted)', background: 'none', border: 'none',
                         cursor: 'pointer', padding: '4px 8px' }}>
                ✕ Remove
              </button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {ar.candidates.filter(c => !c.is_backup).map((c, i) => (
                <CandidateCard key={c.employee_id} c={c} index={i} />
              ))}
            </div>
          </div>
        ))}

        {/* Add requirement box */}
        {result?.status === 'completed' && (result?.additional_requirements?.length || 0) < 2 && (
          <div style={{ marginBottom: 24 }}>
            {!showAddBox ? (
              <button onClick={() => setShowAddBox(true)}
                style={{ width: '100%', padding: '14px', borderRadius: 12,
                         border: '1px dashed var(--border)', background: 'transparent',
                         color: 'var(--muted)', fontSize: 14, cursor: 'pointer',
                         display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                <span style={{ fontSize: 18 }}>+</span>
                Add Additional Requirement
                <span style={{ fontSize: 12, color: 'var(--muted)' }}>
                  ({2 - (result?.additional_requirements?.length || 0)} remaining)
                </span>
              </button>
            ) : (
              <div className="card" style={{ borderColor: 'rgba(59,130,246,0.3)' }}>
                <p style={{ fontSize: 12, fontWeight: 700, color: 'var(--muted)',
                             letterSpacing: '0.08em', marginBottom: 10 }}>
                  ADDITIONAL REQUIREMENT
                </p>
                <textarea className="input" rows={3}
                  style={{ resize: 'none', fontSize: 14, marginBottom: 12 }}
                  placeholder="e.g. We also need a UX designer with Figma experience for the frontend..."
                  value={addText} onChange={e => setAddText(e.target.value)} />
                <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
                  <button onClick={() => { setShowAddBox(false); setAddText('') }}
                    className="btn-secondary" style={{ padding: '10px 20px' }}>
                    Cancel
                  </button>
                  <button onClick={submitAdditional}
                    disabled={addLoading || !addText.trim()}
                    className="btn-primary" style={{ width: 'auto', padding: '10px 20px' }}>
                    {addLoading ? '⏳ Searching...' : '🔍 Find Match'}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {backupCandidates.length > 0 && (
          <>
            <p className="fade-up" style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)',
                            letterSpacing: '0.1em', marginBottom: 12 }}>
              BACKUP CANDIDATES ({backupCandidates.length})
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {backupCandidates.map((c, i) => <CandidateCard key={c.employee_id} c={c} index={i} isBackup={true} />)}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
