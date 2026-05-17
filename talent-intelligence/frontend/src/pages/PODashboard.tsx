import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

const API = 'http://localhost:8000'

type Status = 'pending'|'parsing'|'awaiting_sme'|'enriching'|'matching'|'completed'|'failed'

interface Candidate {
  employee_id: string
  rank: number
  score: number
  matched_skills: string[]
  is_backup?: boolean
  full_name?: string
  title?: string
  seniority_level?: string
  github_stats?: { total_commits: number; top_languages: string[]; active_repos: number }
  availability?: { available_percentage: number; status: string; free_from_date?: string }
  explanation?: string
}

interface WishResult {
  id: string
  status: Status
  parsed_intent?: string
  detected_domains?: string[]
  matched_candidates?: Candidate[]
}

const PIPELINE_STEPS = [
  { key: 'parsing',      label: 'Understanding your wish',        icon: '🔍' },
  { key: 'enriching',    label: 'Building requirements',          icon: '⚙️' },
  { key: 'matching',     label: 'Searching employees',            icon: '🔎' },
  { key: 'completed',    label: 'Team assembled',                 icon: '✅' },
]

const STATUS_ORDER: Status[] = ['pending','parsing','awaiting_sme','enriching','matching','completed']

function stepStatus(step: string, current: Status) {
  const stepIdx = PIPELINE_STEPS.findIndex(s => s.key === step)
  const curIdx  = STATUS_ORDER.indexOf(current)
  const matchIdx = STATUS_ORDER.indexOf(step as Status)
  if (current === 'completed') return 'done'
  if (curIdx > matchIdx) return 'done'
  if (step === current) return 'active'
  return 'pending'
}

function AvailabilityBadge({ avail }: { avail?: Candidate['availability'] }) {
  if (!avail) return null
  const pct = Math.round((avail.available_percentage || 1) * 100)
  const color = pct >= 60 ? 'var(--green)' : pct >= 30 ? 'var(--amber)' : 'var(--red)'
  const dot   = pct >= 60 ? '🟢' : pct >= 30 ? '🟡' : '🔴'
  return (
    <span style={{ color, fontSize: 13, fontWeight: 600 }}>
      {dot} {pct}% available
      {avail.free_from_date && (
        <span style={{ color: 'var(--muted)', fontWeight: 400 }}>
          {' '}· free {new Date(avail.free_from_date).toLocaleDateString('en-GB', { day:'numeric', month:'short' })}
        </span>
      )}
    </span>
  )
}

function ScoreRing({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color = pct >= 80 ? '#10b981' : pct >= 60 ? '#f59e0b' : '#ef4444'
  return (
    <div className="score-ring" style={{
      background: `conic-gradient(${color} ${pct}%, rgba(255,255,255,0.05) 0)`,
      boxShadow: `0 0 16px ${color}40`
    }}>
      <div style={{
        width: 48, height: 48, borderRadius: '50%',
        background: 'var(--surface)', display: 'flex',
        alignItems: 'center', justifyContent: 'center',
        color, fontSize: 14, fontWeight: 800
      }}>
        {pct}
      </div>
    </div>
  )
}

function CandidateCard({ c, index }: { c: Candidate, index: number }) {
  const [open, setOpen] = useState(index < 3)
  return (
    <div className="card fade-up" style={{ animationDelay: `${index * 0.08}s`,
      borderColor: index === 0 ? 'rgba(59,130,246,0.4)' : 'var(--border)',
      background: index === 0 ? 'rgba(59,130,246,0.04)' : 'var(--surface)'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, cursor: 'pointer' }}
           onClick={() => setOpen(o => !o)}>
        <ScoreRing score={c.score} />
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontWeight: 800, fontSize: 16 }}>
              {c.full_name || `Candidate #${c.rank}`}
            </span>
            {index === 0 && (
              <span className="tag tag-blue" style={{ fontSize: 10 }}>BEST MATCH</span>
            )}
            {c.is_backup && (
              <span className="tag tag-gray" style={{ fontSize: 10 }}>BACKUP</span>
            )}
          </div>
          <div style={{ color: 'var(--muted)', fontSize: 13 }}>
            {c.title || 'Employee'} · {c.seniority_level || ''}
          </div>
        </div>
        <div style={{ color: 'var(--muted)', fontSize: 18 }}>{open ? '▲' : '▼'}</div>
      </div>

      {open && (
        <div style={{ marginTop: 20, display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Skills */}
          {c.matched_skills?.length > 0 && (
            <div>
              <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)',
                          letterSpacing: '0.08em', marginBottom: 8 }}>MATCHED SKILLS</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {c.matched_skills.map(s => (
                  <span key={s} className="tag tag-blue">🔧 {s}</span>
                ))}
              </div>
            </div>
          )}

          {/* GitHub */}
          {c.github_stats && (
            <div style={{ background: 'var(--surface2)', borderRadius: 10, padding: '12px 14px' }}>
              <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)',
                          letterSpacing: '0.08em', marginBottom: 6 }}>GITHUB ACTIVITY</p>
              <div style={{ display: 'flex', gap: 20, fontSize: 13 }}>
                <span>📁 <strong>{c.github_stats.active_repos}</strong> active repos</span>
                <span>⚡ <strong>{c.github_stats.total_commits}</strong> commits</span>
              </div>
              {c.github_stats.top_languages?.length > 0 && (
                <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
                  {c.github_stats.top_languages.map(l => (
                    <span key={l} className="tag tag-gray mono">{l}</span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Availability */}
          <AvailabilityBadge avail={c.availability} />

          {/* Explanation */}
          {c.explanation && (
            <div style={{ borderLeft: '3px solid var(--accent)', paddingLeft: 12,
                          color: 'var(--muted)', fontSize: 13, lineHeight: 1.6 }}>
              💬 {c.explanation}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function PODashboard() {
  const navigate = useNavigate()
  const user = JSON.parse(localStorage.getItem('ti_user') || '{}')
  const [wish, setWish] = useState('')
  const [result, setResult] = useState<WishResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [charCount, setCharCount] = useState(0)

  const logout = () => { localStorage.removeItem('ti_user'); navigate('/login') }

  const submitWish = async () => {
    if (!wish.trim() || loading) return
    setLoading(true); setResult(null)
    try {
      const res = await fetch(`${API}/wishes/`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ po_id: user.id || 'po_demo', wish_text: wish }),
      })
      const data = await res.json()
      setResult(data)
      pollStatus(data.id)
    } catch { setLoading(false) }
  }

  const pollStatus = (wishId: string) => {
    const iv = setInterval(async () => {
      try {
        const res = await fetch(`${API}/wishes/${wishId}`)
        const data: WishResult = await res.json()
        setResult(data)
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(iv); setLoading(false)
        }
      } catch { clearInterval(iv); setLoading(false) }
    }, 2000)
  }

  const mainCandidates = result?.matched_candidates?.filter(c => !c.is_backup) || []
  const backupCandidates = result?.matched_candidates?.filter(c => c.is_backup) || []

  return (
    <div className="page min-h-screen">
      {/* Nav */}
      <nav style={{ borderBottom: '1px solid var(--border)', padding: '16px 32px',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    background: 'rgba(8,11,16,0.8)', backdropFilter: 'blur(12px)',
                    position: 'sticky', top: 0, zIndex: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 20 }}>🧠</span>
          <span style={{ fontWeight: 800, letterSpacing: '-0.02em' }}>Talent Intelligence</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ fontSize: 13, color: 'var(--muted)' }}>
            🎯 {user.name || 'Product Owner'}
          </span>
          <button className="btn-secondary" onClick={logout} style={{ padding: '8px 16px' }}>
            Logout
          </button>
        </div>
      </nav>

      <div style={{ maxWidth: 760, margin: '0 auto', padding: '40px 24px' }}>
        {/* Hero */}
        <div className="fade-up" style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: 32, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 8 }}>
            What are you building?
          </h1>
          <p style={{ color: 'var(--muted)', fontSize: 15 }}>
            Describe your project — AI finds the right people from your team.
          </p>
        </div>

        {/* Input */}
        <div className="card fade-up-1" style={{ marginBottom: 24 }}>
          <textarea
            className="input" rows={4}
            style={{ resize: 'none', fontSize: 15, lineHeight: 1.6, marginBottom: 12 }}
            placeholder="e.g. We need software for tuning speaker frequency response on embedded devices with a calibration UI..."
            value={wish}
            onChange={e => { setWish(e.target.value); setCharCount(e.target.value.length) }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>
              {charCount} chars
            </span>
            <button className="btn-primary" onClick={submitWish} disabled={loading || !wish.trim()}
              style={{ width: 'auto', padding: '12px 28px' }}>
              {loading ? '⏳ Processing...' : '🔍 Find Team'}
            </button>
          </div>
        </div>

        {/* Pipeline */}
        {result && (
          <div className="card fade-up" style={{ marginBottom: 24 }}>
            <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)',
                        letterSpacing: '0.1em', marginBottom: 16 }}>PIPELINE</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {PIPELINE_STEPS.map(step => {
                const s = stepStatus(step.key, result.status)
                return (
                  <div key={step.key} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{
                      width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: s === 'done' ? 'rgba(16,185,129,0.15)' :
                                  s === 'active' ? 'rgba(59,130,246,0.15)' : 'var(--surface2)',
                      border: `1px solid ${s === 'done' ? 'var(--green)' : s === 'active' ? 'var(--accent)' : 'var(--border)'}`,
                      fontSize: 13,
                    }}>
                      {s === 'done' ? '✓' : s === 'active' ? '⏳' : step.icon}
                    </div>
                    <span style={{
                      fontSize: 14,
                      color: s === 'done' ? 'var(--green)' : s === 'active' ? 'var(--text)' : 'var(--muted)',
                      fontWeight: s === 'active' ? 700 : 400,
                    }}>
                      {step.label}
                      {s === 'active' && <span style={{ color: 'var(--accent)' }}> ...</span>}
                    </span>
                  </div>
                )
              })}
            </div>
            {result.parsed_intent && (
              <div style={{ marginTop: 16, padding: '12px 14px',
                            background: 'var(--surface2)', borderRadius: 10,
                            fontSize: 13, color: 'var(--muted)', lineHeight: 1.5 }}>
                <span style={{ color: 'var(--accent)', fontWeight: 700 }}>Understood: </span>
                {result.parsed_intent}
              </div>
            )}
            {result.detected_domains?.length > 0 && (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 10 }}>
                {result.detected_domains.map(d => (
                  <span key={d} className="tag tag-amber">{d}</span>
                ))}
              </div>
            )}
            {result.status === 'failed' && (
              <p style={{ color: 'var(--red)', fontSize: 13, marginTop: 12 }}>
                Pipeline failed. Check your OpenAI API key in .env
              </p>
            )}
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
              {mainCandidates.map((c, i) => <CandidateCard key={c.employee_id} c={c} index={i} />)}
            </div>
          </>
        )}

        {backupCandidates.length > 0 && (
          <>
            <p className="fade-up" style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)',
                            letterSpacing: '0.1em', marginBottom: 12 }}>
              BACKUP CANDIDATES ({backupCandidates.length})
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {backupCandidates.map((c, i) => <CandidateCard key={c.employee_id} c={c} index={i + mainCandidates.length} />)}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
