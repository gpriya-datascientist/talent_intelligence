import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { updateAvailability, getPendingSMEWishes, submitSMEInput, triggerSkillExtraction } from '../api'
import { ParticleNetwork } from './PODashboard'

const API = 'http://localhost:8000'

interface Skill { name: string; proficiency: string; is_hands_on: boolean; skill_type: string; evidence?: any }
interface EmployeeProfile {
  id: string; full_name: string; title: string; seniority_level: string
  github_username?: string; github_stats?: any; resume_text?: string
  is_sme?: boolean; sme_domains?: string[]
  skills?: Skill[]
  availability?: { available_percentage: number; status: string; free_from_date?: string; is_soft_open: boolean; soft_open_note?: string; availability_score: number }
}
interface SMEWish {
  id: string; raw_text: string; parsed_intent?: string
  detected_domains?: string[]; ambiguities?: { field: string; question: string }[]
  required_sme_domains?: string[]; sme_inputs?: Record<string, any>
}
interface ResumeVersion { filename: string; uploaded_at: string; skills_count: number; label: string }

const STATUS_OPTS = [
  { value: 'available',           label: 'Available',           color: 'var(--green)'  },
  { value: 'partially_available', label: 'Partially Available', color: 'var(--amber)'  },
  { value: 'busy',                label: 'Busy',                color: 'var(--red)'    },
  { value: 'on_leave',            label: 'On Leave',            color: 'var(--muted)'  },
  { value: 'soft_open',           label: 'Busy but open',       color: 'var(--accent)' },
]

function SkillTag({ skill }: { skill: Skill }) {
  const [tip, setTip] = useState(false)
  const ev = skill.evidence
  const hasTip = ev && (ev.company || ev.project || ev.github_repo)
  return (
    <span className={`tag ${skill.is_hands_on ? 'tag-blue' : 'tag-gray'}`}
      style={{ position: 'relative', cursor: hasTip ? 'pointer' : 'default' }}
      onClick={() => hasTip && setTip(t => !t)}>
      {skill.skill_type === 'tool' ? '🔨' : skill.is_hands_on ? '🔧' : '📖'} {skill.name}
      <span style={{ opacity: 0.6, marginLeft: 2 }}>· {skill.proficiency}</span>
      {ev?.github_confirmed && <span style={{ marginLeft: 4 }}>✓</span>}
      {tip && hasTip && (
        <div style={{
          position: 'absolute', bottom: '130%', left: 0, zIndex: 99,
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: 10, padding: '10px 12px', width: 220,
          fontSize: 12, lineHeight: 1.6, color: 'var(--text)',
          boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
        }}>
          {ev.company && <div><strong>Company:</strong> {ev.company}</div>}
          {ev.project && <div><strong>Project:</strong> {ev.project}</div>}
          {ev.github_repo && <div><strong>GitHub:</strong> {ev.github_repo} ({ev.github_commits} commits)</div>}
          {ev.used_with?.length > 0 && <div><strong>Used with:</strong> {ev.used_with.join(', ')}</div>}
        </div>
      )}
    </span>
  )
}

// ── SME CONSULTATION PANEL ────────────────────────────────────────────────
function SMEConsultationPanel({ profile }: { profile: EmployeeProfile }) {
  const user = JSON.parse(localStorage.getItem('ti_user') || '{}')
  const [wishes, setWishes] = useState<SMEWish[]>([])
  const [loading, setLoading] = useState(true)
  const [activeWish, setActiveWish] = useState<string | null>(null)
  const [answers, setAnswers] = useState<Record<string, Record<string, string>>>({})
  const [submitting, setSubmitting] = useState<string | null>(null)
  const [submitted, setSubmitted] = useState<string[]>([])

  const domain = profile.sme_domains?.[0] || ''

  useEffect(() => {
    if (!domain) return
    getPendingSMEWishes(user.id, domain)
      .then(setWishes).finally(() => setLoading(false))
  }, [domain])

  const setAnswer = (wishId: string, field: string, value: string) => {
    setAnswers(prev => ({ ...prev, [wishId]: { ...(prev[wishId] || {}), [field]: value } }))
  }

  const handleSubmit = async (wish: SMEWish) => {
    setSubmitting(wish.id)
    try {
      await submitSMEInput(wish.id, user.id, domain, answers[wish.id] || {})
      setSubmitted(s => [...s, wish.id])
      setWishes(w => w.filter(x => x.id !== wish.id))
    } catch { }
    setSubmitting(null)
  }

  if (loading) return (
    <div className="card" style={{ opacity: 0.6, fontSize: 13 }}>Loading consultations...</div>
  )

  if (wishes.length === 0 && submitted.length === 0) return (
    <div className="card" style={{ textAlign: 'center', padding: '28px 20px' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>✅</div>
      <p style={{ fontWeight: 600, marginBottom: 4 }}>No pending consultations</p>
      <p style={{ fontSize: 13, color: 'var(--muted)' }}>
        You'll be notified when a new project needs your {domain} expertise
      </p>
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {submitted.map(id => (
        <div key={id} className="card" style={{ borderColor: 'var(--green)', background: 'rgba(16,185,129,0.05)' }}>
          <span style={{ fontSize: 13, color: 'var(--green)' }}>✓ Review submitted — pipeline resumed</span>
        </div>
      ))}

      {wishes.map(wish => {
        const isOpen = activeWish === wish.id
        const myAnswers = answers[wish.id] || {}
        const ambiguities = wish.ambiguities || []
        const allAnswered = ambiguities.every(a => myAnswers[a.field]?.trim())

        return (
          <div key={wish.id} className="card" style={{ borderColor: 'rgba(245,158,11,0.3)' }}>

            {/* Header */}
            <div style={{ cursor: 'pointer' }} onClick={() => setActiveWish(isOpen ? null : wish.id)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--amber)',
                                   letterSpacing: '0.08em' }}>NEEDS YOUR REVIEW</span>
                    {wish.required_sme_domains?.map(d => (
                      <span key={d} className="tag tag-amber" style={{ fontSize: 10 }}>{d}</span>
                    ))}
                  </div>
                  <p style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
                    {wish.parsed_intent || wish.raw_text}
                  </p>
                  <p style={{ fontSize: 12, color: 'var(--muted)' }}>
                    {ambiguities.length} question{ambiguities.length !== 1 ? 's' : ''} need your answer
                  </p>
                </div>
                <span style={{ color: 'var(--muted)', fontSize: 16, marginLeft: 12 }}>
                  {isOpen ? '▲' : '▼'}
                </span>
              </div>
            </div>

            {isOpen && (
              <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>

                {/* PO wish */}
                <div style={{ background: 'var(--surface2)', borderRadius: 10, padding: '12px 14px' }}>
                  <p style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted)',
                               letterSpacing: '0.08em', marginBottom: 6 }}>ORIGINAL WISH</p>
                  <p style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.6, fontStyle: 'italic' }}>
                    "{wish.raw_text}"
                  </p>
                </div>

                {/* System skill guess */}
                {wish.detected_domains && (
                  <div>
                    <p style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted)',
                                 letterSpacing: '0.08em', marginBottom: 8 }}>DETECTED DOMAINS</p>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {wish.detected_domains.map(d => (
                        <span key={d} className="tag tag-blue">{d}</span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Ambiguity Q&A — the professional form */}
                {ambiguities.length > 0 && (
                  <div>
                    <p style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted)',
                                 letterSpacing: '0.08em', marginBottom: 12 }}>
                      YOUR EXPERT INPUT
                    </p>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                      {ambiguities.map((amb, i) => (
                        <div key={amb.field}>
                          <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
                            {i + 1}. {amb.question}
                          </p>

                          {/* Smart input type based on field name */}
                          {amb.field === 'realtime' || amb.field === 'real_time' ? (
                            <div style={{ display: 'flex', gap: 8 }}>
                              {['Real-time', 'Offline', 'Both'].map(opt => (
                                <button key={opt}
                                  onClick={() => setAnswer(wish.id, amb.field, opt)}
                                  style={{
                                    padding: '8px 16px', borderRadius: 20, border: '1px solid',
                                    borderColor: myAnswers[amb.field] === opt ? 'var(--accent)' : 'var(--border)',
                                    background: myAnswers[amb.field] === opt ? 'rgba(59,130,246,0.15)' : 'var(--surface2)',
                                    color: myAnswers[amb.field] === opt ? '#93c5fd' : 'var(--muted)',
                                    fontFamily: 'Syne, sans-serif', fontWeight: 600, fontSize: 12,
                                    cursor: 'pointer',
                                  }}>
                                  {opt}
                                </button>
                              ))}
                            </div>
                          ) : amb.field === 'platform' ? (
                            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                              {['STM32', 'Raspberry Pi', 'Custom FPGA', 'Arduino', 'ESP32', 'Other'].map(opt => (
                                <button key={opt}
                                  onClick={() => setAnswer(wish.id, amb.field, opt)}
                                  style={{
                                    padding: '8px 16px', borderRadius: 20, border: '1px solid',
                                    borderColor: myAnswers[amb.field] === opt ? 'var(--accent)' : 'var(--border)',
                                    background: myAnswers[amb.field] === opt ? 'rgba(59,130,246,0.15)' : 'var(--surface2)',
                                    color: myAnswers[amb.field] === opt ? '#93c5fd' : 'var(--muted)',
                                    fontFamily: 'Syne, sans-serif', fontWeight: 600, fontSize: 12,
                                    cursor: 'pointer',
                                  }}>
                                  {opt}
                                </button>
                              ))}
                            </div>
                          ) : amb.field === 'ui_type' ? (
                            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                              {['Desktop app', 'Web browser', 'On-device touchscreen', 'Mobile app', 'No UI needed'].map(opt => (
                                <button key={opt}
                                  onClick={() => setAnswer(wish.id, amb.field, opt)}
                                  style={{
                                    padding: '8px 16px', borderRadius: 20, border: '1px solid',
                                    borderColor: myAnswers[amb.field] === opt ? 'var(--accent)' : 'var(--border)',
                                    background: myAnswers[amb.field] === opt ? 'rgba(59,130,246,0.15)' : 'var(--surface2)',
                                    color: myAnswers[amb.field] === opt ? '#93c5fd' : 'var(--muted)',
                                    fontFamily: 'Syne, sans-serif', fontWeight: 600, fontSize: 12,
                                    cursor: 'pointer',
                                  }}>
                                  {opt}
                                </button>
                              ))}
                            </div>
                          ) : (
                            // Free text fallback for other question types
                            <textarea className="input" rows={2}
                              style={{ resize: 'none', fontSize: 13 }}
                              placeholder="Your expert answer..."
                              value={myAnswers[amb.field] || ''}
                              onChange={e => setAnswer(wish.id, amb.field, e.target.value)}
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Extra constraints */}
                <div>
                  <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
                    Any additional constraints or must-have skills?
                    <span style={{ color: 'var(--muted)', fontWeight: 400 }}> (optional)</span>
                  </p>
                  <textarea className="input" rows={2} style={{ resize: 'none', fontSize: 13 }}
                    placeholder="e.g. Must have IIR filter experience, BSP bring-up for custom hardware..."
                    value={myAnswers['constraints'] || ''}
                    onChange={e => setAnswer(wish.id, 'constraints', e.target.value)}
                  />
                </div>

                {/* Submit */}
                <button className="btn-primary"
                  onClick={() => handleSubmit(wish)}
                  disabled={!!submitting || (!allAnswered && ambiguities.length > 0)}
                  style={{ opacity: (!allAnswered && ambiguities.length > 0) ? 0.5 : 1 }}>
                  {submitting === wish.id ? 'Submitting...' : '⭐ Submit Expert Review'}
                </button>
                {!allAnswered && ambiguities.length > 0 && (
                  <p style={{ fontSize: 11, color: 'var(--muted)', textAlign: 'center', marginTop: -8 }}>
                    Answer all questions to submit
                  </p>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── RESUME VERSION CARD ───────────────────────────────────────────────────
function ResumeVersionCard({ v }: { v: ResumeVersion }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '10px 14px', background: 'var(--surface2)', borderRadius: 10,
                  marginBottom: 6 }}>
      <div>
        <p style={{ fontSize: 13, fontWeight: 600 }}>📄 {v.label}</p>
        <p style={{ fontSize: 11, color: 'var(--muted)' }}>
          {v.skills_count} skills · {new Date(v.uploaded_at).toLocaleDateString('en-GB', {
            day: 'numeric', month: 'short', year: 'numeric'
          })}
        </p>
      </div>
      <span className="tag tag-green" style={{ fontSize: 10 }}>Active</span>
    </div>
  )
}

// ── MAIN DASHBOARD ────────────────────────────────────────────────────────
export default function EmployeeDashboard() {
  const navigate  = useNavigate()
  const user      = JSON.parse(localStorage.getItem('ti_user') || '{}')
  const [profile, setProfile]   = useState<EmployeeProfile | null>(null)
  const [avail, setAvail]       = useState({ available_percentage: 1.0, status: 'available', free_from_date: '', is_soft_open: false, soft_open_note: '', availability_score: 1.0 })
  const [github, setGithub]     = useState('')
  const [saved, setSaved]       = useState(false)
  const [resumeFile, setResumeFile] = useState<File | null>(null)
  const [uploading, setUploading]   = useState(false)
  const [extracting, setExtracting] = useState(false)
  const [uploadDone, setUploadDone] = useState(false)
  const [loading, setLoading]   = useState(true)
  const [activeTab, setActiveTab]   = useState<'profile'|'sme'>('profile')
  const [resumeVersions, setResumeVersions] = useState<ResumeVersion[]>([])
  const fileRef = useRef<HTMLInputElement>(null)
  const [pocLinks, setPocLinks]       = useState<any[]>([])
  const [showPocForm, setShowPocForm] = useState(false)
  const [pocTitle, setPocTitle]       = useState('')
  const [pocUrl, setPocUrl]           = useState('https://')
  const [pocType, setPocType]         = useState('gitlab')
  const [pocDesc, setPocDesc]         = useState('')
  const [pocSaving, setPocSaving]     = useState(false)

  useEffect(() => {
    const id = user.id || 'emp_demo'
    Promise.all([
      fetch(`${API}/employees/${id}`).then(r => r.ok ? r.json() : null),
      fetch(`${API}/availability/${id}`).then(r => r.ok ? r.json() : null),
    ]).then(([emp, av]) => {
      if (emp) {
        setProfile(emp)
        setGithub(emp.github_username || '')
        setPocLinks(emp.poc_links || [])
        // Build resume version history from profile
        if (emp.resume_text) {
          setResumeVersions([{
            filename:    'resume.pdf',
            uploaded_at: emp.resume_uploaded_at || new Date().toISOString(),
            skills_count: emp.skills?.length || 0,
            label:       `Resume v1 — ${emp.skills?.length || 0} skills extracted`,
          }])
        }
      }
      if (av) setAvail(a => ({ ...a, ...av }))
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const save = async () => {
    const id = user.id || 'emp_demo'
    try {
      // Convert free_from_date to proper ISO format or null
      const freeFromDate = avail.free_from_date
        ? new Date(avail.free_from_date).toISOString()
        : null

      await updateAvailability(id, {
        available_percentage: avail.available_percentage,
        status: avail.status,
        free_from_date: freeFromDate,
        is_soft_open: avail.is_soft_open,
        soft_open_note: avail.soft_open_note || null,
        github_username: github || null,
      })
      setSaved(true)
      setTimeout(() => setSaved(false), 2500)
    } catch (e) {
      console.error('Save failed:', e)
      alert('Failed to save. Check backend is running.')
    }
  }

  const uploadResume = async () => {
    if (!resumeFile) return
    const id = user.id || 'emp_demo'
    setUploading(true)
    try {
      const fd = new FormData(); fd.append('file', resumeFile)
      await fetch(`${API}/employees/${id}/upload-resume`, { method: 'POST', body: fd })
      const label = `Resume v${resumeVersions.length + 1} — extracting skills...`
      setResumeVersions(prev => [{
        filename:    resumeFile.name,
        uploaded_at: new Date().toISOString(),
        skills_count: 0,
        label,
      }, ...prev])
      setUploadDone(true)
      setResumeFile(null)
      // Poll for skills after 30 seconds
      setTimeout(async () => {
        const r = await fetch(`${API}/employees/${id}`)
        if (r.ok) {
          const emp = await r.json()
          const count = emp.skills?.length || 0
          setResumeVersions(prev => prev.map((v, i) =>
            i === 0 ? { ...v, skills_count: count, label: `Resume v${resumeVersions.length + 1} — ${count} skills extracted` } : v
          ))
          setProfile(emp)
        }
      }, 35000)
    } finally {
      setUploading(false)
      setTimeout(() => setUploadDone(false), 3000)
    }
  }

  const addPocLink = async () => {
    if (!pocTitle.trim() || !pocUrl.trim()) return
    const id = user.id || 'emp_demo'
    setPocSaving(true)
    try {
      const res = await fetch(`${API}/employees/${id}/poc-links`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: pocTitle, url: pocUrl, link_type: pocType, description: pocDesc })
      })
      const data = await res.json()
      setPocLinks(data.poc_links || [])
      setPocTitle(''); setPocUrl('https://'); setPocDesc(''); setShowPocForm(false)
    } finally { setPocSaving(false) }
  }

  const removePocLink = async (index: number) => {
    const id = user.id || 'emp_demo'
    const res = await fetch(`${API}/employees/${id}/poc-links/${index}`, { method: 'DELETE' })
    const data = await res.json()
    setPocLinks(data.poc_links || [])
  }

  const logout = () => { localStorage.removeItem('ti_user'); navigate('/login') }
  const pct = Math.round(avail.available_percentage * 100)
  const statusColor = STATUS_OPTS.find(s => s.value === avail.status)?.color || 'var(--muted)'
  const isSME = profile?.is_sme

  if (loading) return (
    <div className="page min-h-screen" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <p style={{ color: 'var(--muted)' }}>Loading profile...</p>
    </div>
  )

  return (
    <div className="page min-h-screen">
      <ParticleNetwork />
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
          <span style={{ fontSize: 13, color: 'var(--muted)' }}>👤 {profile?.full_name || user.name}</span>
          {isSME && <span className="tag tag-amber">⭐ SME</span>}
          <button className="btn-secondary" onClick={logout} style={{ padding: '8px 16px' }}>Logout</button>
        </div>
      </nav>

      <div style={{ maxWidth: 640, margin: '0 auto', padding: '40px 24px', display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* Header */}
        <div className="fade-up">
          <h1 style={{ fontSize: 28, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 4 }}>
            My Profile
          </h1>
          {profile && (
            <p style={{ color: 'var(--muted)', fontSize: 14 }}>
              {profile.title} · {profile.seniority_level}
              {isSME && <span style={{ marginLeft: 8, color: 'var(--amber)' }}>
                ⭐ Expert in {profile.sme_domains?.join(', ')}
              </span>}
            </p>
          )}
        </div>

        {/* Tabs — only show if SME */}
        {isSME && (
          <div style={{ display: 'flex', gap: 8 }}>
            {(['profile', 'sme'] as const).map(t => (
              <button key={t} onClick={() => setActiveTab(t)} style={{
                padding: '8px 20px', borderRadius: 20, border: '1px solid',
                borderColor: activeTab === t ? 'var(--accent)' : 'var(--border)',
                background: activeTab === t ? 'rgba(59,130,246,0.1)' : 'var(--surface2)',
                color: activeTab === t ? '#93c5fd' : 'var(--muted)',
                fontFamily: 'Syne, sans-serif', fontWeight: 600, fontSize: 13, cursor: 'pointer',
              }}>
                {t === 'profile' ? '👤 My Profile' : `⭐ Consultations ${t === 'sme' ? '' : ''}`}
              </button>
            ))}
          </div>
        )}

        {/* SME TAB */}
        {isSME && activeTab === 'sme' && profile && (
          <>
            <div className="card" style={{ borderColor: 'rgba(245,158,11,0.3)', background: 'rgba(245,158,11,0.04)' }}>
              <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--amber)', letterSpacing: '0.1em', marginBottom: 6 }}>
                YOUR EXPERT ROLE
              </p>
              <p style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.7 }}>
                When a Product Owner submits a project that needs <strong style={{ color: 'var(--text)' }}>
                {profile.sme_domains?.join(' or ')} </strong>expertise, you'll be asked to validate the requirements.
                Your answers directly improve the quality of team matches.
              </p>
            </div>
            <SMEConsultationPanel profile={profile} />
          </>
        )}

        {/* PROFILE TAB */}
        {(!isSME || activeTab === 'profile') && (
          <>
            {/* Availability */}
            <div className="card fade-up-1">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em' }}>AVAILABILITY</p>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, color: 'var(--muted)' }}>RANKING SCORE</span>
                  <span className="mono" style={{ fontSize: 22, fontWeight: 800,
                    color: avail.availability_score >= 0.6 ? 'var(--green)' : 'var(--amber)' }}>
                    {Math.round(avail.availability_score * 100)}
                  </span>
                </div>
              </div>

              <div style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: 13, color: 'var(--muted)' }}>Available bandwidth</span>
                  <span className="mono" style={{ fontSize: 18, fontWeight: 800, color: statusColor }}>{pct}%</span>
                </div>
                <input type="range" min={0} max={100} step={10} value={pct}
                  onChange={e => setAvail(p => ({ ...p, available_percentage: +e.target.value / 100 }))}
                  style={{ width: '100%', accentColor: 'var(--accent)', height: 6 }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>
                  <span>Fully booked</span><span>Fully available</span>
                </div>
              </div>

              <div style={{ marginBottom: 16 }}>
                <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>Status</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8 }}>
                  {STATUS_OPTS.map(opt => (
                    <button key={opt.value} onClick={() => setAvail(p => ({ ...p, status: opt.value }))}
                      style={{ padding: '10px 8px', borderRadius: 10, border: '1px solid',
                        borderColor: avail.status === opt.value ? opt.color : 'var(--border)',
                        background: avail.status === opt.value ? `${opt.color}18` : 'var(--surface2)',
                        color: avail.status === opt.value ? opt.color : 'var(--muted)',
                        fontFamily: 'Syne, sans-serif', fontWeight: 600, fontSize: 11,
                        cursor: 'pointer', transition: 'all 0.15s', textAlign: 'center' }}>
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ marginBottom: 16 }}>
                <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>Free from date</p>
                <input type="date" className="input" value={avail.free_from_date?.split('T')[0] || ''}
                  onChange={e => setAvail(p => ({ ...p, free_from_date: e.target.value }))} />
              </div>

              {avail.status === 'soft_open' && (
                <div style={{ marginBottom: 16 }}>
                  <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>Note for project managers</p>
                  <textarea className="input" rows={2} style={{ resize: 'none' }}
                    placeholder="e.g. Available after sprint ends June 30"
                    value={avail.soft_open_note} onChange={e => setAvail(p => ({ ...p, soft_open_note: e.target.value }))} />
                </div>
              )}

              <button className="btn-primary" onClick={save}>{saved ? '✓ Saved!' : 'Save Availability'}</button>
            </div>

            {/* GitHub */}
            <div className="card fade-up-2">
              <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em', marginBottom: 4 }}>GITHUB</p>
              <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 14, lineHeight: 1.6 }}>
                GitHub proves hands-on experience — repos, commits, and languages are scanned to validate your skills.
                Personal projects you've built are your strongest evidence for new opportunities.
              </p>
              <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center' }}>
                <span style={{ fontSize: 13, color: 'var(--muted)', whiteSpace: 'nowrap' }}>github.com/</span>
                <input className="input" placeholder="your-username" value={github} onChange={e => setGithub(e.target.value)} />
                {github && (
                  <a href={`https://github.com/${github}`} target="_blank" rel="noreferrer"
                    style={{ fontSize: 12, color: 'var(--accent)', whiteSpace: 'nowrap', textDecoration: 'none' }}>
                    View ↗
                  </a>
                )}
              </div>
              {profile?.github_stats && (
                <div style={{ background: 'var(--surface2)', borderRadius: 10, padding: '12px 14px' }}>
                  <div style={{ display: 'flex', gap: 20, fontSize: 13, marginBottom: 8 }}>
                    <span>⚡ <strong>{profile.github_stats.total_commits}</strong> commits</span>
                    <span>📁 <strong>{profile.github_stats.active_repos}</strong> active repos</span>
                  </div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {profile.github_stats.top_languages?.map((l: string) => (
                      <span key={l} className="tag tag-gray mono">{l}</span>
                    ))}
                  </div>
                  {profile.github_stats.recent_repos?.length > 0 && (
                    <div style={{ marginTop: 10 }}>
                      <p style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>Recent repos scanned:</p>
                      {profile.github_stats.recent_repos.slice(0, 3).map((r: any) => (
                        <div key={r.name} style={{ fontSize: 12, color: 'var(--muted)', padding: '3px 0' }}>
                          📦 <a href={`https://github.com/${github}/${r.name}`} target="_blank" rel="noreferrer"
                            style={{ color: 'var(--accent)', textDecoration: 'none' }}>{r.name}</a>
                          {r.language && <span style={{ marginLeft: 6, color: 'var(--muted)' }}>· {r.language}</span>}
                          {r.topics?.length > 0 && <span style={{ marginLeft: 6 }}>· {r.topics.slice(0,3).join(', ')}</span>}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
              <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 10 }}>
                ℹ️ GitHub is synced automatically. Your repos are scanned for skills, topics, and README content.
              </p>
            </div>

            {/* POC Links */}
            <div className="card fade-up-2">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em' }}>PROOF OF WORK LINKS</p>
                {!showPocForm && (
                  <button className="btn-secondary" style={{ padding: '6px 14px', fontSize: 11 }}
                    onClick={() => setShowPocForm(true)}>+ Add Link</button>
                )}
              </div>
              <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 14, lineHeight: 1.6 }}>
                Link private GitLab, Confluence, portfolio or live demos.
                Company work not on public GitHub still counts — <strong style={{ color: '#10b981' }}>boosts your ranking score!</strong>
              </p>

              {pocLinks.length > 0 && (
                <div style={{ marginBottom: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {pocLinks.map((link: any, i: number) => (
                    <div key={i} style={{ background: 'var(--surface2)', borderRadius: 10,
                      padding: '12px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                          <span>{link.type === 'gitlab' ? '🦊' : link.type === 'github' ? '🐙' : link.type === 'confluence' ? '📋' : '🔗'}</span>
                          <span style={{ fontSize: 13, fontWeight: 600 }}>{link.title}</span>
                          <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 20,
                            background: 'rgba(16,185,129,0.15)', color: '#10b981', fontWeight: 700 }}>POC VERIFIED</span>
                        </div>
                        <a href={link.url.startsWith('http') ? link.url : `https://${link.url}`}
                          target="_blank" rel="noreferrer"
                          style={{ fontSize: 12, color: 'var(--accent)', textDecoration: 'none' }}>
                          {link.url} ↗
                        </a>
                        {link.description && (
                          <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>{link.description}</p>
                        )}
                      </div>
                      <button onClick={() => removePocLink(i)}
                        style={{ fontSize: 12, color: 'var(--muted)', background: 'none', border: 'none',
                                 cursor: 'pointer', padding: '2px 8px', marginLeft: 8 }}>✕</button>
                    </div>
                  ))}
                </div>
              )}

              {showPocForm && (
                <div style={{ border: '1px solid var(--border)', borderRadius: 10, padding: '14px', marginBottom: 8 }}>
                  <div style={{ marginBottom: 10 }}>
                    <p style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>TYPE</p>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {[
                        { value: 'gitlab',     label: '🦊 GitLab'    },
                        { value: 'github',     label: '🐙 GitHub'    },
                        { value: 'confluence', label: '📋 Confluence' },
                        { value: 'portfolio',  label: '🌐 Portfolio'  },
                        { value: 'demo',       label: '🚀 Live Demo'  },
                        { value: 'other',      label: '🔗 Other'      },
                      ].map(opt => (
                        <button key={opt.value} onClick={() => setPocType(opt.value)}
                          style={{ padding: '6px 12px', borderRadius: 20, border: '1px solid',
                            borderColor: pocType === opt.value ? 'var(--accent)' : 'var(--border)',
                            background: pocType === opt.value ? 'rgba(59,130,246,0.15)' : 'var(--surface2)',
                            color: pocType === opt.value ? '#93c5fd' : 'var(--muted)',
                            fontFamily: 'Syne, sans-serif', fontWeight: 600, fontSize: 11, cursor: 'pointer' }}>
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div style={{ marginBottom: 10 }}>
                    <p style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>PROJECT TITLE *</p>
                    <input className="input" placeholder="e.g. Talent Intelligence Platform"
                      value={pocTitle} onChange={e => setPocTitle(e.target.value)} />
                  </div>
                  <div style={{ marginBottom: 10 }}>
                    <p style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>URL *</p>
                    <input className="input" placeholder="https://gitlab.company.com/your-project"
                      value={pocUrl} onChange={e => setPocUrl(e.target.value)} />
                  </div>
                  <div style={{ marginBottom: 12 }}>
                    <p style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>DESCRIPTION (optional)</p>
                    <textarea className="input" rows={2} style={{ resize: 'none', fontSize: 13 }}
                      placeholder="e.g. Built LangChain RAG pipeline, FAISS vector store, FastAPI backend"
                      value={pocDesc} onChange={e => setPocDesc(e.target.value)} />
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button onClick={() => { setShowPocForm(false); setPocTitle(''); setPocUrl('https://'); setPocDesc('') }}
                      className="btn-secondary" style={{ padding: '10px 16px', flex: 1 }}>Cancel</button>
                    <button onClick={addPocLink} disabled={pocSaving || !pocTitle.trim() || !pocUrl.trim()}
                      className="btn-primary" style={{ flex: 2 }}>
                      {pocSaving ? 'Saving...' : '+ Add Link'}
                    </button>
                  </div>
                </div>
              )}

              {pocLinks.length === 0 && !showPocForm && (
                <p style={{ fontSize: 12, color: 'var(--muted)', textAlign: 'center', padding: '8px 0' }}>
                  No proof of work links yet — add your company projects to boost your ranking!
                </p>
              )}
            </div>

            {/* Resume — versioned */}
            <div className="card fade-up-2">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em' }}>RESUME</p>
                <button className="btn-secondary" style={{ padding: '6px 14px', fontSize: 11 }}
                  onClick={() => fileRef.current?.click()}>+ Upload New</button>
              </div>
              <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 14, lineHeight: 1.6 }}>
                Each version is kept. Skills are extracted automatically after upload and cross-referenced with your GitHub to validate hands-on experience.
              </p>

              {/* Version history */}
              {resumeVersions.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <p style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 8 }}>VERSION HISTORY</p>
                  {resumeVersions.map((v, i) => <ResumeVersionCard key={i} v={v} />)}
                </div>
              )}

              {/* Upload zone */}
              {!profile?.resume_text && resumeVersions.length === 0 && (
                <div onClick={() => fileRef.current?.click()}
                  style={{ border: '2px dashed var(--border)', borderRadius: 12, padding: '28px 20px',
                            textAlign: 'center', cursor: 'pointer', marginBottom: 12, transition: 'border-color 0.2s' }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent)')}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}>
                  <p style={{ fontSize: 24, marginBottom: 8 }}>📄</p>
                  <p style={{ fontSize: 14, fontWeight: 600 }}>Upload your resume</p>
                  <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>PDF only · Max 5MB</p>
                </div>
              )}

              <input ref={fileRef} type="file" accept=".pdf" style={{ display: 'none' }}
                onChange={e => setResumeFile(e.target.files?.[0] || null)} />

              {resumeFile && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ background: 'var(--surface2)', borderRadius: 10, padding: '10px 14px',
                                display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                    <span style={{ fontSize: 13 }}>📎 {resumeFile.name}</span>
                    <span style={{ fontSize: 12, color: 'var(--muted)' }}>{(resumeFile.size / 1024).toFixed(0)} KB</span>
                  </div>
                  <button className="btn-primary" onClick={uploadResume} disabled={uploading || extracting}>
                    {uploading ? 'Uploading...' : extracting ? '🧠 Extracting skills...' : uploadDone ? '✓ Done!' : 'Upload & Extract Skills'}
                  </button>
                  {extracting && (
                    <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 8, textAlign: 'center' }}>
                      GPT-4 is reading your resume and cross-referencing with GitHub...
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Skills */}
            {profile?.skills && profile.skills.length > 0 && (
              <div className="card fade-up-3">
                <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em', marginBottom: 4 }}>
                  MY SKILLS ({profile.skills.length})
                </p>
                <p style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 14 }}>
                  🔧 hands-on · 📖 theoretical · ✓ GitHub verified · Click a skill to see evidence
                </p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {profile.skills.map(s => <SkillTag key={s.name} skill={s} />)}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
