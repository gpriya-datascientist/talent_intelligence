import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

const API = 'http://localhost:8000'

interface EmployeeProfile {
  id: string; full_name: string; title: string; seniority_level: string
  github_username?: string; github_stats?: any; resume_text?: string
  is_sme?: boolean; sme_domains?: string[]
  skills?: { name: string; proficiency: string; is_hands_on: boolean; skill_type: string }[]
  availability?: {
    available_percentage: number; status: string
    free_from_date?: string; is_soft_open: boolean
    soft_open_note?: string; availability_score: number
  }
}

const STATUS_OPTS = [
  { value: 'available',           label: 'Available',              color: 'var(--green)' },
  { value: 'partially_available', label: 'Partially Available',    color: 'var(--amber)' },
  { value: 'busy',                label: 'Busy',                   color: 'var(--red)'   },
  { value: 'on_leave',            label: 'On Leave',               color: 'var(--muted)' },
  { value: 'soft_open',           label: 'Busy but open to talk',  color: 'var(--accent)'},
]

function SkillTag({ skill }: { skill: EmployeeProfile['skills'][0] }) {
  const cls = skill.is_hands_on ? 'tag-blue' : 'tag-gray'
  const icon = skill.skill_type === 'tool' ? '🔨' : skill.is_hands_on ? '🔧' : '📖'
  return (
    <span className={`tag ${cls}`}>
      {icon} {skill.name}
      <span style={{ opacity: 0.6, marginLeft: 2 }}>· {skill.proficiency}</span>
    </span>
  )
}

export default function EmployeeDashboard() {
  const navigate = useNavigate()
  const user = JSON.parse(localStorage.getItem('ti_user') || '{}')
  const [profile, setProfile] = useState<EmployeeProfile | null>(null)
  const [avail, setAvail] = useState({
    available_percentage: 1.0, status: 'available',
    free_from_date: '', is_soft_open: false,
    soft_open_note: '', availability_score: 1.0,
  })
  const [github, setGithub] = useState('')
  const [saved, setSaved] = useState(false)
  const [resumeFile, setResumeFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadDone, setUploadDone] = useState(false)
  const [loading, setLoading] = useState(true)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const empId = user.id || 'emp_demo'
    Promise.all([
      fetch(`${API}/employees/${empId}`).then(r => r.ok ? r.json() : null),
      fetch(`${API}/availability/${empId}`).then(r => r.ok ? r.json() : null),
    ]).then(([emp, av]) => {
      if (emp) { setProfile(emp); setGithub(emp.github_username || '') }
      if (av) setAvail({ ...avail, ...av })
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const save = async () => {
    const empId = user.id || 'emp_demo'
    await fetch(`${API}/availability/${empId}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...avail, github_username: github }),
    })
    setSaved(true); setTimeout(() => setSaved(false), 2500)
  }

  const uploadResume = async () => {
    if (!resumeFile) return
    const empId = user.id || 'emp_demo'
    setUploading(true)
    const fd = new FormData(); fd.append('file', resumeFile)
    await fetch(`${API}/employees/${empId}/upload-resume`, { method: 'POST', body: fd })
    setUploading(false); setUploadDone(true)
    setTimeout(() => setUploadDone(false), 3000)
  }

  const logout = () => { localStorage.removeItem('ti_user'); navigate('/login') }
  const pct = Math.round(avail.available_percentage * 100)
  const statusColor = STATUS_OPTS.find(s => s.value === avail.status)?.color || 'var(--muted)'

  if (loading) return (
    <div className="page min-h-screen flex items-center justify-center">
      <p style={{ color: 'var(--muted)' }}>Loading profile...</p>
    </div>
  )

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
            👤 {profile?.full_name || user.name}
          </span>
          <button className="btn-secondary" onClick={logout} style={{ padding: '8px 16px' }}>
            Logout
          </button>
        </div>
      </nav>

      <div style={{ maxWidth: 640, margin: '0 auto', padding: '40px 24px',
                    display: 'flex', flexDirection: 'column', gap: 20 }}>
        {/* Header */}
        <div className="fade-up">
          <h1 style={{ fontSize: 28, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 4 }}>
            My Profile
          </h1>
          {profile && (
            <p style={{ color: 'var(--muted)', fontSize: 14 }}>
              {profile.title} · {profile.seniority_level}
              {profile.is_sme && (
                <span className="tag tag-amber" style={{ marginLeft: 10 }}>
                  ⭐ SME: {profile.sme_domains?.join(', ')}
                </span>
              )}
            </p>
          )}
        </div>

        {/* Availability */}
        <div className="card fade-up-1">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em' }}>
              AVAILABILITY
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 11, color: 'var(--muted)' }}>RANKING SCORE</span>
              <span className="mono" style={{ fontSize: 22, fontWeight: 800,
                color: avail.availability_score >= 0.6 ? 'var(--green)' : 'var(--amber)' }}>
                {Math.round(avail.availability_score * 100)}
              </span>
            </div>
          </div>

          {/* Slider */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
              <span style={{ fontSize: 13, color: 'var(--muted)' }}>Available bandwidth</span>
              <span className="mono" style={{ fontSize: 18, fontWeight: 800, color: statusColor }}>
                {pct}%
              </span>
            </div>
            <input type="range" min={0} max={100} step={10} value={pct}
              onChange={e => setAvail(p => ({ ...p, available_percentage: +e.target.value / 100 }))}
              style={{ width: '100%', accentColor: 'var(--accent)', height: 6 }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between',
                          fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>
              <span>Fully booked</span><span>Fully available</span>
            </div>
          </div>

          {/* Status */}
          <div style={{ marginBottom: 16 }}>
            <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>Status</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8 }}>
              {STATUS_OPTS.map(opt => (
                <button key={opt.value} onClick={() => setAvail(p => ({ ...p, status: opt.value }))}
                  style={{
                    padding: '10px 8px', borderRadius: 10, border: '1px solid',
                    borderColor: avail.status === opt.value ? opt.color : 'var(--border)',
                    background: avail.status === opt.value ? `${opt.color}18` : 'var(--surface2)',
                    color: avail.status === opt.value ? opt.color : 'var(--muted)',
                    fontFamily: 'Syne, sans-serif', fontWeight: 600, fontSize: 11,
                    cursor: 'pointer', transition: 'all 0.15s', textAlign: 'center',
                  }}>
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Free from date */}
          <div style={{ marginBottom: 16 }}>
            <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>Free from date</p>
            <input type="date" className="input"
              value={avail.free_from_date?.split('T')[0] || ''}
              onChange={e => setAvail(p => ({ ...p, free_from_date: e.target.value }))}
            />
          </div>

          {/* Soft open note */}
          {avail.status === 'soft_open' && (
            <div style={{ marginBottom: 16 }}>
              <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>
                Note for project managers
              </p>
              <textarea className="input" rows={2} style={{ resize: 'none' }}
                placeholder="e.g. Available after sprint ends June 30"
                value={avail.soft_open_note}
                onChange={e => setAvail(p => ({ ...p, soft_open_note: e.target.value }))}
              />
            </div>
          )}

          <button className="btn-primary" onClick={save}>
            {saved ? '✓ Saved!' : 'Save Availability'}
          </button>
        </div>

        {/* Resume */}
        <div className="card fade-up-2">
          <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)',
                      letterSpacing: '0.1em', marginBottom: 16 }}>RESUME</p>
          {profile?.resume_text ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                          marginBottom: 16 }}>
              <div>
                <span className="tag tag-green">✅ Resume loaded</span>
                <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 6 }}>
                  {profile.skills?.length || 0} skills extracted
                </p>
              </div>
              <button className="btn-secondary" onClick={() => fileRef.current?.click()}>
                Re-upload PDF
              </button>
            </div>
          ) : (
            <div onClick={() => fileRef.current?.click()}
              style={{ border: '2px dashed var(--border)', borderRadius: 12, padding: '28px 20px',
                        textAlign: 'center', cursor: 'pointer', marginBottom: 12,
                        transition: 'border-color 0.2s' }}
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
            <>
              <div style={{ background: 'var(--surface2)', borderRadius: 10, padding: '10px 14px',
                            display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <span style={{ fontSize: 13 }}>📎 {resumeFile.name}</span>
                <span style={{ fontSize: 12, color: 'var(--muted)' }}>
                  {(resumeFile.size / 1024).toFixed(0)} KB
                </span>
              </div>
              <button className="btn-primary" onClick={uploadResume} disabled={uploading}>
                {uploading ? 'Uploading...' : uploadDone ? '✓ Uploaded!' : 'Upload Resume'}
              </button>
            </>
          )}
        </div>

        {/* GitHub */}
        <div className="card fade-up-2">
          <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)',
                      letterSpacing: '0.1em', marginBottom: 16 }}>GITHUB</p>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center' }}>
            <span style={{ fontSize: 13, color: 'var(--muted)', whiteSpace: 'nowrap' }}>github.com/</span>
            <input className="input" placeholder="your-username" value={github}
              onChange={e => setGithub(e.target.value)} />
          </div>
          {profile?.github_stats && (
            <div style={{ background: 'var(--surface2)', borderRadius: 10,
                          padding: '12px 14px', fontSize: 13 }}>
              <div style={{ display: 'flex', gap: 20, marginBottom: 8 }}>
                <span>⚡ <strong>{profile.github_stats.total_commits}</strong> commits</span>
                <span>📁 <strong>{profile.github_stats.active_repos}</strong> repos</span>
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {profile.github_stats.top_languages?.map((l: string) => (
                  <span key={l} className="tag tag-gray mono">{l}</span>
                ))}
              </div>
            </div>
          )}
          <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 10 }}>
            GitHub activity proves hands-on experience — boosts your match score
          </p>
        </div>

        {/* Skills */}
        {profile?.skills && profile.skills.length > 0 && (
          <div className="card fade-up-3">
            <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)',
                        letterSpacing: '0.1em', marginBottom: 16 }}>
              MY SKILLS ({profile.skills.length})
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
              {profile.skills.map(s => <SkillTag key={s.name} skill={s} />)}
            </div>
            <p style={{ fontSize: 11, color: 'var(--muted)' }}>
              🔧 hands-on · 📖 theoretical · 🔨 tool
            </p>
          </div>
        )}

      </div>
    </div>
  )
}
