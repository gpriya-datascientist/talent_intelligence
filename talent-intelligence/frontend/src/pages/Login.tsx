import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

export default function Login() {
  const [role, setRole] = useState<'po' | 'employee'>('po')
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const handleContinue = async () => {
    if (!email.trim()) { setError('Please enter your email'); return }
    setLoading(true); setError('')
    await new Promise(r => setTimeout(r, 600))

    if (role === 'po') {
      localStorage.setItem('ti_user', JSON.stringify({
        role: 'po',
        email,
        name: name || email.split('@')[0],
        id: 'po_' + email.replace(/[^a-z0-9]/gi, '_'),
      }))
      navigate('/')
    } else {
      try {
        const res = await fetch(`http://localhost:8000/employees/by-email/${encodeURIComponent(email)}`)
        if (res.ok) {
          const emp = await res.json()
          localStorage.setItem('ti_user', JSON.stringify({
            role: 'employee', email,
            name: emp.full_name, id: emp.id,
          }))
          navigate('/employee')
        } else {
          setError('Employee not found. Check your email or contact HR.')
        }
      } catch {
        // Dev fallback — use first seeded employee
        localStorage.setItem('ti_user', JSON.stringify({
          role: 'employee', email,
          name: email.split('@')[0], id: 'emp_demo',
        }))
        navigate('/employee')
      }
    }
    setLoading(false)
  }

  return (
    <div className="page min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-md fade-up">

        {/* Logo */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-3 mb-4">
            <div style={{
              width: 44, height: 44, borderRadius: 12,
              background: 'linear-gradient(135deg,#3b82f6,#06b6d4)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 22
            }}>🧠</div>
            <span style={{ fontSize: 22, fontWeight: 800, letterSpacing: '-0.02em' }}>
              Talent Intelligence
            </span>
          </div>
          <p style={{ color: 'var(--muted)', fontSize: 14 }}>
            AI-powered team assembly platform
          </p>
        </div>

        <div className="card">
          {/* Role selector */}
          <p style={{ fontSize: 12, fontWeight: 700, color: 'var(--muted)', letterSpacing: '0.1em', marginBottom: 12 }}>
            I AM A
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 24 }}>
            {(['po', 'employee'] as const).map(r => (
              <button key={r} onClick={() => setRole(r)} style={{
                padding: '14px 12px', borderRadius: 12, border: '2px solid',
                borderColor: role === r ? 'var(--accent)' : 'var(--border)',
                background: role === r ? 'rgba(59,130,246,0.1)' : 'var(--surface2)',
                color: role === r ? '#93c5fd' : 'var(--muted)',
                fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: 13,
                cursor: 'pointer', transition: 'all 0.2s',
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6
              }}>
                <span style={{ fontSize: 22 }}>{r === 'po' ? '🎯' : '👤'}</span>
                {r === 'po' ? 'Product Owner' : 'Employee'}
              </button>
            ))}
          </div>

          {/* Fields */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 20 }}>
            {role === 'po' && (
              <input
                className="input" placeholder="Your name"
                value={name} onChange={e => setName(e.target.value)}
              />
            )}
            <input
              className="input" placeholder="your@email.com" type="email"
              value={email} onChange={e => setEmail(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleContinue()}
            />
          </div>

          {error && (
            <p style={{ color: 'var(--red)', fontSize: 13, marginBottom: 12 }}>{error}</p>
          )}

          <button className="btn-primary" onClick={handleContinue} disabled={loading}>
            {loading ? 'Signing in...' : 'Continue →'}
          </button>
        </div>

        <p style={{ textAlign: 'center', color: 'var(--muted)', fontSize: 12, marginTop: 20 }}>
          Demo: use any email as PO · use a seeded employee email as Employee
        </p>
      </div>
    </div>
  )
}
