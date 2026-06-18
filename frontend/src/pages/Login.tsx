import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

// ── PARTICLE NETWORK CANVAS ───────────────────────────────────────────────
function ParticleNetwork() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx    = canvas.getContext('2d')!
    let raf: number
    let mouse = { x: -1000, y: -1000 }

    const resize = () => {
      canvas.width  = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)
    window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY })

    // Create particles
    const COUNT  = 80
    const particles = Array.from({ length: COUNT }, () => ({
      x:   Math.random() * canvas.width,
      y:   Math.random() * canvas.height,
      vx:  (Math.random() - 0.5) * 0.4,
      vy:  (Math.random() - 0.5) * 0.4,
      r:   Math.random() * 2 + 1,
      pulse: Math.random() * Math.PI * 2,
    }))

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update + draw particles
      for (const p of particles) {
        p.x += p.vx
        p.y += p.vy
        p.pulse += 0.02

        // Bounce edges
        if (p.x < 0 || p.x > canvas.width)  p.vx *= -1
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1

        // Mouse attraction
        const dx = mouse.x - p.x
        const dy = mouse.y - p.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 150) {
          p.x += dx * 0.003
          p.y += dy * 0.003
        }

        // Draw particle
        const alpha = 0.5 + Math.sin(p.pulse) * 0.2
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(59,130,246,${alpha})`
        ctx.fill()
      }

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx   = particles[i].x - particles[j].x
          const dy   = particles[i].y - particles[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)

          if (dist < 130) {
            const alpha = (1 - dist / 130) * 0.35
            ctx.beginPath()
            ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.strokeStyle = `rgba(59,130,246,${alpha})`
            ctx.lineWidth   = 0.8
            ctx.stroke()
          }
        }

        // Connection to mouse
        const dx   = particles[i].x - mouse.x
        const dy   = particles[i].y - mouse.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 180) {
          const alpha = (1 - dist / 180) * 0.6
          ctx.beginPath()
          ctx.moveTo(particles[i].x, particles[i].y)
          ctx.lineTo(mouse.x, mouse.y)
          ctx.strokeStyle = `rgba(99,179,246,${alpha})`
          ctx.lineWidth   = 1
          ctx.stroke()
        }
      }

      raf = requestAnimationFrame(draw)
    }

    draw()
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas ref={canvasRef} style={{
      position: 'fixed', top: 0, left: 0,
      width: '100%', height: '100%',
      pointerEvents: 'none', zIndex: 0,
    }} />
  )
}

// ── LOGIN PAGE ─────────────────────────────────────────────────────────────
export default function Login() {
  const [role, setRole]     = useState<'po' | 'employee'>('po')
  const [email, setEmail]   = useState('')
  const [name, setName]     = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState('')
  const navigate = useNavigate()

  const handleContinue = async () => {
    if (!email.trim()) { setError('Please enter your email'); return }
    setLoading(true); setError('')
    await new Promise(r => setTimeout(r, 600))

    if (role === 'po') {
      localStorage.setItem('ti_user', JSON.stringify({
        role: 'po', email,
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
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center',
      justifyContent: 'center', padding: 24, position: 'relative',
      background: 'radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.08) 0%, transparent 60%), radial-gradient(ellipse at 80% 20%, rgba(6,182,212,0.06) 0%, transparent 60%), #080b10',
    }}>
      {/* Particle network */}
      <ParticleNetwork />

      {/* Content */}
      <div style={{ position: 'relative', zIndex: 1, width: '100%', maxWidth: 420 }}
           className="fade-up">

        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 40 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
            <div style={{
              width: 48, height: 48, borderRadius: 14,
              background: 'linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 32px rgba(59,130,246,0.4)',
              fontSize: 24,
            }}>✦</div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1 }}>
                Talent Intelligence
              </div>
              <div style={{ fontSize: 12, color: 'var(--muted)', letterSpacing: '0.08em', marginTop: 3 }}>
                AI-POWERED TEAM ASSEMBLY
              </div>
            </div>
          </div>

          {/* Tagline */}
          <p style={{ color: 'var(--muted)', fontSize: 14, lineHeight: 1.6, maxWidth: 300, margin: '0 auto' }}>
            Find the hidden gems in your team before hiring externally
          </p>
        </div>

        {/* Card */}
        <div style={{
          background: 'rgba(13,17,26,0.85)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(59,130,246,0.2)',
          borderRadius: 20,
          padding: '32px 28px',
          boxShadow: '0 24px 64px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.03)',
        }}>
          {/* Role selector */}
          <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)',
                       letterSpacing: '0.12em', marginBottom: 12 }}>I AM A</p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 28 }}>
            {(['po', 'employee'] as const).map(r => (
              <button key={r} onClick={() => { setRole(r); setError('') }} style={{
                padding: '16px 12px', borderRadius: 14, border: '2px solid',
                borderColor: role === r ? '#3b82f6' : 'rgba(255,255,255,0.07)',
                background: role === r
                  ? 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(6,182,212,0.08))'
                  : 'rgba(255,255,255,0.03)',
                color: role === r ? '#93c5fd' : 'var(--muted)',
                fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: 13,
                cursor: 'pointer', transition: 'all 0.25s',
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
                boxShadow: role === r ? '0 0 20px rgba(59,130,246,0.2)' : 'none',
              }}>
                <span style={{ fontSize: 24 }}>{r === 'po' ? '🎯' : '👤'}</span>
                {r === 'po' ? 'Product Owner' : 'Employee'}
              </button>
            ))}
          </div>

          {/* Fields */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 20 }}>
            {role === 'po' && (
              <input className="input" placeholder="Your name"
                value={name} onChange={e => setName(e.target.value)}
                style={{ background: 'rgba(255,255,255,0.04)' }} />
            )}
            <input className="input" placeholder="your@email.com" type="email"
              value={email} onChange={e => { setEmail(e.target.value); setError('') }}
              onKeyDown={e => e.key === 'Enter' && handleContinue()}
              style={{ background: 'rgba(255,255,255,0.04)' }} />
          </div>

          {error && (
            <p style={{ color: '#f87171', fontSize: 13, marginBottom: 14,
                         padding: '10px 14px', background: 'rgba(239,68,68,0.1)',
                         borderRadius: 10, border: '1px solid rgba(239,68,68,0.2)' }}>
              {error}
            </p>
          )}

          <button onClick={handleContinue} disabled={loading} style={{
            width: '100%', padding: '15px',
            background: loading ? 'rgba(59,130,246,0.4)' : 'linear-gradient(135deg, #3b82f6, #06b6d4)',
            border: 'none', borderRadius: 12,
            color: 'white', fontFamily: 'Syne, sans-serif',
            fontWeight: 700, fontSize: 15, cursor: loading ? 'not-allowed' : 'pointer',
            boxShadow: loading ? 'none' : '0 0 28px rgba(59,130,246,0.35)',
            transition: 'all 0.25s',
          }}>
            {loading ? '⏳ Signing in...' : 'Continue →'}
          </button>
        </div>

        <p style={{ textAlign: 'center', color: 'var(--muted)', fontSize: 11,
                     marginTop: 20, letterSpacing: '0.04em' }}>
          Demo: use any email as PO · use a seeded employee email as Employee
        </p>
      </div>
    </div>
  )
}
