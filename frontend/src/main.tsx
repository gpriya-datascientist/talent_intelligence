import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import './index.css'
import Login from './pages/Login'
import PODashboard from './pages/PODashboard'
import EmployeeDashboard from './pages/EmployeeDashboard'

function ProtectedRoute({ children, role }: { children: React.ReactNode, role: string }) {
  const user = JSON.parse(localStorage.getItem('ti_user') || 'null')
  if (!user) return <Navigate to="/login" replace />
  if (user.role !== role) return <Navigate to={user.role === 'po' ? '/' : '/employee'} replace />
  return <>{children}</>
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<ProtectedRoute role="po"><PODashboard /></ProtectedRoute>} />
        <Route path="/employee" element={<ProtectedRoute role="employee"><EmployeeDashboard /></ProtectedRoute>} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)
