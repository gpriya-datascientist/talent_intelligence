import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import './index.css'
import PODashboard from './pages/PODashboard'
import EmployeeDashboard from './pages/EmployeeDashboard'

function Nav() {
  return (
    <nav className="bg-gray-900 border-b border-gray-800 px-8 py-4 flex gap-6">
      <span className="font-bold text-blue-400 mr-4">🧠 Talent Intelligence</span>
      <Link to="/" className="text-gray-300 hover:text-white text-sm">Product Owner</Link>
      <Link to="/employee" className="text-gray-300 hover:text-white text-sm">My Availability</Link>
    </nav>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<PODashboard />} />
        <Route path="/employee" element={<EmployeeDashboard />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)
