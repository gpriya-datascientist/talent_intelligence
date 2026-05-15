import { useState, useEffect } from "react";

const API = "http://localhost:8000";
const EMPLOYEE_ID = "emp_demo"; // In real app: from auth context

interface AvailabilityState {
  available_percentage: number;
  status: string;
  free_from_date: string;
  is_soft_open: boolean;
  soft_open_note: string;
  preferred_domains: string;
  availability_score: number;
}

export default function EmployeeDashboard() {
  const [avail, setAvail] = useState<AvailabilityState>({
    available_percentage: 1.0,
    status: "available",
    free_from_date: "",
    is_soft_open: false,
    soft_open_note: "",
    preferred_domains: "",
    availability_score: 1.0,
  });
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    fetch(`${API}/availability/${EMPLOYEE_ID}`)
      .then(r => r.json())
      .then(data => setAvail(data))
      .catch(() => {});
  }, []);

  const save = async () => {
    const res = await fetch(`${API}/availability/${EMPLOYEE_ID}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(avail),
    });
    const data = await res.json();
    setAvail(prev => ({ ...prev, availability_score: data.availability_score }));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const pct = Math.round(avail.available_percentage * 100);

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      <h1 className="text-3xl font-bold mb-2">My Availability</h1>
      <p className="text-gray-400 mb-8">Let the system know your current bandwidth.</p>

      <div className="max-w-lg space-y-6">
        {/* Availability Slider */}
        <div className="bg-gray-900 rounded-xl p-5">
          <label className="text-sm text-gray-400 block mb-3">
            Available bandwidth: <span className="text-white font-bold text-lg">{pct}%</span>
          </label>
          <input
            type="range" min={0} max={100} step={10}
            value={pct}
            onChange={(e) => setAvail(prev => ({ ...prev, available_percentage: Number(e.target.value) / 100 }))}
            className="w-full accent-blue-500"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Fully booked</span>
            <span>Fully available</span>
          </div>
        </div>

        {/* Status */}
        <div className="bg-gray-900 rounded-xl p-5">
          <label className="text-sm text-gray-400 block mb-2">Status</label>
          <select
            className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-white"
            value={avail.status}
            onChange={(e) => setAvail(prev => ({ ...prev, status: e.target.value }))}
          >
            <option value="available">Available</option>
            <option value="partially_available">Partially Available</option>
            <option value="busy">Busy</option>
            <option value="on_leave">On Leave</option>
            <option value="soft_open">Busy but open to approach</option>
          </select>
        </div>

        {/* Free from date */}
        <div className="bg-gray-900 rounded-xl p-5">
          <label className="text-sm text-gray-400 block mb-2">Free from date</label>
          <input
            type="date"
            className="bg-gray-800 border border-gray-700 rounded-lg p-2 text-white w-full"
            value={avail.free_from_date?.split("T")[0] || ""}
            onChange={(e) => setAvail(prev => ({ ...prev, free_from_date: e.target.value }))}
          />
        </div>

        {/* Soft open note */}
        {avail.is_soft_open && (
          <div className="bg-gray-900 rounded-xl p-5">
            <label className="text-sm text-gray-400 block mb-2">Note for project managers</label>
            <textarea
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-white resize-none h-20"
              placeholder="e.g. Available after current sprint ends on June 30"
              value={avail.soft_open_note}
              onChange={(e) => setAvail(prev => ({ ...prev, soft_open_note: e.target.value }))}
            />
          </div>
        )}

        {/* Availability score preview */}
        <div className="bg-gray-900 rounded-xl p-5 flex justify-between items-center">
          <span className="text-gray-400 text-sm">Your ranking score</span>
          <span className="text-2xl font-bold text-green-400">
            {(avail.availability_score * 100).toFixed(0)}
          </span>
        </div>

        <button
          onClick={save}
          className="w-full bg-blue-600 hover:bg-blue-700 py-3 rounded-xl font-semibold"
        >
          {saved ? "Saved ✓" : "Save Availability"}
        </button>
      </div>
    </div>
  );
}
