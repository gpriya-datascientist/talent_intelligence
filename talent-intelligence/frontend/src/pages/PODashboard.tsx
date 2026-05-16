import { useState } from "react";

const API = "http://localhost:8000";

type WishStatus = "pending" | "parsing" | "enriching" | "matching" | "completed" | "failed";

interface Candidate {
  employee_id: string;
  rank: number;
  score: number;
  matched_skills: string[];
}

interface WishResult {
  id: string;
  status: WishStatus;
  parsed_intent: string | null;
  matched_candidates: Candidate[] | null;
}

const STATUS_LABELS: Record<WishStatus, string> = {
  pending: "Queued...",
  parsing: "Understanding your wish...",
  enriching: "Consulting domain experts...",
  matching: "Finding best candidates...",
  completed: "Done!",
  failed: "Something went wrong",
};

export default function PODashboard() {
  const [wishText, setWishText] = useState("");
  const [result, setResult] = useState<WishResult | null>(null);
  const [loading, setLoading] = useState(false);

  const submitWish = async () => {
    if (!wishText.trim()) return;
    setLoading(true);
    setResult(null);

    const res = await fetch(`${API}/wishes/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ po_id: "po_demo", wish_text: wishText }),
    });
    const data = await res.json();
    setResult(data);
    pollStatus(data.id);
  };

  const pollStatus = async (wishId: string) => {
    const interval = setInterval(async () => {
      const res = await fetch(`${API}/wishes/${wishId}`);
      const data: WishResult = await res.json();
      setResult(data);
      if (data.status === "completed" || data.status === "failed") {
        clearInterval(interval);
        setLoading(false);
      }
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      <h1 className="text-3xl font-bold mb-2">Talent Intelligence</h1>
      <p className="text-gray-400 mb-8">Describe your project — we'll find the right people.</p>

      <div className="max-w-2xl">
        <textarea
          className="w-full bg-gray-900 border border-gray-700 rounded-xl p-4 text-white resize-none h-32 focus:outline-none focus:border-blue-500"
          placeholder="e.g. We need to build speaker tuning software for embedded devices..."
          value={wishText}
          onChange={(e) => setWishText(e.target.value)}
        />
        <button
          onClick={submitWish}
          disabled={loading}
          className="mt-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 px-6 py-2 rounded-lg font-semibold"
        >
          {loading ? "Processing..." : "Find Team"}
        </button>
      </div>

      {result && (
        <div className="max-w-2xl mt-8">
          <div className="bg-gray-900 rounded-xl p-4 mb-4">
            <span className="text-sm text-gray-400">Status: </span>
            <span className="font-semibold text-blue-400">
              {STATUS_LABELS[result.status]}
            </span>
            {result.parsed_intent && (
              <p className="text-gray-300 mt-2 text-sm">"{result.parsed_intent}"</p>
            )}
          </div>

          {result.matched_candidates && (
            <div className="space-y-3">
              <h2 className="text-lg font-semibold">Recommended Team</h2>
              {result.matched_candidates.filter(c => !c.is_backup).map((c) => (
                <div key={c.employee_id} className="bg-gray-900 rounded-xl p-4 border border-gray-800">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-mono text-sm text-gray-400">#{c.rank}</span>
                    <span className="text-green-400 font-semibold">{(c.score * 100).toFixed(0)}% match</span>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {c.matched_skills.map((s) => (
                      <span key={s} className="bg-blue-900 text-blue-200 text-xs px-2 py-1 rounded-full">{s}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
