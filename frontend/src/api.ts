// api.ts — typed API hooks for both dashboards
const API = "http://localhost:8000";

// ── WISHES ────────────────────────────────────────────────────────────────
export async function submitWish(poId: string, wishText: string) {
  const res = await fetch(`${API}/wishes/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ po_id: poId, wish_text: wishText }),
  });
  if (!res.ok) throw new Error("Failed to submit wish");
  return res.json();
}

export async function getWish(wishId: string) {
  const res = await fetch(`${API}/wishes/${wishId}`);
  if (!res.ok) throw new Error("Failed to fetch wish");
  return res.json();
}

// ── SME ───────────────────────────────────────────────────────────────────
export async function getPendingSMEWishes(smeId: string, domain: string) {
  const res = await fetch(`${API}/wishes/pending-sme?domain=${encodeURIComponent(domain)}&sme_id=${smeId}`);
  if (!res.ok) return [];
  return res.json();
}

export async function submitSMEInput(
  wishId: string,
  smeId: string,
  domain: string,
  answers: Record<string, unknown>
) {
  const res = await fetch(`${API}/wishes/${wishId}/sme-input`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sme_id: smeId, domain, answers }),
  });
  if (!res.ok) throw new Error("Failed to submit SME input");
  return res.json();
}

// ── AVAILABILITY ──────────────────────────────────────────────────────────
export async function updateAvailability(employeeId: string, data: object) {
  const res = await fetch(`${API}/availability/${employeeId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to update availability");
  return res.json();
}

export async function getAvailability(employeeId: string) {
  const res = await fetch(`${API}/availability/${employeeId}`);
  if (!res.ok) return null;
  return res.json();
}

// ── EMPLOYEE ──────────────────────────────────────────────────────────────
export async function triggerSkillExtraction(employeeId: string) {
  const res = await fetch(`${API}/employees/${employeeId}/extract-skills`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Extraction failed");
  return res.json();
}

export async function updateGithub(employeeId: string, username: string) {
  const res = await fetch(`${API}/employees/${employeeId}/github`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ github_username: username }),
  });
  if (!res.ok) throw new Error("Failed to update GitHub");
  return res.json();
}
