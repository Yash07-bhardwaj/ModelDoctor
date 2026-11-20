import axios from "axios";

const http = axios.create({ baseURL: import.meta.env.VITE_API_URL || "" });

/** Turn any axios error into a readable message (backend returns {detail, code}). */
export function errorMessage(err) {
  if (err?.response?.data?.detail) return String(err.response.data.detail);
  if (err?.response) return `Request failed (${err.response.status})`;
  if (err?.request) return "Cannot reach the ModelDoctor API. Is the backend running on port 8000?";
  return err?.message || "Unexpected error";
}

const data = (p) => p.then((r) => r.data);

export const api = {
  health: () => data(http.get("/api/health")),
  listSessions: () => data(http.get("/api/sessions")),
  deleteSession: (id) => data(http.delete(`/api/sessions/${id}`)),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return data(http.post("/api/datasets/upload", form));
  },
  profile: (id, target) => data(http.get(`/api/datasets/${id}/profile`, { params: { target_column: target } })),
  train: (id, body) => data(http.post(`/api/train/${id}`, body)),
  status: (id) => data(http.get(`/api/train/${id}/status`)),
  report: (id) => data(http.get(`/api/report/${id}`)),
  schema: (id) => data(http.get(`/api/schema/${id}`)),
  predict: (id, features) => data(http.post(`/api/predict/${id}`, { features })),
  predictBatch: (id, file) => {
    const form = new FormData();
    form.append("file", file);
    return data(http.post(`/api/predict/${id}/batch`, form));
  },
  feedback: (id, items) => data(http.post(`/api/feedback/${id}`, { items })),
  monitoring: (id, params = {}) => data(http.get(`/api/monitoring/${id}`, { params })),
};
