// frontend/src/services/api.js
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function getRecommendations({ user_id = null, liked_book_ids = [], k = 20 }) {
  const payload = { user_id, liked_book_ids };
  const res = await axios.post(`${API_BASE}/recommend?k=${k}`, payload);
  return res.data;
}

export async function getBooks() {
  const res = await axios.get(`${API_BASE}/books`);
  return res.data;
}

