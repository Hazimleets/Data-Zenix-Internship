// frontend/src/services/api.js

import axios from "axios";

const API_BASE = "http://localhost:8000";

export async function getRecommendations({ user_id = null, liked_book_ids = [], k = 10 }) {
  try {
    const payload = { user_id, liked_book_ids, k };
    const res = await axios.post(`${API_BASE}/recommend`, payload);
    return res.data.recommendations;
  } catch (err) {
    console.error("Error fetching recommendations:", err);
    return [];
  }
}

