// recommendation-system/frontend/src/components/SearchOnboard.jsx

import React, { useState, useEffect } from "react";
import axios from "axios";

export default function SearchOnboard({ liked, setLiked }) {
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState([]);

  const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

  // Fetch books from backend
  async function fetchBooks(query = "") {
    try {
      const res = await axios.get(`${API_BASE}/books`, { params: { q: query } });
      setSuggestions(res.data || []);
    } catch (err) {
      console.error("Error fetching books:", err);
    }
  }

  useEffect(() => {
    fetchBooks(""); // initial load
  }, []);

  function handleInputChange(e) {
    const value = e.target.value;
    setInput(value);
    fetchBooks(value);
  }

  function toggleLike(bookId) {
    if (liked.includes(bookId)) {
      setLiked(liked.filter((id) => id !== bookId));
    } else {
      setLiked([...liked, bookId]);
    }
  }

  return (
    <div className="bg-white p-6 rounded-2xl shadow mb-6">
      <h2 className="text-lg font-semibold mb-3">
        Select a few books you like to get started
      </h2>
      <input
        type="text"
        className="w-full border border-gray-300 rounded-lg p-2 mb-3 focus:ring focus:ring-indigo-200"
        placeholder="Search for a book..."
        value={input}
        onChange={handleInputChange}
      />
      {suggestions.length > 0 && (
        <ul className="border rounded-lg divide-y max-h-60 overflow-y-auto">
          {suggestions.map((s) => (
            <li
              key={s.book_id}
              className={`p-2 cursor-pointer hover:bg-gray-100 ${
                liked.includes(s.book_id) ? "bg-indigo-100" : ""
              }`}
              onClick={() => toggleLike(s.book_id)}
            >
              {s.title} — {s.author}
            </li>
          ))}
        </ul>
      )}
      {liked.length > 0 && (
        <div className="mt-3 text-sm text-gray-600">
          Selected: {liked.join(", ")}
        </div>
      )}
    </div>
  );
}
