// frontend/src/pages/Home.jsx

import React, { useState } from "react";
import BookCard from "../components/BookCard";
import SearchOnboard from "../components/SearchOnboard";
import { getRecommendations } from "../services/api";

export default function Home() {
  const [liked, setLiked] = useState([]);
  const [recs, setRecs] = useState([]);
  const [loading, setLoading] = useState(false);

  async function fetchRecs() {
    setLoading(true);
    try {
      const data = await getRecommendations({ liked_book_ids: liked, k: 12 });
      setRecs(data.recommendations || []);
    } catch (err) {
      console.error(err);
      alert("Error fetching recommendations");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-white to-indigo-100">
      <div className="max-w-7xl mx-auto px-6 py-10">
        {/* Header */}
        <header className="mb-10 text-center">
          <h1 className="text-5xl font-extrabold text-indigo-700 drop-shadow-sm">
            📚 Smart Book Recommender
          </h1>
          <p className="mt-3 text-gray-600 text-lg">
            Get personalized book recommendations by selecting your favorites
          </p>
        </header>

        {/* Onboarding */}
        <SearchOnboard liked={liked} setLiked={setLiked} />

        {/* Actions */}
        <div className="flex justify-center mb-8 gap-4">
          <button
            onClick={fetchRecs}
            disabled={loading || liked.length === 0}
            className="px-6 py-3 rounded-xl font-semibold text-white shadow-lg transition disabled:opacity-50
                       bg-indigo-600 hover:bg-indigo-700"
          >
            {loading ? "Loading…" : "✨ Get Recommendations"}
          </button>
          <div className="px-5 py-3 bg-white rounded-xl shadow-md">
            ❤️ {liked.length} books liked
          </div>
        </div>

        {/* Recommendations */}
        {recs.length > 0 && (
          <section>
            <h2 className="mb-6 text-2xl font-bold text-indigo-700 text-center">
              Your Recommendations
            </h2>
            <div className="grid gap-6 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
              {recs.map((r) => (
                <BookCard
                  key={r.book_id}
                  book={{ ...r, title: r.book_id }}
                  onSelect={() => {}}
                />
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
