// frontend/src/pages/Recommendations.jsx

import React, { useEffect, useState } from "react";
import { getRecommendations } from "../services/api";
import BookCard from "../components/BookCard";

export default function Recommendations({ liked }) {
  const [recs, setRecs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchRecs() {
      try {
        setLoading(true);
        const data = await getRecommendations({ liked_book_ids: liked, k: 12 });
        setRecs(data.recommendations || []);
      } catch (err) {
        console.error(err);
        alert("Error fetching recommendations");
      } finally {
        setLoading(false);
      }
    }
    if (liked.length > 0) {
      fetchRecs();
    }
  }, [liked]);

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Your Recommendations</h2>
      {loading ? (
        <div className="text-gray-500">Loading recommendations…</div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {recs.map((r) => (
            <BookCard
              key={r.book_id}
              book={{ ...r, title: r.book_id }}
              onSelect={() => {}}
            />
          ))}
        </div>
      )}
    </div>
  );
}
