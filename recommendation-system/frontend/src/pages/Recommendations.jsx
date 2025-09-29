// frontend/src/pages/Recommendations.jsx

import { useEffect, useState } from "react";
import { getRecommendations } from "../services/api.js"; // fixed path

export default function Recommendations() {
  const [books, setBooks] = useState([]);

  useEffect(() => {
    async function fetchData() {
      try {
        // You can pass user_id, liked_book_ids, or k if needed
        const data = await getRecommendations({ user_id: 1, liked_book_ids: [], k: 6 });
        setBooks(data);
      } catch (error) {
        console.error("Error fetching recommendations:", error);
      }
    }
    fetchData();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-100 p-10">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-8 text-center">
          🚀 Recommended Books
        </h1>

        {books.length === 0 ? (
          <p className="text-center text-gray-500 text-lg">
            No recommendations found. Try liking some books first!
          </p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {books.map((book, idx) => (
              <div
                key={idx}
                className="p-6 rounded-2xl shadow-xl bg-white hover:shadow-2xl transition-shadow border border-gray-100"
              >
                <h2 className="text-xl font-semibold text-gray-800 mb-2">
                  {book.title}
                </h2>
                <p className="text-gray-600 italic">{book.author}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
