// frontend/src/components/BookCard.jsx

import React from "react";

export default function BookCard({book, onSelect}) {
  return (
    <div
      className="bg-white shadow-md rounded-2xl p-4 hover:shadow-xl transition cursor-pointer"
      onClick={() => onSelect(book)}
    >
      <div className="h-40 flex items-center justify-center mb-3 bg-gray-100 rounded-lg">
        <span className="text-sm text-gray-400">Cover</span>
      </div>
      <h3 className="font-semibold text-lg truncate">{book.title || book.book_id}</h3>
      <p className="text-sm text-gray-500">{book.authors || "Unknown author"}</p>
      {book.score && (
        <div className="mt-2 text-xs text-indigo-600">
          Score: {Number(book.score).toFixed(3)}
        </div>
      )}
    </div>
  );
}
