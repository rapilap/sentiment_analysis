import { useState } from "react";

function Form() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await fetch(import.meta.env.VITE_API_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    setResult(data.prediction);
  };

  return (
    <div>
      <h2 className="text-5xl">Sentiment Analysis</h2>
      <div className="flex flex-col bg-red-500 h-96 justify-center items-center">
        <form
          action=""
          onSubmit={handleSubmit}
          className="flex flex-col bg-blue-400 w-full items-center gap-5"
        >
          <input
            type="text"
            name="text"
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Insert sentece here..."
            className="w-1/2 text-xl p-3 rounded-md"
          />
          <button
            type="submit"
            className="bg-red-600 w-1/2 p-3 text-xl rounded-md hover:bg-red-500"
          >
            Analyze
          </button>
        </form>
      </div>
      {result && (
        <p>
          Prediction: <strong>{result}</strong>
        </p>
      )}
    </div>
  );
}

export default Form;
