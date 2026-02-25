"use client";

import { useState } from "react";

export default function Home() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<{
    prediction: string;
    probability: number;
    is_spam: boolean;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const checkSpam = async () => {
    if (!message.trim()) {
      setError("Please enter a message to analyze.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        throw new Error("Failed to connect to the backend server.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Server connection failed. Is the Flask backend running?");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen flex flex-col items-center justify-center p-6 md:p-24 overflow-hidden selection:bg-blue-500/30">
      {/* Decorative Background Elements */}
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-blue-600/10 rounded-full blur-[120px] animate-float opacity-30" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-indigo-600/10 rounded-full blur-[120px] animate-float opacity-30" />

      <div className="z-10 w-full max-w-2xl space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-1000">
        <header className="text-center space-y-3">
          <div className="inline-flex items-center px-3 py-1 rounded-full border border-blue-400/20 bg-blue-400/5 text-blue-400 text-[10px] font-bold tracking-[0.2em] uppercase mb-2">
            AI-Powered Analysis
          </div>
          <h1 className="text-5xl md:text-6xl font-black tracking-tight text-white leading-tight">
            Spam<span className="text-blue-500">Detector</span>
          </h1>
          <p className="text-slate-400 text-base md:text-lg max-w-md mx-auto leading-relaxed font-medium">
            Instantly identify suspicious emails and messages using machine learning.
          </p>
        </header>

        <div className="glass rounded-[2rem] p-1 shadow-2xl">
          <div className="bg-[#0a0f1d]/60 rounded-[calc(2rem-4px)] p-6 md:p-8 space-y-6">
            <div className="space-y-3">
              <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-widest text-slate-500">
                <span>Message Content</span>
                <span>{message.length} characters</span>
              </div>
              <textarea
                className="w-full h-48 p-5 rounded-2xl bg-black/40 border border-white/5 text-white placeholder-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-500/50 transition-all resize-none shadow-inner"
                placeholder="Paste your email here..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
              />
            </div>

            <button
              onClick={checkSpam}
              disabled={loading}
              className={`w-full py-4 rounded-xl font-bold text-base transition-all transform active:scale-[0.99] flex items-center justify-center space-x-2 ${loading
                  ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                  : "bg-blue-600 text-white hover:bg-blue-500 shadow-lg shadow-blue-600/20"
                }`}
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <span>Check for Spam</span>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-bold text-center">
                {error}
              </div>
            )}

            {result && (
              <div className={`mt-6 p-6 rounded-2xl border animate-in zoom-in-95 duration-500 ${result.is_spam
                  ? "bg-red-500/5 border-red-500/20 shadow-[0_0_30px_-5px_rgba(239,68,68,0.2)]"
                  : "bg-emerald-500/5 border-emerald-500/20 shadow-[0_0_30px_-5px_rgba(16,185,129,0.2)]"
                }`}>
                <div className="flex flex-col items-center space-y-4">
                  <div className="text-center">
                    <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mb-1">Prediction</p>
                    <h2 className={`text-4xl font-black ${result.is_spam ? "text-red-500" : "text-emerald-500"}`}>
                      {result.prediction}
                    </h2>
                  </div>

                  <div className="w-full max-w-xs space-y-2">
                    <div className="flex justify-between items-center text-[10px] font-black opacity-50 uppercase tracking-widest">
                      <span>Certainty</span>
                      <span>{result.probability}%</span>
                    </div>
                    <div className="w-full bg-black/40 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full transition-all duration-700 ease-out ${result.is_spam ? "bg-red-500" : "bg-emerald-500"}`}
                        style={{ width: `${result.probability}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="text-center">
          <p className="text-slate-700 text-[10px] font-bold tracking-[0.3em] uppercase">
            Powered by Scikit-Learn • Multinomial NB
          </p>
        </footer>
      </div>
    </main>
  );
}
