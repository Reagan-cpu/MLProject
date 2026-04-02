"use client";

import { useState, useRef } from "react";

const SAMPLE_MESSAGES = [
  "Congratulations! You've won a $1000 gift card. Click here now!",
  "Hey, are we still on for lunch tomorrow at 1pm?",
  "URGENT: Your account will be suspended. Verify immediately!",
  "The quarterly report is attached. Let me know your thoughts.",
];

export default function Home() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<{
    prediction: string;
    probability: number;
    is_spam: boolean;
    model_used?: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedModel, setSelectedModel] = useState("tfidf");
  const [history, setHistory] = useState<
    Array<{ message: string; prediction: string; time: string; is_spam: boolean; model: string }>
  >([]);
  const [animateResult, setAnimateResult] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const checkSpam = async () => {
    if (!message.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    setAnimateResult(false);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, model: selectedModel }),
      });
      if (!response.ok) throw new Error("Server connection failed");
      const data = await response.json();
      setResult(data);
      setTimeout(() => setAnimateResult(true), 50);
      setHistory((prev) => [
        {
          message: message.substring(0, 60),
          prediction: data.prediction,
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          is_spam: data.is_spam,
          model: selectedModel.toUpperCase(),
        },
        ...prev.slice(0, 4),
      ]);
    } catch {
      setError("Backend offline — is Flask running on port 5000?");
    } finally {
      setLoading(false);
    }
  };

  const handleSample = (s: string) => {
    setMessage(s);
    setResult(null);
    textareaRef.current?.focus();
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,300;1,9..144,400&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg: #f7f5f0;
          --surface: #ffffff;
          --surface2: #f2efe9;
          --border: #e4dfd6;
          --border2: #d4cfc5;
          --text: #1a1714;
          --text2: #4a4540;
          --muted: #8c8680;
          --muted2: #b5b0aa;
          --spam: #c0392b;
          --spam-light: #fdf2f1;
          --spam-border: #f5c6c2;
          --safe: #1a6b45;
          --safe-light: #f0f8f4;
          --safe-border: #b8dfc9;
          --accent: #2d5be3;
          --font-head: 'Fraunces', Georgia, serif;
          --font-body: 'DM Sans', system-ui, sans-serif;
        }

        body {
          background: var(--bg);
          color: var(--text);
          font-family: var(--font-body);
          -webkit-font-smoothing: antialiased;
        }

        .app {
          min-height: 100vh;
          display: grid;
          grid-template-rows: auto 1fr auto;
        }

        /* NAV */
        nav {
          background: var(--surface);
          border-bottom: 1px solid var(--border);
          padding: 0 48px;
          height: 60px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          position: sticky;
          top: 0;
          z-index: 50;
        }

        .nav-logo {
          display: flex;
          align-items: center;
          gap: 10px;
          font-family: var(--font-head);
          font-size: 18px;
          font-weight: 600;
          color: var(--text);
          letter-spacing: -0.3px;
        }

        .logo-icon {
          width: 30px;
          height: 30px;
          background: var(--text);
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 15px;
        }

        .nav-badge {
          font-family: var(--font-body);
          font-size: 11px;
          font-weight: 500;
          color: var(--muted);
          background: var(--surface2);
          border: 1px solid var(--border);
          padding: 3px 10px;
          border-radius: 20px;
          letter-spacing: 0.02em;
        }

        /* MAIN */
        .main {
          max-width: 1060px;
          width: 100%;
          margin: 0 auto;
          padding: 52px 40px;
          display: grid;
          grid-template-columns: 1fr 320px;
          gap: 36px;
          align-items: start;
        }

        /* LEFT */
        .left { display: flex; flex-direction: column; gap: 24px; }

        .hero { margin-bottom: 4px; }

        .hero-label {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          font-size: 12px;
          font-weight: 500;
          color: var(--muted);
          letter-spacing: 0.06em;
          text-transform: uppercase;
          margin-bottom: 14px;
        }

        .live-dot {
          width: 6px; height: 6px;
          background: #22c55e;
          border-radius: 50%;
          box-shadow: 0 0 0 0 rgba(34,197,94,0.4);
          animation: ping 2s ease-in-out infinite;
        }

        @keyframes ping {
          0%, 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
          50% { box-shadow: 0 0 0 5px rgba(34,197,94,0); }
        }

        h1 {
          font-family: var(--font-head);
          font-size: clamp(38px, 4.5vw, 58px);
          font-weight: 600;
          line-height: 1.05;
          letter-spacing: -1.5px;
          color: var(--text);
          margin-bottom: 14px;
        }

        h1 em {
          font-style: italic;
          font-weight: 300;
          color: var(--muted);
        }

        .subtitle {
          font-size: 15px;
          font-weight: 300;
          color: var(--text2);
          line-height: 1.65;
          max-width: 400px;
        }

        /* INPUT CARD */
        .input-card {
          background: var(--surface);
          border: 1.5px solid var(--border);
          border-radius: 14px;
          overflow: hidden;
          transition: border-color 0.2s, box-shadow 0.2s;
        }

        .input-card:focus-within {
          border-color: var(--border2);
          box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        }

        .card-top-bar {
          padding: 12px 18px;
          background: var(--surface2);
          border-bottom: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .traffic-lights {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .tl { width: 10px; height: 10px; border-radius: 50%; }
        .tl-r { background: #ff5f57; }
        .tl-y { background: #febc2e; }
        .tl-g { background: #28c840; }

        .file-label {
          font-size: 12px;
          font-weight: 400;
          color: var(--muted);
        }

        .char-count {
          font-size: 11px;
          color: var(--muted2);
        }

        textarea {
          width: 100%;
          min-height: 148px;
          padding: 18px 20px;
          background: var(--surface);
          color: var(--text);
          font-family: var(--font-body);
          font-size: 14px;
          line-height: 1.75;
          font-weight: 300;
          border: none;
          outline: none;
          resize: none;
        }

        textarea::placeholder { color: var(--muted2); }

        .card-bottom-bar {
          padding: 12px 16px;
          background: var(--surface2);
          border-top: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
        }

        .sample-group {
          display: flex;
          align-items: center;
          gap: 6px;
          flex-wrap: wrap;
        }

        .sample-label {
          font-size: 11px;
          color: var(--muted2);
          font-weight: 400;
        }

        .sample-btn {
          font-family: var(--font-body);
          font-size: 11px;
          font-weight: 500;
          padding: 4px 11px;
          border-radius: 20px;
          border: 1px solid var(--border2);
          background: var(--surface);
          color: var(--text2);
          cursor: pointer;
          transition: all 0.15s;
        }

        .sample-btn:hover {
          background: var(--text);
          color: var(--bg);
          border-color: var(--text);
        }

        /* MODEL SELECTOR */
        .model-selector {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .model-label {
          font-size: 11px;
          color: var(--muted2);
          font-weight: 500;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .model-select {
          padding: 6px 10px;
          border-radius: 6px;
          border: 1px solid var(--border2);
          background: var(--surface);
          color: var(--text);
          font-family: var(--font-body);
          font-size: 12px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s;
        }

        .model-select:hover {
          border-color: var(--border);
          background: var(--surface2);
        }

        .model-select:focus {
          outline: none;
          border-color: var(--accent);
          box-shadow: 0 0 0 2px rgba(45, 91, 227, 0.1);
        }
        .analyze-btn {
          display: flex;
          align-items: center;
          gap: 7px;
          padding: 9px 20px;
          border-radius: 8px;
          border: none;
          cursor: pointer;
          font-family: var(--font-body);
          font-size: 13px;
          font-weight: 600;
          letter-spacing: 0.01em;
          transition: all 0.18s;
          flex-shrink: 0;
          background: var(--text);
          color: var(--bg);
        }

        .analyze-btn:disabled {
          background: var(--border2);
          color: var(--muted);
          cursor: not-allowed;
        }

        .analyze-btn:not(:disabled):hover {
          background: var(--text2);
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .spinner {
          width: 12px; height: 12px;
          border: 2px solid rgba(255,255,255,0.3);
          border-top-color: #fff;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* ERROR */
        .error-bar {
          padding: 12px 16px;
          border-radius: 10px;
          background: var(--spam-light);
          border: 1px solid var(--spam-border);
          font-size: 13px;
          color: var(--spam);
          display: flex;
          align-items: center;
          gap: 8px;
        }

        /* RESULT */
        .result-card {
          border-radius: 14px;
          padding: 24px 28px;
          opacity: 0;
          transform: translateY(6px);
          transition: opacity 0.35s ease, transform 0.35s ease;
        }

        .result-card.visible { opacity: 1; transform: translateY(0); }
        .result-spam { background: var(--spam-light); border: 1.5px solid var(--spam-border); }
        .result-safe { background: var(--safe-light); border: 1.5px solid var(--safe-border); }

        .result-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          margin-bottom: 18px;
        }

        .verdict-label {
          font-size: 11px;
          font-weight: 500;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--muted);
          margin-bottom: 6px;
        }

        .verdict-text {
          font-family: var(--font-head);
          font-size: 40px;
          font-weight: 600;
          letter-spacing: -1.5px;
          line-height: 1;
        }

        .verdict-spam { color: var(--spam); }
        .verdict-safe { color: var(--safe); }

        .confidence-badge {
          font-size: 12px;
          font-weight: 600;
          padding: 6px 14px;
          border-radius: 20px;
          flex-shrink: 0;
          margin-top: 4px;
        }

        .badge-spam {
          background: rgba(192,57,43,0.1);
          color: var(--spam);
          border: 1px solid var(--spam-border);
        }

        .badge-safe {
          background: rgba(26,107,69,0.1);
          color: var(--safe);
          border: 1px solid var(--safe-border);
        }

        .progress-track {
          height: 3px;
          background: var(--border);
          border-radius: 3px;
          overflow: hidden;
          margin-bottom: 14px;
        }

        .progress-fill {
          height: 100%;
          border-radius: 3px;
          transition: width 1.1s cubic-bezier(0.16,1,0.3,1);
        }

        .fill-spam { background: var(--spam); }
        .fill-safe { background: var(--safe); }

        .result-note {
          font-size: 13px;
          font-weight: 300;
          color: var(--text2);
          line-height: 1.65;
        }

        /* RIGHT SIDEBAR */
        .right {
          display: flex;
          flex-direction: column;
          gap: 16px;
          position: sticky;
          top: 76px;
        }

        .sidebar-card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 14px;
          overflow: hidden;
        }

        .sidebar-title {
          padding: 12px 18px;
          border-bottom: 1px solid var(--border);
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--muted);
          background: var(--surface2);
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        /* STEPS */
        .steps { padding: 4px 0; }

        .step {
          display: flex;
          align-items: flex-start;
          gap: 12px;
          padding: 14px 18px;
          position: relative;
        }

        .step:not(:last-child)::after {
          content: '';
          position: absolute;
          left: 27px;
          top: 42px;
          bottom: 0;
          width: 1px;
          background: var(--border);
        }

        .step-num {
          width: 18px; height: 18px;
          border-radius: 50%;
          border: 1.5px solid var(--border2);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 9px;
          font-weight: 600;
          color: var(--muted);
          flex-shrink: 0;
          margin-top: 1px;
          position: relative;
          z-index: 1;
          background: var(--surface);
        }

        .step-title { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 3px; }
        .step-desc { font-size: 12px; font-weight: 300; color: var(--muted); line-height: 1.55; }

        /* STATS */
        .stats-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1px;
          background: var(--border);
        }

        .stat {
          background: var(--surface);
          padding: 16px;
          text-align: center;
        }

        .stat-value {
          font-family: var(--font-head);
          font-size: 22px;
          font-weight: 600;
          letter-spacing: -0.5px;
          color: var(--text);
        }

        .stat-label {
          font-size: 11px;
          font-weight: 400;
          color: var(--muted);
          margin-top: 2px;
        }

        /* HISTORY */
        .history-list {
          max-height: 240px;
          overflow-y: auto;
        }

        .history-list::-webkit-scrollbar { width: 3px; }
        .history-list::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

        .history-item {
          padding: 12px 18px;
          display: flex;
          align-items: flex-start;
          gap: 10px;
          border-bottom: 1px solid var(--border);
          transition: background 0.12s;
        }

        .history-item:last-child { border-bottom: none; }
        .history-item:hover { background: var(--surface2); }

        .h-dot {
          width: 6px; height: 6px;
          border-radius: 50%;
          flex-shrink: 0;
          margin-top: 5px;
        }

        .h-dot-spam { background: var(--spam); }
        .h-dot-safe { background: var(--safe); }

        .h-msg {
          font-size: 12px;
          font-weight: 300;
          color: var(--text2);
          line-height: 1.5;
          flex: 1;
          overflow: hidden;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
        }

        .h-meta { flex-shrink: 0; text-align: right; }

        .h-verdict {
          font-size: 10px;
          font-weight: 600;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .h-verdict-spam { color: var(--spam); }
        .h-verdict-safe { color: var(--safe); }

        .h-time { font-size: 10px; color: var(--muted2); margin-top: 2px; }

        .h-model { font-size: 9px; color: var(--muted); margin-top: 1px; font-weight: 500; }

        .clear-btn {
          background: none;
          border: none;
          color: var(--muted2);
          cursor: pointer;
          font-size: 11px;
          font-family: var(--font-body);
          font-weight: 400;
          transition: color 0.15s;
        }

        .clear-btn:hover { color: var(--spam); }

        /* FOOTER */
        footer {
          border-top: 1px solid var(--border);
          padding: 18px 48px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: var(--surface);
        }

        .footer-text {
          font-size: 11px;
          color: var(--muted2);
          font-weight: 400;
        }

        @media (max-width: 768px) {
          nav { padding: 0 20px; }
          .main { grid-template-columns: 1fr; padding: 28px 20px; }
          .right { position: static; }
          footer { padding: 16px 20px; }
        }
      `}</style>

      <div className="app">
        {/* NAV */}
        <nav>
          <div className="nav-logo">
            <div className="logo-icon">🛡</div>
            SpamDetect
          </div>
          <span className="nav-badge">ML-Powered · v2.0</span>
        </nav>

        {/* MAIN */}
        <div className="main">
          {/* LEFT */}
          <div className="left">
            <div className="hero">
              <div className="hero-label">
                <span className="live-dot" />
                Real-time analysis
              </div>
              <h1>
                Is it spam<br />
                <em>or legit?</em>
              </h1>
              <p className="subtitle">
                Paste any message below. Our ML model classifies it instantly with a confidence score.
              </p>
            </div>

            {/* INPUT CARD */}
            <div className="input-card">
              <div className="card-top-bar">
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div className="traffic-lights">
                    <div className="tl tl-r" />
                    <div className="tl tl-y" />
                    <div className="tl tl-g" />
                  </div>
                  <span className="file-label">Paste your message</span>
                </div>
                <span className="char-count">{message.length} / 500</span>
              </div>

              <textarea
                ref={textareaRef}
                placeholder="Drop any email, SMS, or text here..."
                value={message}
                onChange={(e) => {
                  setMessage(e.target.value.substring(0, 500));
                  setResult(null);
                }}
              />

              <div className="card-bottom-bar">
                <div style={{ display: "flex", alignItems: "center", gap: 24, flex: 1 }}>
                  <div className="sample-group">
                    <span className="sample-label">Try:</span>
                    {["Spam", "Ham", "Phishing"].map((label, i) => (
                      <button
                        key={i}
                        className="sample-btn"
                        onClick={() => handleSample(SAMPLE_MESSAGES[i])}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  <div className="model-selector">
                    <span className="model-label">Model:</span>
                    <select
                      className="model-select"
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                    >
                      <option value="tfidf">TF-IDF (Fast)</option>
                      <option value="bert">BERT (Accurate)</option>
                    </select>
                  </div>
                </div>

                <button
                  className="analyze-btn"
                  onClick={checkSpam}
                  disabled={loading || !message.trim()}
                >
                  {loading ? (
                    <><div className="spinner" /> Analyzing…</>
                  ) : (
                    <>Analyze &rarr;</>
                  )}
                </button>
              </div>
            </div>

            {/* ERROR */}
            {error && (
              <div className="error-bar">
                <span>⚠</span> {error}
              </div>
            )}

            {/* RESULT */}
            {result && (
              <div
                className={`result-card ${result.is_spam ? "result-spam" : "result-safe"} ${animateResult ? "visible" : ""}`}
              >
                <div className="result-header">
                  <div>
                    <div className="verdict-label">Verdict</div>
                    <div className={`verdict-text ${result.is_spam ? "verdict-spam" : "verdict-safe"}`}>
                      {result.prediction}
                    </div>
                  </div>
                  <div className={`confidence-badge ${result.is_spam ? "badge-spam" : "badge-safe"}`}>
                    {result.probability}% confident
                  </div>
                </div>

                <div className="progress-track">
                  <div
                    className={`progress-fill ${result.is_spam ? "fill-spam" : "fill-safe"}`}
                    style={{ width: animateResult ? `${result.probability}%` : "0%" }}
                  />
                </div>

                <p className="result-note">
                  {result.is_spam
                    ? "This message exhibits patterns associated with spam or phishing. Exercise caution before clicking links or sharing personal information."
                    : "This message does not exhibit typical spam characteristics and appears to be legitimate."}
                  <br />
                  <span style={{ marginTop: "8px", display: "block", fontSize: "11px", color: "var(--muted)", fontWeight: 500 }}>
                    Analyzed with {result.model_used === "bert" ? "BERT" : "TF-IDF"} model
                  </span>
                </p>
              </div>
            )}
          </div>

          {/* RIGHT SIDEBAR */}
          <div className="right">
            {/* HOW IT WORKS */}
            <div className="sidebar-card">
              <div className="sidebar-title">How it works</div>
              <div className="steps">
                {[
                  { n: "01", title: "Paste message", desc: "Drop any email, SMS, or text into the editor." },
                  { n: "02", title: "ML processes it", desc: "Naive Bayes + TF-IDF vectorizer scores every token." },
                  { n: "03", title: "Verdict returned", desc: "Get spam/ham result with a calibrated confidence score." },
                ].map((s) => (
                  <div className="step" key={s.n}>
                    <div className="step-num">{s.n}</div>
                    <div>
                      <div className="step-title">{s.title}</div>
                      <div className="step-desc">{s.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* MODEL STATS */}
            <div className="sidebar-card">
              <div className="sidebar-title">Model stats</div>
              <div className="stats-grid">
                {[
                  { v: "98.6%", l: "Accuracy" },
                  { v: "<50ms", l: "Latency" },
                  { v: "5,500", l: "Training msgs" },
                  { v: "2-class", l: "Output" },
                ].map((s) => (
                  <div className="stat" key={s.l}>
                    <div className="stat-value">{s.v}</div>
                    <div className="stat-label">{s.l}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* HISTORY */}
            {history.length > 0 && (
              <div className="sidebar-card">
                <div className="sidebar-title">
                  <span>Recent checks</span>
                  <button className="clear-btn" onClick={() => setHistory([])}>
                    Clear all
                  </button>
                </div>
                <div className="history-list">
                  {history.map((item, i) => (
                    <div className="history-item" key={i}>
                      <div className={`h-dot ${item.is_spam ? "h-dot-spam" : "h-dot-safe"}`} />
                      <div className="h-msg">
                        {item.message}{item.message.length >= 60 ? "…" : ""}
                      </div>
                      <div className="h-meta">
                        <div className={`h-verdict ${item.is_spam ? "h-verdict-spam" : "h-verdict-safe"}`}>
                          {item.prediction}
                        </div>
                        <div className="h-time">{item.time}</div>
                        <div className="h-model">{item.model}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* FOOTER */}
        <footer>
          <span className="footer-text">SpamDetect © 2025 · Flask backend required on :5000</span>
          <span className="footer-text">Built with ML + Next.js</span>
        </footer>
      </div>
    </>
  );
}