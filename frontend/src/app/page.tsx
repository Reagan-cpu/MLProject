"use client";

import { useState, useEffect, useRef } from "react";

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
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState<
    Array<{ message: string; prediction: string; time: string; is_spam: boolean }>
  >([]);
  const [focused, setFocused] = useState(false);
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
        body: JSON.stringify({ message }),
      });
      if (!response.ok) throw new Error("Server connection failed");
      const data = await response.json();
      setResult(data);
      setTimeout(() => setAnimateResult(true), 50);
      setHistory((prev) => [
        { message: message.substring(0, 60), prediction: data.prediction, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), is_spam: data.is_spam },
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
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg: #0a0a0a;
          --surface: #111111;
          --surface2: #181818;
          --border: #222222;
          --border2: #2a2a2a;
          --text: #f0f0f0;
          --muted: #555555;
          --muted2: #777777;
          --spam: #ff3b3b;
          --spam-bg: rgba(255,59,59,0.07);
          --spam-border: rgba(255,59,59,0.2);
          --safe: #00d68f;
          --safe-bg: rgba(0,214,143,0.07);
          --safe-border: rgba(0,214,143,0.2);
          --accent: #c8ff57;
          --font-head: 'Syne', sans-serif;
          --font-mono: 'DM Mono', monospace;
        }

        body { background: var(--bg); color: var(--text); font-family: var(--font-head); }

        .app {
          min-height: 100vh;
          display: grid;
          grid-template-rows: auto 1fr auto;
        }

        /* NAV */
        nav {
          border-bottom: 1px solid var(--border);
          padding: 20px 40px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          position: sticky;
          top: 0;
          background: var(--bg);
          z-index: 50;
        }
        .nav-logo {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 15px;
          font-weight: 700;
          letter-spacing: -0.3px;
        }
        .logo-mark {
          width: 28px;
          height: 28px;
          background: var(--accent);
          border-radius: 6px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 13px;
        }
        .nav-tag {
          font-family: var(--font-mono);
          font-size: 10px;
          color: var(--muted2);
          border: 1px solid var(--border2);
          padding: 3px 8px;
          border-radius: 100px;
          letter-spacing: 0.05em;
        }

        /* MAIN LAYOUT */
        .main {
          max-width: 1100px;
          width: 100%;
          margin: 0 auto;
          padding: 60px 40px;
          display: grid;
          grid-template-columns: 1fr 340px;
          gap: 32px;
          align-items: start;
        }

        /* LEFT */
        .left { display: flex; flex-direction: column; gap: 28px; }

        .eyebrow {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-family: var(--font-mono);
          font-size: 10px;
          letter-spacing: 0.15em;
          text-transform: uppercase;
          color: var(--muted2);
        }
        .eyebrow-dot {
          width: 5px; height: 5px;
          background: var(--accent);
          border-radius: 50%;
          animation: pulse 2s ease-in-out infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        h1 {
          font-size: clamp(40px, 5vw, 64px);
          font-weight: 800;
          line-height: 1.0;
          letter-spacing: -2px;
          color: var(--text);
        }
        h1 em {
          font-style: normal;
          color: var(--accent);
        }

        .subtitle {
          font-family: var(--font-mono);
          font-size: 13px;
          color: var(--muted2);
          line-height: 1.7;
          max-width: 420px;
          font-weight: 300;
        }

        /* INPUT CARD */
        .input-card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 16px;
          overflow: hidden;
          transition: border-color 0.2s;
        }
        .input-card.focused { border-color: var(--border2); }

        .card-header {
          padding: 16px 20px;
          border-bottom: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: var(--surface2);
        }
        .card-header-left {
          display: flex;
          align-items: center;
          gap: 8px;
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--muted2);
          letter-spacing: 0.05em;
        }
        .dot { width: 7px; height: 7px; border-radius: 50%; }
        .dot-red { background: #ff5f57; }
        .dot-yellow { background: #febc2e; }
        .dot-green { background: #28c840; }

        .char-count {
          font-family: var(--font-mono);
          font-size: 10px;
          color: var(--muted);
        }

        textarea {
          width: 100%;
          min-height: 160px;
          padding: 20px;
          background: var(--surface);
          color: var(--text);
          font-family: var(--font-mono);
          font-size: 13px;
          line-height: 1.8;
          border: none;
          outline: none;
          resize: none;
          font-weight: 300;
        }
        textarea::placeholder { color: var(--muted); }

        .card-footer {
          padding: 12px 20px;
          border-top: 1px solid var(--border);
          background: var(--surface2);
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
        }

        .samples-row {
          display: flex;
          gap: 6px;
          flex-wrap: wrap;
        }
        .sample-btn {
          font-family: var(--font-mono);
          font-size: 10px;
          padding: 4px 10px;
          border-radius: 100px;
          border: 1px solid var(--border2);
          background: transparent;
          color: var(--muted2);
          cursor: pointer;
          transition: all 0.15s;
          letter-spacing: 0.02em;
        }
        .sample-btn:hover { border-color: var(--muted); color: var(--text); }

        /* ANALYZE BUTTON */
        .analyze-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 20px;
          border-radius: 8px;
          border: none;
          cursor: pointer;
          font-family: var(--font-head);
          font-size: 13px;
          font-weight: 700;
          letter-spacing: 0.02em;
          transition: all 0.2s;
          flex-shrink: 0;
          background: var(--accent);
          color: #0a0a0a;
        }
        .analyze-btn:disabled {
          background: var(--surface2);
          color: var(--muted);
          cursor: not-allowed;
        }
        .analyze-btn:not(:disabled):hover { opacity: 0.85; transform: translateY(-1px); }

        .spinner {
          width: 13px; height: 13px;
          border: 2px solid rgba(0,0,0,0.3);
          border-top-color: #000;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ERROR */
        .error-bar {
          padding: 12px 16px;
          border-radius: 10px;
          background: rgba(255,59,59,0.08);
          border: 1px solid rgba(255,59,59,0.2);
          font-family: var(--font-mono);
          font-size: 12px;
          color: #ff7070;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        /* RESULT */
        .result-card {
          border-radius: 14px;
          padding: 28px;
          opacity: 0;
          transform: translateY(8px);
          transition: opacity 0.4s ease, transform 0.4s ease;
        }
        .result-card.visible { opacity: 1; transform: translateY(0); }
        .result-spam { background: var(--spam-bg); border: 1px solid var(--spam-border); }
        .result-safe { background: var(--safe-bg); border: 1px solid var(--safe-border); }

        .result-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 20px;
        }

        .verdict {
          display: flex;
          align-items: baseline;
          gap: 10px;
        }
        .verdict-label {
          font-size: 11px;
          font-family: var(--font-mono);
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: var(--muted2);
        }
        .verdict-text {
          font-size: 36px;
          font-weight: 800;
          letter-spacing: -1.5px;
          line-height: 1;
        }
        .verdict-spam { color: var(--spam); }
        .verdict-safe { color: var(--safe); }

        .prob-pill {
          font-family: var(--font-mono);
          font-size: 13px;
          font-weight: 500;
          padding: 6px 14px;
          border-radius: 100px;
        }
        .pill-spam { background: var(--spam-bg); border: 1px solid var(--spam-border); color: var(--spam); }
        .pill-safe { background: var(--safe-bg); border: 1px solid var(--safe-border); color: var(--safe); }

        .progress-track {
          width: 100%;
          height: 4px;
          background: var(--border);
          border-radius: 100px;
          overflow: hidden;
          margin-bottom: 16px;
        }
        .progress-fill {
          height: 100%;
          border-radius: 100px;
          transition: width 1s cubic-bezier(0.16,1,0.3,1);
        }
        .fill-spam { background: var(--spam); }
        .fill-safe { background: var(--safe); }

        .result-note {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--muted2);
          line-height: 1.6;
        }

        /* RIGHT SIDEBAR */
        .right { display: flex; flex-direction: column; gap: 16px; position: sticky; top: 88px; }

        .sidebar-card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 14px;
          overflow: hidden;
        }
        .sidebar-title {
          padding: 14px 18px;
          border-bottom: 1px solid var(--border);
          font-size: 11px;
          font-family: var(--font-mono);
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: var(--muted2);
          background: var(--surface2);
        }

        /* STEPS */
        .steps { padding: 6px 0; }
        .step {
          display: flex;
          align-items: flex-start;
          gap: 0;
          padding: 14px 18px;
          position: relative;
        }
        .step:not(:last-child)::after {
          content: '';
          position: absolute;
          left: 27px;
          top: 40px;
          bottom: 0;
          width: 1px;
          background: var(--border);
        }
        .step-num {
          width: 18px;
          height: 18px;
          border-radius: 50%;
          border: 1px solid var(--border2);
          background: var(--surface2);
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: var(--font-mono);
          font-size: 9px;
          color: var(--muted2);
          flex-shrink: 0;
          margin-right: 12px;
          margin-top: 1px;
          position: relative;
          z-index: 1;
        }
        .step-body { flex: 1; }
        .step-title { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 3px; }
        .step-desc { font-family: var(--font-mono); font-size: 11px; color: var(--muted); font-weight: 300; line-height: 1.5; }

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
          font-size: 22px;
          font-weight: 800;
          letter-spacing: -1px;
          color: var(--accent);
        }
        .stat-label {
          font-family: var(--font-mono);
          font-size: 10px;
          color: var(--muted);
          margin-top: 2px;
          letter-spacing: 0.04em;
        }

        /* HISTORY */
        .history-list { padding: 8px 0; max-height: 260px; overflow-y: auto; }
        .history-list::-webkit-scrollbar { width: 3px; }
        .history-list::-webkit-scrollbar-track { background: transparent; }
        .history-list::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

        .history-item {
          padding: 12px 18px;
          display: flex;
          align-items: flex-start;
          gap: 10px;
          border-bottom: 1px solid var(--border);
          transition: background 0.15s;
        }
        .history-item:last-child { border-bottom: none; }
        .history-item:hover { background: var(--surface2); }
        .h-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
        .h-dot-spam { background: var(--spam); }
        .h-dot-safe { background: var(--safe); }
        .h-msg { font-family: var(--font-mono); font-size: 11px; color: var(--muted2); line-height: 1.5; flex: 1; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; font-weight: 300; }
        .h-meta { font-family: var(--font-mono); font-size: 10px; color: var(--muted); flex-shrink: 0; text-align: right; }
        .h-verdict { font-size: 10px; font-weight: 600; letter-spacing: 0.05em; }
        .h-verdict-spam { color: var(--spam); }
        .h-verdict-safe { color: var(--safe); }

        /* FOOTER */
        footer {
          border-top: 1px solid var(--border);
          padding: 20px 40px;
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        .footer-text {
          font-family: var(--font-mono);
          font-size: 10px;
          color: var(--muted);
          letter-spacing: 0.05em;
        }

        @media (max-width: 768px) {
          nav { padding: 16px 20px; }
          .main { grid-template-columns: 1fr; padding: 32px 20px; }
          .right { position: static; }
          footer { padding: 16px 20px; }
        }
      `}</style>

      <div className="app">
        {/* NAV */}
        <nav>
          <div className="nav-logo">
            <div className="logo-mark">⚡</div>
            SpamDetect
          </div>
          <span className="nav-tag">v2.0 · ML-Powered</span>
        </nav>

        {/* MAIN */}
        <div className="main">
          {/* LEFT COLUMN */}
          <div className="left">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-dot" />
                Real-time analysis
              </div>
            </div>

            <h1>
              Is it spam<br />or <em>legit?</em>
            </h1>

            <p className="subtitle">
              Paste any message below. Our ML model classifies it instantly with a confidence score — no guessing.
            </p>

            {/* INPUT CARD */}
            <div className={`input-card ${focused ? "focused" : ""}`}>
              <div className="card-header">
                <div className="card-header-left">
                  <div className="dot dot-red" />
                  <div className="dot dot-yellow" />
                  <div className="dot dot-green" />
                  <span style={{ marginLeft: 8 }}>message.txt</span>
                </div>
                <span className="char-count">{message.length} / 500</span>
              </div>

              <textarea
                ref={textareaRef}
                placeholder="Paste your email, SMS, or any message here..."
                value={message}
                onChange={(e) => { setMessage(e.target.value.substring(0, 500)); setResult(null); }}
                onFocus={() => setFocused(true)}
                onBlur={() => setFocused(false)}
              />

              <div className="card-footer">
                <div className="samples-row">
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--muted)", alignSelf: "center", marginRight: 2 }}>Try:</span>
                  {["Spam", "Ham", "Phish"].map((label, i) => (
                    <button key={i} className="sample-btn" onClick={() => handleSample(SAMPLE_MESSAGES[i])}>
                      {label}
                    </button>
                  ))}
                </div>

                <button
                  className="analyze-btn"
                  onClick={checkSpam}
                  disabled={loading || !message.trim()}
                >
                  {loading ? (
                    <><div className="spinner" /> Analyzing…</>
                  ) : (
                    <>Analyze</>
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
              <div className={`result-card ${result.is_spam ? "result-spam" : "result-safe"} ${animateResult ? "visible" : ""}`}>
                <div className="result-top">
                  <div className="verdict">
                    <div>
                      <div className="verdict-label">Verdict</div>
                      <div className={`verdict-text ${result.is_spam ? "verdict-spam" : "verdict-safe"}`}>
                        {result.prediction}
                      </div>
                    </div>
                  </div>
                  <div className={`prob-pill ${result.is_spam ? "pill-spam" : "pill-safe"}`}>
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
                    ? "This message exhibits patterns commonly associated with spam or phishing. Exercise caution before clicking any links or sharing personal information."
                    : "This message does not exhibit typical spam characteristics. It appears to be a legitimate message."}
                </p>
              </div>
            )}
          </div>

          {/* RIGHT SIDEBAR */}
          <div className="right">
            {/* STEPS */}
            <div className="sidebar-card">
              <div className="sidebar-title">How it works</div>
              <div className="steps">
                {[
                  { n: "01", title: "Paste message", desc: "Drop any email, SMS, or text into the editor above." },
                  { n: "02", title: "ML processes it", desc: "Naive Bayes + TF-IDF vectorizer scores every token." },
                  { n: "03", title: "Verdict returned", desc: "Get spam/ham result with a calibrated confidence score." },
                ].map((s) => (
                  <div className="step" key={s.n}>
                    <div className="step-num">{s.n}</div>
                    <div className="step-body">
                      <div className="step-title">{s.title}</div>
                      <div className="step-desc">{s.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* STATS */}
            <div className="sidebar-card">
              <div className="sidebar-title">Model stats</div>
              <div className="stats-grid">
                {[
                  { v: "98.6%", l: "Accuracy" },
                  { v: "<50ms", l: "Latency" },
                  { v: "5.5k", l: "Training msgs" },
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
                <div className="sidebar-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span>Recent checks</span>
                  <button
                    onClick={() => setHistory([])}
                    style={{ background: "none", border: "none", color: "var(--muted)", cursor: "pointer", fontSize: 10, fontFamily: "var(--font-mono)", letterSpacing: "0.05em" }}
                  >
                    clear
                  </button>
                </div>
                <div className="history-list">
                  {history.map((item, i) => (
                    <div className="history-item" key={i}>
                      <div className={`h-dot ${item.is_spam ? "h-dot-spam" : "h-dot-safe"}`} />
                      <div className="h-msg">{item.message}{item.message.length >= 60 ? "…" : ""}</div>
                      <div className="h-meta">
                        <div className={`h-verdict ${item.is_spam ? "h-verdict-spam" : "h-verdict-safe"}`}>{item.prediction}</div>
                        <div>{item.time}</div>
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