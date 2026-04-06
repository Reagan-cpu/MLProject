"use client";

import { useState, useRef } from "react";

/* ───────── types ───────── */
interface ExplanationFeature {
  word: string;
  weight: number;
  direction: "spam" | "ham";
}

interface PredictResult {
  prediction: string;
  probability: number;
  is_spam: boolean;
  model_used: string;
  explanation: ExplanationFeature[];
  top_spam_words: string[];
  top_ham_words: string[];
}

interface FidelityResult {
  original_spam_prob: number;
  reduced_spam_prob: number;
  probability_drop: number;
  prediction_flipped: boolean;
  removed_words: string[];
  fidelity_score: number;
}

interface ExplainResult {
  message: string;
  model_used: string;
  prediction: string;
  probability: number;
  is_spam: boolean;
  explanations: {
    lime: {
      method: string;
      features: ExplanationFeature[];
      top_spam_words: string[];
      top_ham_words: string[];
    };
    shap: {
      method: string;
      features: ExplanationFeature[];
      top_spam_words: string[];
      top_ham_words: string[];
      error?: string;
    };
    agreement: {
      lime_top3_spam: string[];
      shap_top3_spam: string[];
      overlap: string[];
      agreement_score: number | null;
    };
  };
  fidelity: {
    lime: FidelityResult;
    shap: FidelityResult | null;
  };
}

interface AttackModelResult {
  original: { prediction: string; spam_probability: number; key_features: ExplanationFeature[] };
  adversarial: { prediction: string; spam_probability: number; key_features: ExplanationFeature[] };
  fooled: boolean;
  confidence_shift: number;
}

interface AdversarialResult {
  original_text: string;
  attacks: {
    [key: string]: {
      perturbed_text: string;
      attack_level: string;
      attack_description: string;
      models: { [key: string]: AttackModelResult };
    };
  };
}

interface HistoryItem {
  message: string;
  prediction: string;
  time: string;
  is_spam: boolean;
  model: string;
}

/* ───────── constants ───────── */
const SAMPLE_MESSAGES = [
  "Congratulations! You've won a $1000 gift card. Click here now!",
  "Hey, are we still on for lunch tomorrow at 1pm?",
  "URGENT: Your account will be suspended. Verify immediately!",
  "The quarterly report is attached. Let me know your thoughts.",
];

const API = "http://localhost:5000";

/* ───────── component ───────── */
export default function Home() {
  const [message, setMessage] = useState("");
  const [activeTab, setActiveTab] = useState<"detect" | "explain" | "adversarial">("detect");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [explainResult, setExplainResult] = useState<ExplainResult | null>(null);
  const [advResult, setAdvResult] = useState<AdversarialResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedModel, setSelectedModel] = useState("tfidf");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [animateResult, setAnimateResult] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  /* ── API calls ── */
  const checkSpam = async () => {
    if (!message.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    setAnimateResult(false);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, model: selectedModel }),
      });
      if (!res.ok) throw new Error("Server connection failed");
      const data: PredictResult = await res.json();
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

  const runExplain = async () => {
    if (!message.trim()) return;
    setLoading(true);
    setError("");
    setExplainResult(null);
    setAnimateResult(false);
    try {
      const res = await fetch(`${API}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, model: selectedModel }),
      });
      if (!res.ok) throw new Error("Server connection failed");
      const data: ExplainResult = await res.json();
      setExplainResult(data);
      setTimeout(() => setAnimateResult(true), 50);
    } catch {
      setError("Backend offline — is Flask running on port 5000?");
    } finally {
      setLoading(false);
    }
  };

  const runAdversarial = async () => {
    if (!message.trim()) return;
    setLoading(true);
    setError("");
    setAdvResult(null);
    setAnimateResult(false);
    try {
      const res = await fetch(`${API}/evaluate_adversarial`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      if (!res.ok) throw new Error("Server connection failed");
      const data: AdversarialResult = await res.json();
      setAdvResult(data);
      setTimeout(() => setAnimateResult(true), 50);
    } catch {
      setError("Backend offline — is Flask running on port 5000?");
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = () => {
    if (activeTab === "detect") checkSpam();
    else if (activeTab === "explain") runExplain();
    else runAdversarial();
  };

  const handleSample = (s: string) => {
    setMessage(s);
    setResult(null);
    setExplainResult(null);
    setAdvResult(null);
    textareaRef.current?.focus();
  };

  /* ── Feature bar component ── */
  const FeatureBar = ({ feature, maxWeight }: { feature: ExplanationFeature; maxWeight: number }) => {
    const pct = Math.min(Math.abs(feature.weight) / Math.max(maxWeight, 0.001) * 100, 100);
    const isSpam = feature.direction === "spam";
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <span style={{
          fontFamily: "var(--font-mono)", fontSize: 12, fontWeight: 600, width: 90,
          textAlign: "right", color: isSpam ? "var(--spam)" : "var(--safe)",
        }}>
          {feature.word}
        </span>
        <div style={{ flex: 1, height: 8, background: "var(--surface2)", borderRadius: 4, overflow: "hidden" }}>
          <div style={{
            width: `${pct}%`, height: "100%", borderRadius: 4,
            background: isSpam
              ? "linear-gradient(90deg, #e74c3c, #c0392b)"
              : "linear-gradient(90deg, #27ae60, #1a6b45)",
            transition: "width 0.6s cubic-bezier(0.16,1,0.3,1)",
          }} />
        </div>
        <span style={{ fontSize: 10, color: "var(--muted)", width: 50, textAlign: "right", fontFamily: "var(--font-mono)" }}>
          {feature.weight > 0 ? "+" : ""}{feature.weight.toFixed(4)}
        </span>
      </div>
    );
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,300;1,9..144,400&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@400;500;600&display=swap');

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
          --accent-light: #eef2fd;
          --accent-border: #bcc9f2;
          --warn: #d4850a;
          --warn-light: #fdf6ec;
          --warn-border: #f0d9a8;
          --font-head: 'Fraunces', Georgia, serif;
          --font-body: 'DM Sans', system-ui, sans-serif;
          --font-mono: 'JetBrains Mono', monospace;
        }

        body { background: var(--bg); color: var(--text); font-family: var(--font-body); -webkit-font-smoothing: antialiased; }

        .app { min-height: 100vh; display: grid; grid-template-rows: auto 1fr auto; }

        /* NAV */
        nav { background: var(--surface); border-bottom: 1px solid var(--border); padding: 0 48px; height: 60px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 50; }
        .nav-logo { display: flex; align-items: center; gap: 10px; font-family: var(--font-head); font-size: 18px; font-weight: 600; color: var(--text); letter-spacing: -0.3px; }
        .logo-icon { width: 30px; height: 30px; background: var(--text); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 15px; }
        .nav-badge { font-family: var(--font-body); font-size: 11px; font-weight: 500; color: var(--muted); background: var(--surface2); border: 1px solid var(--border); padding: 3px 10px; border-radius: 20px; letter-spacing: 0.02em; }

        /* TABS */
        .tab-bar { display: flex; gap: 2px; background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 3px; }
        .tab-btn { font-family: var(--font-body); font-size: 12px; font-weight: 500; padding: 7px 16px; border-radius: 8px; border: none; cursor: pointer; transition: all 0.18s; background: transparent; color: var(--muted); }
        .tab-btn:hover { color: var(--text2); }
        .tab-btn.active { background: var(--surface); color: var(--text); box-shadow: 0 1px 4px rgba(0,0,0,0.08); font-weight: 600; }

        /* MAIN */
        .main { max-width: 1100px; width: 100%; margin: 0 auto; padding: 52px 40px; display: grid; grid-template-columns: 1fr 320px; gap: 36px; align-items: start; }
        .left { display: flex; flex-direction: column; gap: 24px; }
        .hero { margin-bottom: 4px; }
        .hero-label { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; font-weight: 500; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 14px; }
        .live-dot { width: 6px; height: 6px; background: #22c55e; border-radius: 50%; box-shadow: 0 0 0 0 rgba(34,197,94,0.4); animation: ping 2s ease-in-out infinite; }
        @keyframes ping { 0%, 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); } 50% { box-shadow: 0 0 0 5px rgba(34,197,94,0); } }
        h1 { font-family: var(--font-head); font-size: clamp(38px, 4.5vw, 58px); font-weight: 600; line-height: 1.05; letter-spacing: -1.5px; color: var(--text); margin-bottom: 14px; }
        h1 em { font-style: italic; font-weight: 300; color: var(--muted); }
        .subtitle { font-size: 15px; font-weight: 300; color: var(--text2); line-height: 1.65; max-width: 400px; }

        /* INPUT CARD */
        .input-card { background: var(--surface); border: 1.5px solid var(--border); border-radius: 14px; overflow: hidden; transition: border-color 0.2s, box-shadow 0.2s; }
        .input-card:focus-within { border-color: var(--border2); box-shadow: 0 4px 24px rgba(0,0,0,0.06); }
        .card-top-bar { padding: 12px 18px; background: var(--surface2); border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
        .traffic-lights { display: flex; align-items: center; gap: 6px; }
        .tl { width: 10px; height: 10px; border-radius: 50%; }
        .tl-r { background: #ff5f57; } .tl-y { background: #febc2e; } .tl-g { background: #28c840; }
        .file-label { font-size: 12px; font-weight: 400; color: var(--muted); }
        .char-count { font-size: 11px; color: var(--muted2); }
        textarea { width: 100%; min-height: 148px; padding: 18px 20px; background: var(--surface); color: var(--text); font-family: var(--font-body); font-size: 14px; line-height: 1.75; font-weight: 300; border: none; outline: none; resize: none; }
        textarea::placeholder { color: var(--muted2); }
        .card-bottom-bar { padding: 12px 16px; background: var(--surface2); border-top: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; gap: 12px; }
        .sample-group { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
        .sample-label { font-size: 11px; color: var(--muted2); font-weight: 400; }
        .sample-btn { font-family: var(--font-body); font-size: 11px; font-weight: 500; padding: 4px 11px; border-radius: 20px; border: 1px solid var(--border2); background: var(--surface); color: var(--text2); cursor: pointer; transition: all 0.15s; }
        .sample-btn:hover { background: var(--text); color: var(--bg); border-color: var(--text); }

        /* MODEL SELECTOR */
        .model-selector { display: flex; align-items: center; gap: 8px; }
        .model-label { font-size: 11px; color: var(--muted2); font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
        .model-select { padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border2); background: var(--surface); color: var(--text); font-family: var(--font-body); font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.15s; }
        .model-select:hover { border-color: var(--border); background: var(--surface2); }
        .model-select:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px rgba(45,91,227,0.1); }

        .analyze-btn { display: flex; align-items: center; gap: 7px; padding: 9px 20px; border-radius: 8px; border: none; cursor: pointer; font-family: var(--font-body); font-size: 13px; font-weight: 600; letter-spacing: 0.01em; transition: all 0.18s; flex-shrink: 0; background: var(--text); color: var(--bg); }
        .analyze-btn:disabled { background: var(--border2); color: var(--muted); cursor: not-allowed; }
        .analyze-btn:not(:disabled):hover { background: var(--text2); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .spinner { width: 12px; height: 12px; border: 2px solid rgba(255,255,255,0.3); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ERROR */
        .error-bar { padding: 12px 16px; border-radius: 10px; background: var(--spam-light); border: 1px solid var(--spam-border); font-size: 13px; color: var(--spam); display: flex; align-items: center; gap: 8px; }

        /* RESULT */
        .result-card { border-radius: 14px; padding: 24px 28px; opacity: 0; transform: translateY(6px); transition: opacity 0.35s ease, transform 0.35s ease; }
        .result-card.visible { opacity: 1; transform: translateY(0); }
        .result-spam { background: var(--spam-light); border: 1.5px solid var(--spam-border); }
        .result-safe { background: var(--safe-light); border: 1.5px solid var(--safe-border); }
        .result-accent { background: var(--accent-light); border: 1.5px solid var(--accent-border); }
        .result-warn { background: var(--warn-light); border: 1.5px solid var(--warn-border); }
        .result-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 18px; }
        .verdict-label { font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
        .verdict-text { font-family: var(--font-head); font-size: 40px; font-weight: 600; letter-spacing: -1.5px; line-height: 1; }
        .verdict-spam { color: var(--spam); } .verdict-safe { color: var(--safe); }
        .confidence-badge { font-size: 12px; font-weight: 600; padding: 6px 14px; border-radius: 20px; flex-shrink: 0; margin-top: 4px; }
        .badge-spam { background: rgba(192,57,43,0.1); color: var(--spam); border: 1px solid var(--spam-border); }
        .badge-safe { background: rgba(26,107,69,0.1); color: var(--safe); border: 1px solid var(--safe-border); }
        .progress-track { height: 3px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 14px; }
        .progress-fill { height: 100%; border-radius: 3px; transition: width 1.1s cubic-bezier(0.16,1,0.3,1); }
        .fill-spam { background: var(--spam); } .fill-safe { background: var(--safe); }
        .result-note { font-size: 13px; font-weight: 300; color: var(--text2); line-height: 1.65; }

        /* EXPLANATION SECTION */
        .explain-section { margin-top: 18px; padding-top: 18px; border-top: 1px solid var(--border); }
        .section-title { font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; display: flex; align-items: center; gap: 6px; }

        .explain-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
        .explain-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
        .explain-panel-title { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin-bottom: 12px; }

        /* ADVERSARIAL */
        .adv-attack-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 12px; transition: all 0.2s; }
        .adv-attack-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
        .adv-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .adv-name { font-size: 13px; font-weight: 600; color: var(--text); }
        .adv-level { font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
        .adv-level-character { background: #fef3c7; color: #92400e; }
        .adv-level-word { background: #dbeafe; color: #1e40af; }
        .adv-level-sentence { background: #ede9fe; color: #6d28d9; }
        .adv-perturbed { font-family: var(--font-mono); font-size: 11px; color: var(--text2); background: var(--surface2); padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; line-height: 1.5; word-break: break-all; }
        .adv-model-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-top: 1px solid var(--border); }
        .adv-model-name { font-size: 11px; font-weight: 600; color: var(--muted); width: 50px; text-transform: uppercase; }
        .adv-fooled { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; }
        .adv-fooled-yes { background: rgba(192,57,43,0.12); color: var(--spam); }
        .adv-fooled-no { background: rgba(26,107,69,0.12); color: var(--safe); }
        .adv-shift { font-size: 10px; color: var(--muted); font-family: var(--font-mono); }

        /* FIDELITY */
        .fidelity-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
        .fidelity-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; padding: 4px 0; }
        .fidelity-label { color: var(--muted); font-weight: 400; }
        .fidelity-value { font-weight: 600; font-family: var(--font-mono); font-size: 12px; }
        .fidelity-good { color: var(--safe); }
        .fidelity-bad { color: var(--spam); }

        /* AGREEMENT BADGE */
        .agreement-badge { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 12px; }
        .agreement-high { background: rgba(26,107,69,0.12); color: var(--safe); }
        .agreement-mid { background: rgba(212,133,10,0.12); color: var(--warn); }
        .agreement-low { background: rgba(192,57,43,0.12); color: var(--spam); }

        /* RIGHT SIDEBAR */
        .right { display: flex; flex-direction: column; gap: 16px; position: sticky; top: 76px; }
        .sidebar-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
        .sidebar-title { padding: 12px 18px; border-bottom: 1px solid var(--border); font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); background: var(--surface2); display: flex; justify-content: space-between; align-items: center; }
        .steps { padding: 4px 0; }
        .step { display: flex; align-items: flex-start; gap: 12px; padding: 14px 18px; position: relative; }
        .step:not(:last-child)::after { content: ''; position: absolute; left: 27px; top: 42px; bottom: 0; width: 1px; background: var(--border); }
        .step-num { width: 18px; height: 18px; border-radius: 50%; border: 1.5px solid var(--border2); display: flex; align-items: center; justify-content: center; font-size: 9px; font-weight: 600; color: var(--muted); flex-shrink: 0; margin-top: 1px; position: relative; z-index: 1; background: var(--surface); }
        .step-title { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 3px; }
        .step-desc { font-size: 12px; font-weight: 300; color: var(--muted); line-height: 1.55; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); }
        .stat { background: var(--surface); padding: 16px; text-align: center; }
        .stat-value { font-family: var(--font-head); font-size: 22px; font-weight: 600; letter-spacing: -0.5px; color: var(--text); }
        .stat-label { font-size: 11px; font-weight: 400; color: var(--muted); margin-top: 2px; }
        .history-list { max-height: 240px; overflow-y: auto; }
        .history-list::-webkit-scrollbar { width: 3px; }
        .history-list::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
        .history-item { padding: 12px 18px; display: flex; align-items: flex-start; gap: 10px; border-bottom: 1px solid var(--border); transition: background 0.12s; }
        .history-item:last-child { border-bottom: none; }
        .history-item:hover { background: var(--surface2); }
        .h-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
        .h-dot-spam { background: var(--spam); } .h-dot-safe { background: var(--safe); }
        .h-msg { font-size: 12px; font-weight: 300; color: var(--text2); line-height: 1.5; flex: 1; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
        .h-meta { flex-shrink: 0; text-align: right; }
        .h-verdict { font-size: 10px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
        .h-verdict-spam { color: var(--spam); } .h-verdict-safe { color: var(--safe); }
        .h-time { font-size: 10px; color: var(--muted2); margin-top: 2px; }
        .h-model { font-size: 9px; color: var(--muted); margin-top: 1px; font-weight: 500; }
        .clear-btn { background: none; border: none; color: var(--muted2); cursor: pointer; font-size: 11px; font-family: var(--font-body); font-weight: 400; transition: color 0.15s; }
        .clear-btn:hover { color: var(--spam); }
        footer { border-top: 1px solid var(--border); padding: 18px 48px; display: flex; align-items: center; justify-content: space-between; background: var(--surface); }
        .footer-text { font-size: 11px; color: var(--muted2); font-weight: 400; }

        @media (max-width: 768px) {
          nav { padding: 0 20px; }
          .main { grid-template-columns: 1fr; padding: 28px 20px; }
          .right { position: static; }
          footer { padding: 16px 20px; }
          .explain-grid { grid-template-columns: 1fr; }
        }
      `}</style>

      <div className="app">
        {/* NAV */}
        <nav>
          <div className="nav-logo">
            <div className="logo-icon">🛡</div>
            SpamDetect
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div className="tab-bar">
              <button className={`tab-btn ${activeTab === "detect" ? "active" : ""}`} onClick={() => setActiveTab("detect")}>🔍 Detect</button>
              <button className={`tab-btn ${activeTab === "explain" ? "active" : ""}`} onClick={() => setActiveTab("explain")}>🧠 Explain</button>
              <button className={`tab-btn ${activeTab === "adversarial" ? "active" : ""}`} onClick={() => setActiveTab("adversarial")}>⚔️ Adversarial</button>
            </div>
            <span className="nav-badge">XAI Research · v3.0</span>
          </div>
        </nav>

        {/* MAIN */}
        <div className="main">
          {/* LEFT */}
          <div className="left">
            <div className="hero">
              <div className="hero-label">
                <span className="live-dot" />
                {activeTab === "detect" && "Real-time analysis"}
                {activeTab === "explain" && "Explainable AI"}
                {activeTab === "adversarial" && "Adversarial robustness"}
              </div>
              <h1>
                {activeTab === "detect" && <>Is it spam<br /><em>or legit?</em></>}
                {activeTab === "explain" && <>Why does it<br /><em>think that?</em></>}
                {activeTab === "adversarial" && <>Can it be<br /><em>fooled?</em></>}
              </h1>
              <p className="subtitle">
                {activeTab === "detect" && "Paste any message below. Our ML model classifies it instantly with a confidence score."}
                {activeTab === "explain" && "Compare LIME vs SHAP explanations side-by-side. Measure explanation fidelity and cross-method agreement."}
                {activeTab === "adversarial" && "Test how 9 adversarial attack strategies affect model predictions. Compare TF-IDF vs BERT robustness."}
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
                onChange={(e) => { setMessage(e.target.value.substring(0, 500)); setResult(null); setExplainResult(null); setAdvResult(null); }}
              />
              <div className="card-bottom-bar">
                <div style={{ display: "flex", alignItems: "center", gap: 24, flex: 1 }}>
                  <div className="sample-group">
                    <span className="sample-label">Try:</span>
                    {["Spam", "Ham", "Phishing", "Normal"].map((label, i) => (
                      <button key={i} className="sample-btn" onClick={() => handleSample(SAMPLE_MESSAGES[i])}>{label}</button>
                    ))}
                  </div>
                  {activeTab !== "adversarial" && (
                    <div className="model-selector">
                      <span className="model-label">Model:</span>
                      <select className="model-select" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                        <option value="tfidf">TF-IDF (Fast)</option>
                        <option value="bert">BERT (Accurate)</option>
                      </select>
                    </div>
                  )}
                </div>
                <button className="analyze-btn" onClick={handleAnalyze} disabled={loading || !message.trim()}>
                  {loading ? (<><div className="spinner" /> Analyzing…</>) : (
                    <>
                      {activeTab === "detect" && "Analyze →"}
                      {activeTab === "explain" && "Explain →"}
                      {activeTab === "adversarial" && "Attack →"}
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* ERROR */}
            {error && <div className="error-bar"><span>⚠</span> {error}</div>}

            {/* ═══ DETECT TAB RESULT ═══ */}
            {activeTab === "detect" && result && (
              <div className={`result-card ${result.is_spam ? "result-spam" : "result-safe"} ${animateResult ? "visible" : ""}`}>
                <div className="result-header">
                  <div>
                    <div className="verdict-label">Verdict</div>
                    <div className={`verdict-text ${result.is_spam ? "verdict-spam" : "verdict-safe"}`}>{result.prediction}</div>
                  </div>
                  <div className={`confidence-badge ${result.is_spam ? "badge-spam" : "badge-safe"}`}>{result.probability}% confident</div>
                </div>
                <div className="progress-track">
                  <div className={`progress-fill ${result.is_spam ? "fill-spam" : "fill-safe"}`} style={{ width: animateResult ? `${result.probability}%` : "0%" }} />
                </div>

                {/* Explanation Bars */}
                {result.explanation && result.explanation.length > 0 && (
                  <div className="explain-section">
                    <div className="section-title">🔬 Why this classification</div>
                    {(() => {
                      const maxW = Math.max(...result.explanation.map(f => Math.abs(f.weight)));
                      return result.explanation.map((f, i) => <FeatureBar key={i} feature={f} maxWeight={maxW} />);
                    })()}
                  </div>
                )}

                <p className="result-note" style={{ marginTop: 14 }}>
                  {result.is_spam
                    ? "This message exhibits patterns associated with spam or phishing."
                    : "This message appears to be legitimate."}
                  <br />
                  <span style={{ marginTop: 8, display: "block", fontSize: 11, color: "var(--muted)", fontWeight: 500 }}>
                    Analyzed with {result.model_used === "bert" ? "BERT" : "TF-IDF"} model
                  </span>
                </p>
              </div>
            )}

            {/* ═══ EXPLAIN TAB RESULT ═══ */}
            {activeTab === "explain" && explainResult && (
              <div className={`result-card result-accent ${animateResult ? "visible" : ""}`}>
                <div className="result-header">
                  <div>
                    <div className="verdict-label">Prediction</div>
                    <div className={`verdict-text ${explainResult.is_spam ? "verdict-spam" : "verdict-safe"}`}>{explainResult.prediction}</div>
                  </div>
                  <div className={`confidence-badge ${explainResult.is_spam ? "badge-spam" : "badge-safe"}`}>{explainResult.probability}%</div>
                </div>

                {/* LIME vs SHAP side-by-side */}
                <div className="section-title" style={{ marginTop: 14 }}>📊 LIME vs SHAP Comparison</div>
                <div className="explain-grid">
                  {/* LIME */}
                  <div className="explain-panel">
                    <div className="explain-panel-title">LIME</div>
                    {explainResult.explanations.lime.features.slice(0, 6).map((f, i) => {
                      const maxW = Math.max(...explainResult.explanations.lime.features.map(x => Math.abs(x.weight)));
                      return <FeatureBar key={i} feature={f} maxWeight={maxW} />;
                    })}
                  </div>
                  {/* SHAP */}
                  <div className="explain-panel">
                    <div className="explain-panel-title" style={{ color: "#8b5cf6" }}>SHAP</div>
                    {explainResult.explanations.shap.error ? (
                      <p style={{ fontSize: 12, color: "var(--muted)" }}>{explainResult.explanations.shap.error}</p>
                    ) : (
                      explainResult.explanations.shap.features.slice(0, 6).map((f, i) => {
                        const maxW = Math.max(...explainResult.explanations.shap.features.map(x => Math.abs(x.weight)));
                        return <FeatureBar key={i} feature={f} maxWeight={maxW} />;
                      })
                    )}
                  </div>
                </div>

                {/* Agreement */}
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
                  <span style={{ fontSize: 12, fontWeight: 500, color: "var(--muted)" }}>Method Agreement:</span>
                  {explainResult.explanations.agreement.agreement_score !== null ? (
                    <span className={`agreement-badge ${
                      explainResult.explanations.agreement.agreement_score >= 0.66 ? "agreement-high" :
                      explainResult.explanations.agreement.agreement_score >= 0.33 ? "agreement-mid" : "agreement-low"
                    }`}>
                      {(explainResult.explanations.agreement.agreement_score * 100).toFixed(0)}% overlap
                    </span>
                  ) : (
                    <span className="agreement-badge agreement-low">N/A</span>
                  )}
                  {explainResult.explanations.agreement.overlap.length > 0 && (
                    <span style={{ fontSize: 11, color: "var(--text2)" }}>
                      Shared: {explainResult.explanations.agreement.overlap.join(", ")}
                    </span>
                  )}
                </div>

                {/* Fidelity */}
                <div className="section-title">🎯 Explanation Fidelity</div>
                <div className="explain-grid">
                  <div className="fidelity-card">
                    <div style={{ fontSize: 11, fontWeight: 600, color: "var(--accent)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.06em" }}>LIME Fidelity</div>
                    <div className="fidelity-row">
                      <span className="fidelity-label">Original spam prob</span>
                      <span className="fidelity-value">{(explainResult.fidelity.lime.original_spam_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="fidelity-row">
                      <span className="fidelity-label">After removing top-3</span>
                      <span className="fidelity-value">{(explainResult.fidelity.lime.reduced_spam_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="fidelity-row">
                      <span className="fidelity-label">Prediction flipped?</span>
                      <span className={`fidelity-value ${explainResult.fidelity.lime.prediction_flipped ? "fidelity-good" : "fidelity-bad"}`}>
                        {explainResult.fidelity.lime.prediction_flipped ? "✓ Yes" : "✗ No"}
                      </span>
                    </div>
                    <div className="fidelity-row">
                      <span className="fidelity-label">Fidelity score</span>
                      <span className={`fidelity-value ${explainResult.fidelity.lime.fidelity_score > 0.3 ? "fidelity-good" : "fidelity-bad"}`}>
                        {explainResult.fidelity.lime.fidelity_score.toFixed(4)}
                      </span>
                    </div>
                    <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 6 }}>
                      Removed: {explainResult.fidelity.lime.removed_words.join(", ")}
                    </div>
                  </div>
                  {explainResult.fidelity.shap && (
                    <div className="fidelity-card">
                      <div style={{ fontSize: 11, fontWeight: 600, color: "#8b5cf6", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.06em" }}>SHAP Fidelity</div>
                      <div className="fidelity-row">
                        <span className="fidelity-label">Original spam prob</span>
                        <span className="fidelity-value">{(explainResult.fidelity.shap.original_spam_prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="fidelity-row">
                        <span className="fidelity-label">After removing top-3</span>
                        <span className="fidelity-value">{(explainResult.fidelity.shap.reduced_spam_prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="fidelity-row">
                        <span className="fidelity-label">Prediction flipped?</span>
                        <span className={`fidelity-value ${explainResult.fidelity.shap.prediction_flipped ? "fidelity-good" : "fidelity-bad"}`}>
                          {explainResult.fidelity.shap.prediction_flipped ? "✓ Yes" : "✗ No"}
                        </span>
                      </div>
                      <div className="fidelity-row">
                        <span className="fidelity-label">Fidelity score</span>
                        <span className={`fidelity-value ${explainResult.fidelity.shap.fidelity_score > 0.3 ? "fidelity-good" : "fidelity-bad"}`}>
                          {explainResult.fidelity.shap.fidelity_score.toFixed(4)}
                        </span>
                      </div>
                      <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 6 }}>
                        Removed: {explainResult.fidelity.shap.removed_words.join(", ")}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* ═══ ADVERSARIAL TAB RESULT ═══ */}
            {activeTab === "adversarial" && advResult && (
              <div className={`result-card result-warn ${animateResult ? "visible" : ""}`}>
                <div className="result-header">
                  <div>
                    <div className="verdict-label">Adversarial Robustness Test</div>
                    <div className="verdict-text" style={{ fontSize: 28, color: "var(--warn)" }}>
                      {Object.values(advResult.attacks).reduce((acc, atk) =>
                        acc + Object.values(atk.models).filter(m => m.fooled).length, 0
                      )} evasions detected
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: "var(--muted)", textAlign: "right" }}>
                    {Object.keys(advResult.attacks).length} attacks<br />tested
                  </div>
                </div>

                {Object.entries(advResult.attacks).map(([name, attack]) => (
                  <div key={name} className="adv-attack-card">
                    <div className="adv-header">
                      <span className="adv-name">{name.replace(/_/g, " ")}</span>
                      <span className={`adv-level adv-level-${attack.attack_level}`}>{attack.attack_level}</span>
                    </div>
                    <div className="adv-perturbed">&ldquo;{attack.perturbed_text.substring(0, 120)}{attack.perturbed_text.length > 120 ? "…" : ""}&rdquo;</div>

                    {Object.entries(attack.models).map(([modelName, modelResult]) => (
                      <div key={modelName} className="adv-model-row">
                        <span className="adv-model-name">{modelName}</span>
                        <span style={{ fontSize: 11, color: "var(--text2)", flex: 1 }}>
                          {modelResult.original.prediction} → {modelResult.adversarial.prediction}
                        </span>
                        <span className="adv-shift">Δ {modelResult.confidence_shift > 0 ? "+" : ""}{modelResult.confidence_shift}%</span>
                        <span className={`adv-fooled ${modelResult.fooled ? "adv-fooled-yes" : "adv-fooled-no"}`}>
                          {modelResult.fooled ? "EVADED" : "HELD"}
                        </span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* RIGHT SIDEBAR */}
          <div className="right">
            <div className="sidebar-card">
              <div className="sidebar-title">
                {activeTab === "detect" && "How it works"}
                {activeTab === "explain" && "About XAI"}
                {activeTab === "adversarial" && "Attack types"}
              </div>
              <div className="steps">
                {activeTab === "detect" && [
                  { n: "01", title: "Paste message", desc: "Drop any email, SMS, or text into the editor." },
                  { n: "02", title: "ML processes it", desc: "TF-IDF or BERT encoder vectorizes every token." },
                  { n: "03", title: "Verdict + explanation", desc: "Get spam/ham result with LIME feature importance." },
                ].map((s) => (
                  <div className="step" key={s.n}><div className="step-num">{s.n}</div><div><div className="step-title">{s.title}</div><div className="step-desc">{s.desc}</div></div></div>
                ))}
                {activeTab === "explain" && [
                  { n: "01", title: "LIME", desc: "Perturbs input and observes prediction changes to find important words." },
                  { n: "02", title: "SHAP", desc: "Uses Shapley values from game theory for theoretically grounded attributions." },
                  { n: "03", title: "Fidelity test", desc: "Removes top features to verify if the explanation is genuinely faithful." },
                ].map((s) => (
                  <div className="step" key={s.n}><div className="step-num">{s.n}</div><div><div className="step-title">{s.title}</div><div className="step-desc">{s.desc}</div></div></div>
                ))}
                {activeTab === "adversarial" && [
                  { n: "C", title: "Character-level", desc: "Leetspeak, homoglyphs, char insertion/deletion/swap." },
                  { n: "W", title: "Word-level", desc: "Synonym replacement, benign word injection to dilute signal." },
                  { n: "S", title: "Sentence-level", desc: "Word shuffling, invisible Unicode characters." },
                ].map((s) => (
                  <div className="step" key={s.n}><div className="step-num">{s.n}</div><div><div className="step-title">{s.title}</div><div className="step-desc">{s.desc}</div></div></div>
                ))}
              </div>
            </div>

            <div className="sidebar-card">
              <div className="sidebar-title">Research capabilities</div>
              <div className="stats-grid">
                {[
                  { v: "2", l: "Models" },
                  { v: "9", l: "Attacks" },
                  { v: "2", l: "Explainers" },
                  { v: "3", l: "Levels" },
                ].map((s) => (
                  <div className="stat" key={s.l}>
                    <div className="stat-value">{s.v}</div>
                    <div className="stat-label">{s.l}</div>
                  </div>
                ))}
              </div>
            </div>

            {history.length > 0 && (
              <div className="sidebar-card">
                <div className="sidebar-title">
                  <span>Recent checks</span>
                  <button className="clear-btn" onClick={() => setHistory([])}>Clear all</button>
                </div>
                <div className="history-list">
                  {history.map((item, i) => (
                    <div className="history-item" key={i}>
                      <div className={`h-dot ${item.is_spam ? "h-dot-spam" : "h-dot-safe"}`} />
                      <div className="h-msg">{item.message}{item.message.length >= 60 ? "…" : ""}</div>
                      <div className="h-meta">
                        <div className={`h-verdict ${item.is_spam ? "h-verdict-spam" : "h-verdict-safe"}`}>{item.prediction}</div>
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

        <footer>
          <span className="footer-text">SpamDetect XAI Research © 2025 · Flask backend on :5000</span>
          <span className="footer-text">LIME + SHAP + Adversarial Testing</span>
        </footer>
      </div>
    </>
  );
}