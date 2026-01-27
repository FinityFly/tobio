import React, { useState } from "react";
import { ArrowRight } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface ChatAgentServiceProps {
  csvData: string | null;
}

const FIXED_PROMPT = `
Role:
You are a helpful Volleyball Coaching and Performance Analytics Agent. You act as an experienced volleyball coach, scout, and data analyst.

Input:
You will be given CSV data representing volleyball statistics. The CSV may include (but is not limited to):

individual player stats (serving, attacking, receiving, setting, blocking, digging, errors)

team-level stats (side-out %, point-scoring %, rotation performance, serve pressure, efficiency ratings)

match logs, rally data, rotations, or point-by-point records

Your Job:
Analyze the data as an elite volleyball coach would.
You must:

Extract insights from the data (patterns, trends, outliers, consistencies).

Identify weak spots—both team-level and player-level.

Identify strengths and standout performers.

Explain causes behind performance problems whenever possible.

Suggest concrete improvements, such as:

training drills

tactical adjustments

rotation optimization

player-specific development focus

Communicate clearly, using:

bullet points

concise explanations

percentages, averages, and comparisons when calculable

visual metaphors or coaching terminology when helpful

Act like a real coach, not just a statistician—focus on what actually changes performance.

Output Requirements:
Do not say any phrases about being an AI model and do not say any preamble like "Let's do ()" nor any unnecessary end comments that don't add any value.
When responding, structure your output like this:

1. Summary of the Data

Brief high-level overview of what the dataset represents.

2. Key Strengths

Top improvements, best players, strongest rotations, best skills, etc.

3. Key Weaknesses

Where the team struggles (skills, rotations, efficiency issues, consistency problems).

4. Detailed Statistical Insights

Numbers, anomalies, correlations, inefficiencies, matchup patterns, etc.

5. Recommendations & Coaching Plan

Step-by-step improvements (drills, tactical changes, substitutions, rotation tweaks, player development goals).

6. Optional Visualizations

If asked, generate charts/tables based on the data.

Tone:
Professional, coach-like, direct, constructive, and actionable.
`;

export const ChatAgentService: React.FC<ChatAgentServiceProps> = ({ csvData }) => {
  const geminiApiKey = import.meta.env.VITE_GEMENI_API_KEY as string;
  console.log("Gemini API Key:", geminiApiKey);
  const [messages, setMessages] = useState<Array<{ role: "user" | "agent"; text: string }>>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [csvAnalyzed, setCsvAnalyzed] = useState(false);

  // Send initial analysis when csvData is available
  React.useEffect(() => {
    if (csvData && !csvAnalyzed) {
      const analyzeCsv = async () => {
        setLoading(true);
        try {
          const response = await fetch(
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent',
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'x-goog-api-key': geminiApiKey,
              },
              body: JSON.stringify({
                contents: [
                  {
                    parts: [
                      { text: `${FIXED_PROMPT}\n\nAnalyze the following volleyball CSV data and provide a full coaching report. Do not say anything related to not having enough data.\n\n${csvData}` }
                    ]
                  }
                ]
              })
            }
          );
          const data = await response.json();
          const agentText = data.candidates?.[0]?.content?.parts?.[0]?.text || '(No response)';
          setMessages([{ role: 'agent', text: agentText }]);
        } catch (err) {
          setMessages([{ role: 'agent', text: 'Error: Could not reach Gemini.' }]);
        }
        setLoading(false);
        setCsvAnalyzed(true);
      };
      analyzeCsv();
    }
  }, [csvData, csvAnalyzed, geminiApiKey]);

  const sendMessage = async (text: string) => {
    setMessages((msgs) => [...msgs, { role: 'user', text }]);
    setLoading(true);
    try {
      // Build conversation history for Gemini
      const history = [
        { parts: [ { text: `${FIXED_PROMPT}\n\nCSV Data:\n${csvData || ''}` } ] }
      ];
      messages.forEach(m => {
        history.push({ parts: [ { text: m.text } ] });
      });
      history.push({ parts: [ { text } ] });

      const response = await fetch(
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-goog-api-key': geminiApiKey,
          },
          body: JSON.stringify({ contents: history })
        }
      );
      const data = await response.json();
      const agentText = data.candidates?.[0]?.content?.parts?.[0]?.text || '(No response)';
      setMessages((msgs) => [...msgs, { role: 'agent', text: agentText }]);
    } catch (err) {
      setMessages((msgs) => [...msgs, { role: 'agent', text: 'Error: Could not reach Gemini.' }]);
    }
    setLoading(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    sendMessage(input.trim());
    setInput("");
  };

  // Set sidebar width CSS variable to 500px when agent submenu is open
  React.useEffect(() => {
    document.documentElement.style.setProperty('--sidebar-width', `480px`);
    return () => {
      document.documentElement.style.setProperty('--sidebar-width', `320px`);
    };
  }, []);

  return (
    <div className="chat-agent-service" style={{ display: "flex", flexDirection: "column", height: "100%", width: "var(--sidebar-width)" }}>
      <div className="chat-agent-messages" style={{ flex: 1, overflowY: "auto", marginBottom: "1rem", background: "#f3f4f6", borderRadius: 8, padding: 8 }}>
        {messages.length === 0 && <div style={{ color: "#888" }}>Ask a question about the CSV statistics...</div>}
        {messages.map((msg, i) => (
          <div key={i} style={{ marginBottom: 8, textAlign: "left" }}>
            {msg.role === "user" ? (
              <span style={{ background: "#6366F1", color: "#fff", padding: "6px 12px", borderRadius: 8, display: "inline-block", maxWidth: "50%" }}>{msg.text}</span>
            ) : (
              <span style={{ background: "#fff", color: "#222", padding: "6px 12px", borderRadius: 8, display: "inline-block", maxWidth: "80%" }}>
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </span>
            )}
          </div>
        ))}
        {loading && <div style={{ color: "#6366F1" }}>Agent is thinking...</div>}
      </div>
      <form onSubmit={handleSubmit} style={{ width: "auto" }}>
        <div style={{ display: "flex", alignItems: "center", maxWidth: 400, width: "100%", gap: "0.5rem" }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your question..."
            style={{ flex: 1, minWidth: 0, boxSizing: "border-box", padding: "0.7rem", borderRadius: 8, border: "1px solid #e2e8f0", fontSize: "1rem" }}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()} style={{ background: "#6366F1", color: "#fff", border: "none", borderRadius: 8, padding: "0.5rem", display: "flex", alignItems: "center", justifyContent: "center", cursor: loading ? "not-allowed" : "pointer" }}>
            <ArrowRight size={22} />
          </button>
        </div>
      </form>
    </div>
  );
};
