import React, { useState } from 'react';
import { streamAgentChat } from '../api';
import type { TraceData } from '../api';

interface AgentConsoleProps {
  patientId: string;
  onTraceReceived: (trace: TraceData) => void;
  onAnswerCreated?: (answer: any) => void;
}

interface StreamEventItem {
  id: string;
  event: string;
  data: any;
  timestamp: string;
}

export const AgentConsole: React.FC<AgentConsoleProps> = ({
  patientId,
  onTraceReceived,
  onAnswerCreated,
}) => {
  const [role, setRole] = useState<'workflow' | 'intake' | 'specialist' | 'pharmacy'>('workflow');
  const [message, setMessage] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [handoffId, setHandoffId] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [events, setEvents] = useState<StreamEventItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleSend = async () => {
    if (!message.trim() || !patientId || streaming) return;

    setStreaming(true);
    setError(null);
    const newEvents: StreamEventItem[] = [];

    try {
      await streamAgentChat(
        role,
        {
          patient_id: patientId,
          message: message.trim(),
          session_id: sessionId.trim() || undefined,
          handoff_id: handoffId.trim() || undefined,
        },
        (eventName, eventData) => {
          const item: StreamEventItem = {
            id: `${Date.now()}-${Math.random()}`,
            event: eventName,
            data: eventData,
            timestamp: new Date().toLocaleTimeString(),
          };
          newEvents.push(item);
          setEvents([...newEvents]);

          if (eventName === 'trace') {
            onTraceReceived(eventData as TraceData);
          } else if (eventName === 'final') {
            if (eventData.answer_id && onAnswerCreated) {
              onAnswerCreated(eventData);
            }
          }
        }
      );
    } catch (err: any) {
      setError(err.message || 'Stream connection failed');
    } finally {
      setStreaming(false);
    }
  };

  return (
    <div className="chat-container" id="agent-console">
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <label style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Agent:</label>
          <select
            className="input-field"
            id="agent-role-select"
            value={role}
            onChange={(e) => setRole(e.target.value as any)}
            style={{ flex: 1 }}
          >
            <option value="workflow">End-to-End Clinical Workflow</option>
            <option value="intake">Intake Agent</option>
            <option value="specialist">Specialist Agent</option>
            <option value="pharmacy">Pharmacy Safety Agent</option>
          </select>
        </div>

        {role !== 'workflow' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            <input
              type="text"
              className="input-field"
              id="session-id-input"
              placeholder="Session ID (optional)"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
            />
            <input
              type="text"
              className="input-field"
              id="handoff-id-input"
              placeholder="Handoff ID (optional)"
              value={handoffId}
              onChange={(e) => setHandoffId(e.target.value)}
            />
          </div>
        )}

        <textarea
          className="input-field"
          id="chat-message-input"
          placeholder="Enter patient complaint or clinical instructions..."
          rows={3}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          disabled={streaming}
          style={{ resize: 'vertical' }}
        />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Patient ID: {patientId || 'None selected'}
          </span>
          <button
            className="btn btn-primary"
            id="chat-send-btn"
            onClick={handleSend}
            disabled={streaming || !patientId || !message.trim()}
          >
            {streaming ? 'Reasoning...' : 'Send Message'}
          </button>
        </div>
      </div>

      {error && (
        <div
          className="glass-card"
          id="chat-error-banner"
          style={{ borderColor: 'var(--accent-rose)', color: '#fca5a5', fontSize: '12px' }}
        >
          {error}
        </div>
      )}

      <div className="stream-feed" id="agent-stream-feed">
        {events.length === 0 && (
          <div
            style={{
              color: 'var(--text-muted)',
              fontSize: '12px',
              textAlign: 'center',
              padding: '24px 0',
            }}
          >
            No agent reasoning active. Submit a clinical query to stream events.
          </div>
        )}

        {events.map((ev) => (
          <div key={ev.id} className={`stream-item stream-${ev.event}`}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '11px',
                fontWeight: 600,
                marginBottom: '4px',
                opacity: 0.8,
              }}
            >
              <span>EVENT: {ev.event.toUpperCase()}</span>
              <span>{ev.timestamp}</span>
            </div>

            {ev.event === 'tool_call' && (
              <div>
                <strong>Calling Tool:</strong> {ev.data.tool}
                <pre style={{ fontSize: '11px', marginTop: '4px', overflowX: 'auto' }}>
                  {JSON.stringify(ev.data.args, null, 2)}
                </pre>
              </div>
            )}

            {ev.event === 'tool_result' && (
              <div>
                <strong>Tool Result ({ev.data.tool}):</strong>
                <pre style={{ fontSize: '11px', marginTop: '4px', overflowX: 'auto' }}>
                  {JSON.stringify(ev.data.result, null, 2)}
                </pre>
              </div>
            )}

            {ev.event === 'token' && <div>{ev.data.token}</div>}

            {ev.event === 'final' && (
              <div>
                <div style={{ fontWeight: 700, color: '#d8b4fe', marginBottom: '4px' }}>
                  Final Clinical Recommendation
                </div>
                {ev.data.diagnosis && (
                  <div>
                    <strong>Diagnosis:</strong> {ev.data.diagnosis}
                  </div>
                )}
                {ev.data.recommendation && (
                  <div style={{ marginTop: '4px' }}>
                    <strong>Plan:</strong> {ev.data.recommendation}
                  </div>
                )}
                {ev.data.answer_id && (
                  <div style={{ marginTop: '6px', fontSize: '11px', fontFamily: 'monospace' }}>
                    Answer ID: {ev.data.answer_id}
                  </div>
                )}
              </div>
            )}

            {ev.event === 'trace' && (
              <div>
                <span style={{ color: 'var(--accent-cyan)' }}>
                  Graph Trace Loaded: {ev.data.used?.length || 0} used,{' '}
                  {ev.data.rejected?.length || 0} rejected
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
