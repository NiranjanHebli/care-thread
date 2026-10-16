import React, { useState, useEffect } from 'react';
import { fetchWhyNot } from '../api';
import type { WhyNotResult } from '../api';

interface WhyNotDrawerProps {
  currentAnswerId?: string;
  selectedNodeId?: string;
  onClose?: () => void;
}

export const WhyNotDrawer: React.FC<WhyNotDrawerProps> = ({
  currentAnswerId,
  selectedNodeId,
  onClose,
}) => {
  const [candidateId, setCandidateId] = useState(selectedNodeId || '');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<WhyNotResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (selectedNodeId) {
      setCandidateId(selectedNodeId);
    }
  }, [selectedNodeId]);

  const handleQuery = async () => {
    if (!currentAnswerId || !candidateId.trim()) {
      setError('Please provide both an Answer ID and a Candidate Node ID.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetchWhyNot(currentAnswerId, candidateId.trim());
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Why-not query failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-card" id="why-not-drawer">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h4 style={{ fontSize: '13px', color: 'var(--accent-rose)' }}>
          Why-Not Explainability Analyzer
        </h4>
        {onClose && (
          <button
            className="btn btn-secondary"
            id="close-whynot-btn"
            onClick={onClose}
            style={{ padding: '2px 8px', fontSize: '11px' }}
          >
            X
          </button>
        )}
      </div>

      <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '6px' }}>
        Investigate why an alternative fact, guideline, or candidate was rejected during reasoning.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '10px' }}>
        <div>
          <label style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Target Answer ID:</label>
          <input
            type="text"
            className="input-field"
            id="why-not-answer-id"
            value={currentAnswerId || ''}
            readOnly
            placeholder="Select or generate an answer first"
            style={{ width: '100%', fontFamily: 'monospace', fontSize: '11px' }}
          />
        </div>

        <div>
          <label style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Candidate Node ID:</label>
          <input
            type="text"
            className="input-field"
            id="why-not-input"
            value={candidateId}
            onChange={(e) => setCandidateId(e.target.value)}
            placeholder="e.g. fact_123 or node ID"
            style={{ width: '100%', fontFamily: 'monospace', fontSize: '12px' }}
          />
        </div>

        <button
          className="btn btn-primary"
          id="why-not-btn"
          onClick={handleQuery}
          disabled={loading || !currentAnswerId || !candidateId.trim()}
          style={{ marginTop: '4px' }}
        >
          {loading ? 'Analyzing Graph...' : 'Explain Rejection'}
        </button>
      </div>

      {error && (
        <div
          id="why-not-error"
          style={{
            marginTop: '10px',
            padding: '8px',
            borderRadius: 'var(--radius-sm)',
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            color: '#fca5a5',
            fontSize: '12px',
          }}
        >
          {error}
        </div>
      )}

      {result && (
        <div
          className="glass-card"
          id="why-not-result"
          style={{ marginTop: '12px', background: '#172033', borderColor: 'var(--accent-rose)' }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Rejection Code:
            </span>
            <span
              style={{
                fontSize: '12px',
                fontWeight: 700,
                color: '#f87171',
                fontFamily: 'monospace',
                background: 'rgba(239, 68, 68, 0.15)',
                padding: '2px 6px',
                borderRadius: '4px',
              }}
            >
              {result.reason_code}
            </span>
          </div>

          <div style={{ marginTop: '10px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>
              Blocking Graph Path:
            </div>
            {result.blocking_path && result.blocking_path.length > 0 ? (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                  fontFamily: 'monospace',
                  fontSize: '11px',
                }}
              >
                {result.blocking_path.map((step, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '4px 8px',
                      background: 'rgba(0,0,0,0.3)',
                      borderRadius: '4px',
                    }}
                  >
                    <span style={{ color: 'var(--accent-cyan)' }}>{step.source}</span>
                    <span style={{ color: '#f59e0b' }}> -[{step.relationship}]-&gt; </span>
                    <span style={{ color: '#e11d48' }}>{step.target}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div
                style={{
                  fontSize: '11px',
                  color: 'var(--text-secondary)',
                  fontStyle: 'italic',
                }}
              >
                No explicit blocking relationship path found in the active subgraph.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
