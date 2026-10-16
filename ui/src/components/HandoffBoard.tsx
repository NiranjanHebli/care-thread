import React, { useState, useEffect, useCallback } from 'react';
import { fetchHandoffs, fetchFlags, resolveFlag } from '../api';
import type { HandoffRecord, FlagRecord } from '../api';

interface HandoffBoardProps {
  patientId: string;
}

export const HandoffBoard: React.FC<HandoffBoardProps> = ({ patientId }) => {
  const [handoffs, setHandoffs] = useState<HandoffRecord[]>([]);
  const [flags, setFlags] = useState<FlagRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedHandoff, setSelectedHandoff] = useState<HandoffRecord | null>(null);

  const loadData = useCallback(async () => {
    if (!patientId) return;
    setLoading(true);
    try {
      const [hList, fList] = await Promise.all([
        fetchHandoffs(patientId),
        fetchFlags(patientId),
      ]);
      setHandoffs(hList);
      setFlags(fList);
    } catch (err) {
      console.error('Failed to load handoffs/flags:', err);
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleResolve = async (flagId: string) => {
    try {
      await resolveFlag(flagId, patientId, 'Clinician override via UI');
      await loadData();
    } catch (err) {
      console.error('Resolve failed:', err);
    }
  };

  const columns = [
    { id: 'draft', title: 'Draft', color: '#94a3b8' },
    { id: 'open', title: 'Open', color: '#38bdf8' },
    { id: 'accepted', title: 'Accepted', color: '#f59e0b' },
    { id: 'completed', title: 'Completed', color: '#10b981' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '14px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Clinical Handoff &amp; Safety Board</h3>
        <button
          className="btn btn-secondary"
          id="refresh-handoffs-btn"
          onClick={loadData}
          disabled={loading}
          style={{ fontSize: '12px' }}
        >
          {loading ? 'Refreshing...' : 'Refresh Board'}
        </button>
      </div>

      {/* Safety Flags Section */}
      <div className="glass-card" id="flags-container">
        <h4 style={{ fontSize: '13px', color: 'var(--accent-rose)', marginBottom: '8px' }}>
          Active Safety Flags &amp; Clinical Conflicts
        </h4>
        {flags.length === 0 ? (
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            No active safety flags detected for this patient.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {flags.map((f) => (
              <div
                key={f.flag_id}
                className="glass-card"
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '8px 12px',
                  background: 'rgba(239, 68, 68, 0.1)',
                  borderColor: 'rgba(239, 68, 68, 0.25)',
                }}
              >
                <div>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <span
                      style={{
                        fontSize: '10px',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        padding: '1px 6px',
                        borderRadius: '3px',
                        background: f.severity === 'critical' ? '#dc2626' : '#d97706',
                        color: 'white',
                      }}
                    >
                      {f.severity}
                    </span>
                    <span style={{ fontSize: '12px', fontWeight: 600 }}>{f.flag_type}</span>
                  </div>
                  <div style={{ fontSize: '11px', color: '#fca5a5', marginTop: '2px' }}>
                    {f.reason}
                  </div>
                </div>

                {f.status !== 'resolved' ? (
                  <button
                    className="btn btn-danger"
                    id={`resolve-flag-btn-${f.flag_id}`}
                    onClick={() => handleResolve(f.flag_id)}
                    style={{ fontSize: '11px', padding: '4px 10px' }}
                  >
                    Resolve
                  </button>
                ) : (
                  <span style={{ fontSize: '11px', color: 'var(--accent-emerald)' }}>
                    Resolved
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Kanban Board */}
      <div className="handoff-grid" id="handoff-board">
        {columns.map((col) => {
          const colHandoffs = handoffs.filter(
            (h) => h.status.toLowerCase() === col.id.toLowerCase()
          );
          return (
            <div key={col.id} className="handoff-column" id={`handoff-column-${col.id}`}>
              <div className="column-header" style={{ borderTop: `3px solid ${col.color}` }}>
                <span>{col.title}</span>
                <span
                  style={{
                    background: 'rgba(255,255,255,0.1)',
                    padding: '1px 6px',
                    borderRadius: '999px',
                  }}
                >
                  {colHandoffs.length}
                </span>
              </div>
              <div className="column-body">
                {colHandoffs.length === 0 ? (
                  <div
                    style={{
                      fontSize: '11px',
                      color: 'var(--text-muted)',
                      textAlign: 'center',
                      padding: '16px 0',
                    }}
                  >
                    No items
                  </div>
                ) : (
                  colHandoffs.map((h) => (
                    <div
                      key={h.handoff_id}
                      className="glass-card"
                      onClick={() => setSelectedHandoff(h)}
                      style={{
                        cursor: 'pointer',
                        padding: '10px',
                        borderLeft: `3px solid ${col.color}`,
                      }}
                    >
                      <div
                        style={{
                          fontSize: '11px',
                          color: 'var(--text-muted)',
                          display: 'flex',
                          justifyContent: 'space-between',
                        }}
                      >
                        <span>{h.sender_role} &rarr; {h.receiver_role}</span>
                        <span style={{ textTransform: 'uppercase' }}>{h.priority || 'NORMAL'}</span>
                      </div>
                      <div
                        style={{
                          fontSize: '12px',
                          fontWeight: 500,
                          marginTop: '4px',
                          lineHeight: '1.4',
                        }}
                      >
                        {h.clinical_summary || 'Clinical task handoff'}
                      </div>
                      <div
                        style={{
                          fontSize: '10px',
                          color: 'var(--text-muted)',
                          fontFamily: 'monospace',
                          marginTop: '6px',
                        }}
                      >
                        ID: {(h.handoff_id || '').slice(0, 14)}...
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          );
        })}
      </div>

      {selectedHandoff && (
        <div className="glass-card" id="handoff-detail-modal" style={{ marginTop: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ fontSize: '13px', color: 'var(--accent-cyan)' }}>
              Handoff Packet Detail: {selectedHandoff.handoff_id}
            </h4>
            <button
              className="btn btn-secondary"
              onClick={() => setSelectedHandoff(null)}
              style={{ padding: '2px 8px', fontSize: '11px' }}
            >
              Close
            </button>
          </div>
          <pre
            style={{
              fontSize: '11px',
              marginTop: '8px',
              background: '#0a0e17',
              padding: '8px',
              borderRadius: '4px',
              overflowX: 'auto',
            }}
          >
            {JSON.stringify(selectedHandoff, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};
