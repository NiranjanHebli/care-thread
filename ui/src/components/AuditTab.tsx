import React, { useState, useEffect, useCallback } from 'react';
import { fetchAuditLogs } from '../api';
import type { AuditEvent } from '../api';

interface AuditTabProps {
  patientId: string;
}

export const AuditTab: React.FC<AuditTabProps> = ({ patientId }) => {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('');

  const loadAudit = useCallback(async () => {
    if (!patientId) return;
    setLoading(true);
    try {
      const data = await fetchAuditLogs(patientId, 100);
      setEvents(data);
    } catch (err) {
      console.error('Failed to load audit logs:', err);
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    loadAudit();
  }, [loadAudit]);

  const filteredEvents = events.filter((ev) => {
    if (!filter.trim()) return true;
    const f = filter.toLowerCase();
    return (
      ev.action?.toLowerCase().includes(f) ||
      ev.agent_role?.toLowerCase().includes(f) ||
      ev.status?.toLowerCase().includes(f) ||
      ev.event_id?.toLowerCase().includes(f)
    );
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Clinical Audit Trail</h3>
          <span
            style={{
              fontSize: '11px',
              padding: '2px 8px',
              borderRadius: '999px',
              background: 'rgba(255,255,255,0.1)',
            }}
          >
            {events.length} events
          </span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <input
            type="text"
            className="input-field"
            id="audit-filter-input"
            placeholder="Filter actions or roles..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ fontSize: '12px', padding: '4px 8px' }}
          />
          <button
            className="btn btn-secondary"
            id="refresh-audit-btn"
            onClick={loadAudit}
            disabled={loading}
            style={{ fontSize: '12px' }}
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>

      <div
        className="glass-card"
        id="audit-table-container"
        style={{ flex: 1, overflowY: 'auto', padding: 0 }}
      >
        <table
          id="audit-table"
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '12px',
            textAlign: 'left',
          }}
        >
          <thead>
            <tr
              style={{
                borderBottom: '1px solid var(--border-subtle)',
                background: 'rgba(17, 24, 39, 0.9)',
                position: 'sticky',
                top: 0,
                color: 'var(--text-secondary)',
              }}
            >
              <th style={{ padding: '8px 12px' }}>Timestamp</th>
              <th style={{ padding: '8px 12px' }}>Agent Role</th>
              <th style={{ padding: '8px 12px' }}>Action</th>
              <th style={{ padding: '8px 12px' }}>Patient</th>
              <th style={{ padding: '8px 12px' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {filteredEvents.length === 0 ? (
              <tr>
                <td
                  colSpan={5}
                  style={{
                    padding: '24px',
                    textAlign: 'center',
                    color: 'var(--text-muted)',
                  }}
                >
                  No audit events recorded for this patient.
                </td>
              </tr>
            ) : (
              filteredEvents.map((ev) => (
                <tr
                  key={ev.event_id}
                  style={{
                    borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                    transition: 'background 0.15s ease',
                  }}
                >
                  <td
                    style={{
                      padding: '8px 12px',
                      color: 'var(--text-muted)',
                      fontFamily: 'monospace',
                      fontSize: '11px',
                    }}
                  >
                    {ev.timestamp}
                  </td>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{ev.agent_role}</td>
                  <td
                    style={{
                      padding: '8px 12px',
                      fontFamily: 'monospace',
                      color: 'var(--accent-cyan)',
                    }}
                  >
                    {ev.action}
                  </td>
                  <td style={{ padding: '8px 12px', color: 'var(--text-secondary)' }}>
                    {ev.patient_id}
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <span
                      style={{
                        padding: '2px 6px',
                        borderRadius: '4px',
                        fontSize: '10px',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        background:
                          ev.status === 'success' || ev.status === 'ok'
                            ? 'rgba(16, 185, 129, 0.15)'
                            : 'rgba(239, 68, 68, 0.15)',
                        color:
                          ev.status === 'success' || ev.status === 'ok'
                            ? 'var(--accent-emerald)'
                            : 'var(--accent-rose)',
                      }}
                    >
                      {ev.status}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
