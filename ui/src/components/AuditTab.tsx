import React, { useState, useEffect, useCallback } from 'react';
import { fetchAuditLogs } from '../api';
import type { AuditEvent } from '../api';

interface AuditTabProps {
  patientId: string;
}

interface ActionBadge {
  label: string;
  badgeStyle: React.CSSProperties;
}

const formatAuditAction = (actionStr: string): ActionBadge => {
  const raw = String(actionStr || '').toLowerCase();

  switch (raw) {
    case 'list_handoffs':
      return {
        label: 'Review Handoffs',
        badgeStyle: {
          background: 'rgba(245, 158, 11, 0.12)',
          color: '#fbbf24',
          border: '1px solid rgba(245, 158, 11, 0.28)',
        },
      };
    case 'list_flags':
      return {
        label: 'Check Safety Flags',
        badgeStyle: {
          background: 'rgba(244, 63, 94, 0.12)',
          color: '#fb7185',
          border: '1px solid rgba(244, 63, 94, 0.28)',
        },
      };
    case 'raise_flag':
      return {
        label: 'Raise Safety Flag',
        badgeStyle: {
          background: 'rgba(239, 68, 68, 0.16)',
          color: '#f87171',
          border: '1px solid rgba(239, 68, 68, 0.35)',
        },
      };
    case 'resolve_flag':
      return {
        label: 'Resolve Safety Flag',
        badgeStyle: {
          background: 'rgba(16, 185, 129, 0.12)',
          color: '#34d399',
          border: '1px solid rgba(16, 185, 129, 0.3)',
        },
      };
    case 'create_handoff':
      return {
        label: 'Create Handoff Packet',
        badgeStyle: {
          background: 'rgba(249, 115, 22, 0.12)',
          color: '#fb923c',
          border: '1px solid rgba(249, 115, 22, 0.3)',
        },
      };
    case 'accept_handoff':
      return {
        label: 'Accept Handoff',
        badgeStyle: {
          background: 'rgba(245, 158, 11, 0.12)',
          color: '#fcd34d',
          border: '1px solid rgba(245, 158, 11, 0.28)',
        },
      };
    case 'complete_handoff':
      return {
        label: 'Complete Handoff',
        badgeStyle: {
          background: 'rgba(16, 185, 129, 0.12)',
          color: '#34d399',
          border: '1px solid rgba(16, 185, 129, 0.3)',
        },
      };
    case 'query_memory':
    case 'recall':
      return {
        label: 'Recall Graph Memory',
        badgeStyle: {
          background: 'rgba(14, 165, 233, 0.12)',
          color: '#38bdf8',
          border: '1px solid rgba(14, 165, 233, 0.28)',
        },
      };
    case 'assert_fact':
    case 'remember':
      return {
        label: 'Record Clinical Fact',
        badgeStyle: {
          background: 'rgba(16, 185, 129, 0.14)',
          color: '#10b981',
          border: '1px solid rgba(16, 185, 129, 0.3)',
        },
      };
    case 'supersede_fact':
      return {
        label: 'Supersede Fact (CAS)',
        badgeStyle: {
          background: 'rgba(245, 158, 11, 0.14)',
          color: '#f59e0b',
          border: '1px solid rgba(245, 158, 11, 0.3)',
        },
      };
    case 'chat_workflow':
    case 'workflow':
      return {
        label: 'Agent Workflow Session',
        badgeStyle: {
          background: 'rgba(147, 51, 234, 0.12)',
          color: '#c084fc',
          border: '1px solid rgba(147, 51, 234, 0.3)',
        },
      };
    case 'start_session':
      return {
        label: 'Start Visit Session',
        badgeStyle: {
          background: 'rgba(99, 102, 241, 0.12)',
          color: '#a5b4fc',
          border: '1px solid rgba(99, 102, 241, 0.28)',
        },
      };
    case 'append_message':
      return {
        label: 'Record Turn Message',
        badgeStyle: {
          background: 'rgba(100, 116, 139, 0.12)',
          color: '#cbd5e1',
          border: '1px solid rgba(100, 116, 139, 0.25)',
        },
      };
    default: {
      const words = raw.replace(/_/g, ' ').split(' ');
      const formatted = words.map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
      return {
        label: formatted,
        badgeStyle: {
          background: 'rgba(56, 189, 248, 0.08)',
          color: '#38bdf8',
          border: '1px solid rgba(56, 189, 248, 0.2)',
        },
      };
    }
  }
};

const formatAgentRole = (role: string): { label: string; badgeStyle: React.CSSProperties } => {
  const r = String(role || '').toLowerCase();
  if (r.includes('intake')) {
    return {
      label: 'Intake Agent',
      badgeStyle: {
        background: 'rgba(67, 56, 202, 0.15)',
        color: '#a5b4fc',
        border: '1px solid rgba(99, 102, 241, 0.3)',
      },
    };
  }
  if (r.includes('specialist')) {
    return {
      label: 'Specialist Agent',
      badgeStyle: {
        background: 'rgba(79, 70, 229, 0.15)',
        color: '#c7d2fe',
        border: '1px solid rgba(99, 102, 241, 0.3)',
      },
    };
  }
  if (r.includes('pharmacy')) {
    return {
      label: 'Pharmacy Agent',
      badgeStyle: {
        background: 'rgba(5, 150, 105, 0.15)',
        color: '#6ee7b7',
        border: '1px solid rgba(16, 185, 129, 0.3)',
      },
    };
  }
  if (r === 'clinician') {
    return {
      label: 'Clinician',
      badgeStyle: {
        background: 'rgba(2, 132, 199, 0.15)',
        color: '#38bdf8',
        border: '1px solid rgba(2, 132, 199, 0.3)',
      },
    };
  }
  return {
    label: role ? role.charAt(0).toUpperCase() + role.slice(1) : 'System',
    badgeStyle: {
      background: 'rgba(100, 116, 139, 0.12)',
      color: '#cbd5e1',
      border: '1px solid rgba(100, 116, 139, 0.25)',
    },
  };
};

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
    const actionMeta = formatAuditAction(ev.action);
    const roleMeta = formatAgentRole(ev.agent_role);
    return (
      ev.action?.toLowerCase().includes(f) ||
      actionMeta.label.toLowerCase().includes(f) ||
      ev.agent_role?.toLowerCase().includes(f) ||
      roleMeta.label.toLowerCase().includes(f) ||
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
                  <td style={{ padding: '8px 12px' }}>
                    <span
                      style={{
                        display: 'inline-block',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontWeight: 600,
                        ...formatAgentRole(ev.agent_role).badgeStyle,
                      }}
                    >
                      {formatAgentRole(ev.agent_role).label}
                    </span>
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <span
                      style={{
                        display: 'inline-block',
                        padding: '3px 8px',
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontWeight: 500,
                        ...formatAuditAction(ev.action).badgeStyle,
                      }}
                    >
                      {formatAuditAction(ev.action).label}
                    </span>
                  </td>
                  <td style={{ padding: '8px 12px', color: 'var(--text-secondary)' }}>
                    <span style={{ fontFamily: 'monospace', fontSize: '11px' }}>
                      {ev.patient_id}
                    </span>
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
