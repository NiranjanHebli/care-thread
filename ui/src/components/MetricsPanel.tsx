import React, { useState, useEffect, useCallback } from 'react';
import { fetchMetricsText } from '../api';

export const MetricsPanel: React.FC = () => {
  const [rawMetrics, setRawMetrics] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const loadMetrics = useCallback(async () => {
    setLoading(true);
    try {
      const text = await fetchMetricsText();
      setRawMetrics(text);
    } catch (err) {
      console.error('Failed to load /metrics:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMetrics();
    if (!autoRefresh) return;
    const interval = setInterval(loadMetrics, 4000);
    return () => clearInterval(interval);
  }, [loadMetrics, autoRefresh]);

  // Parse lines for quick stats
  const metricLines = rawMetrics.split('\n').filter((l) => l && !l.startsWith('#'));

  return (
    <div
      style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '12px' }}
      id="metrics-dashboard"
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h3 style={{ fontSize: '15px', fontWeight: 600 }}>System &amp; Benchmark Metrics</h3>
          <span
            style={{
              fontSize: '11px',
              padding: '2px 8px',
              borderRadius: '999px',
              background: 'rgba(16, 185, 129, 0.15)',
              color: 'var(--accent-emerald)',
            }}
          >
            Live Prometheus Source
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label
            style={{
              fontSize: '12px',
              color: 'var(--text-secondary)',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <input
              type="checkbox"
              id="metrics-auto-refresh"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh (4s)
          </label>
          <button
            className="btn btn-secondary"
            id="refresh-metrics-btn"
            onClick={loadMetrics}
            disabled={loading}
            style={{ fontSize: '12px' }}
          >
            {loading ? 'Refreshing...' : 'Poll Metrics'}
          </button>
        </div>
      </div>

      <div
        className="glass-card"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '12px',
        }}
      >
        <div className="glass-card" style={{ background: '#0a0e17' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Total Metric Records</div>
          <div style={{ fontSize: '20px', fontWeight: 700, marginTop: '4px' }}>
            {metricLines.length}
          </div>
        </div>
        <div className="glass-card" style={{ background: '#0a0e17' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Status</div>
          <div
            style={{
              fontSize: '20px',
              fontWeight: 700,
              color: 'var(--accent-emerald)',
              marginTop: '4px',
            }}
          >
            ACTIVE
          </div>
        </div>
      </div>

      <div
        className="glass-card"
        style={{ flex: 1, overflowY: 'auto', background: '#070b12', padding: '12px' }}
      >
        <div
          style={{
            fontSize: '11px',
            fontWeight: 600,
            color: 'var(--text-secondary)',
            marginBottom: '8px',
          }}
        >
          Raw Prometheus Stream (/metrics):
        </div>
        <pre
          id="raw-metrics-pre"
          style={{
            fontFamily: 'monospace',
            fontSize: '11px',
            lineHeight: '1.4',
            color: '#a7f3d0',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
          }}
        >
          {rawMetrics || 'Loading Prometheus metrics...'}
        </pre>
      </div>
    </div>
  );
};
