import React, { useState, useEffect, useCallback } from 'react';
import {
  fetchPatients,
  fetchPatientGraph,
  fetchAnswerTrace,
} from './api';
import type {
  PatientSummary,
  GraphPayload,
  CytoscapeNodeData,
  TraceData,
} from './api';
import { GraphCanvas } from './components/GraphCanvas';
import { NodeInspector } from './components/NodeInspector';
import { AgentConsole } from './components/AgentConsole';
import { WhyNotDrawer } from './components/WhyNotDrawer';
import { HandoffBoard } from './components/HandoffBoard';
import { AuditTab } from './components/AuditTab';
import { MetricsPanel } from './components/MetricsPanel';

export const App: React.FC = () => {
  // Navigation & selection state
  const [activeTab, setActiveTab] = useState<
    'graph' | 'agent' | 'handoffs' | 'audit' | 'metrics'
  >('graph');
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [asOfDate, setAsOfDate] = useState<string>('');

  // Graph and trace state
  const [graphData, setGraphData] = useState<GraphPayload | null>(null);
  const [selectedNode, setSelectedNode] = useState<CytoscapeNodeData | null>(null);
  const [activeTrace, setActiveTrace] = useState<TraceData | null>(null);
  const [recentAnswers, setRecentAnswers] = useState<Array<{ id: string; summary: string }>>([]);
  const [selectedAnswerId, setSelectedAnswerId] = useState<string>('');
  const [manualAnswerId, setManualAnswerId] = useState<string>('');

  // Why-not drawer state
  const [whyNotCandidateId, setWhyNotCandidateId] = useState<string>('');
  const [showWhyNot, setShowWhyNot] = useState(false);

  // Load all patients on mount or refresh
  const loadPatients = useCallback(async () => {
    try {
      const list = await fetchPatients('');
      setPatients(list);
      if (list.length > 0) {
        setSelectedPatientId((curr) => {
          if (!curr || !list.some((p) => p.patient_id === curr)) {
            const john = list.find((p) =>
              (p.name || '').toLowerCase().includes('john doe')
            );
            return john ? john.patient_id : list[0].patient_id;
          }
          return curr;
        });
      }
    } catch (err) {
      console.error('Failed to load patients:', err);
    }
  }, []);

  useEffect(() => {
    loadPatients();
  }, [loadPatients]);

  // Derived filtered patients list based on search term
  const filteredPatients = patients.filter((p) => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return true;
    return (
      (p.name || '').toLowerCase().includes(q) ||
      (p.patient_id || '').toLowerCase().includes(q)
    );
  });

  const handleSearchChange = (query: string) => {
    setSearchQuery(query);
    const q = query.trim().toLowerCase();
    if (!q) return;
    const match = patients.find(
      (p) =>
        (p.patient_id || '').toLowerCase() === q ||
        (p.name || '').toLowerCase() === q ||
        (p.name || '').toLowerCase().includes(q) ||
        (p.patient_id || '').toLowerCase().includes(q)
    );
    if (match) {
      setSelectedPatientId(match.patient_id);
    }
  };

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      const q = searchQuery.trim();
      const match = patients.find(
        (p) =>
          (p.patient_id || '').toLowerCase() === q.toLowerCase() ||
          (p.name || '').toLowerCase().includes(q.toLowerCase()) ||
          (p.patient_id || '').toLowerCase().includes(q.toLowerCase())
      );
      if (match) {
        setSelectedPatientId(match.patient_id);
      } else if (q.startsWith('p_') || q.length > 3) {
        setSelectedPatientId(q);
      }
    }
  };

  // Load graph when patient or as-of changes
  const loadGraph = useCallback(async () => {
    if (!selectedPatientId) return;
    try {
      const data = await fetchPatientGraph(selectedPatientId, asOfDate || undefined);
      setGraphData(data);
      setSelectedNode(null);
    } catch (err) {
      console.error('Failed to load patient graph:', err);
    }
  }, [selectedPatientId, asOfDate]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  // Handle Answer selection for trace highlight
  const handleSelectAnswer = async (ansId: string) => {
    setSelectedAnswerId(ansId);
    if (!ansId) {
      setActiveTrace(null);
      return;
    }
    try {
      const trace = await fetchAnswerTrace(ansId);
      setActiveTrace(trace);
    } catch (err) {
      console.error('Failed to fetch trace for answer:', err);
    }
  };

  const handleManualTrace = async () => {
    if (!manualAnswerId.trim()) return;
    await handleSelectAnswer(manualAnswerId.trim());
  };

  const handleWhyNotFromNode = (nodeId: string) => {
    setWhyNotCandidateId(nodeId);
    setShowWhyNot(true);
  };

  const handleAnswerCreated = (answerData: any) => {
    if (answerData?.answer_id) {
      const newAns = {
        id: answerData.answer_id,
        summary: answerData.diagnosis || answerData.recommendation || 'Clinical recommendation',
      };
      setRecentAnswers((prev) => [newAns, ...prev.filter((a) => a.id !== newAns.id)]);
      setSelectedAnswerId(newAns.id);
    }
  };

  return (
    <div className="app-wrapper">
      {/* 1. Mandatory Clinical Safety Banner */}
      <div className="safety-banner" id="clinical-safety-banner">
        <span>[SYNTHETIC DATA, NOT FOR CLINICAL USE]</span>
        <span style={{ opacity: 0.85, fontWeight: 500 }}>
          Simulated AI Memory Demonstration Platform
        </span>
      </div>

      {/* 2. Brand & Primary Navigation */}
      <header className="app-header" id="app-header">
        <div className="brand-section">
          <div className="brand-logo">
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="white"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </div>
          <div>
            <div className="brand-title">CareThread</div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
              Clinic Visit Memory &amp; Multi-Agent Provenance
            </div>
          </div>
          <span className="brand-badge">Bitemporal Graph</span>
        </div>

        <nav className="tabs-navigation" id="main-navigation">
          <button
            className={`tab-btn ${activeTab === 'graph' ? 'active' : ''}`}
            id="tab-graph"
            onClick={() => setActiveTab('graph')}
          >
            Graph &amp; Provenance
          </button>
          <button
            className={`tab-btn ${activeTab === 'agent' ? 'active' : ''}`}
            id="tab-agent"
            onClick={() => setActiveTab('agent')}
          >
            Agent Console
          </button>
          <button
            className={`tab-btn ${activeTab === 'handoffs' ? 'active' : ''}`}
            id="tab-handoffs"
            onClick={() => setActiveTab('handoffs')}
          >
            Handoffs &amp; Safety
          </button>
          <button
            className={`tab-btn ${activeTab === 'audit' ? 'active' : ''}`}
            id="tab-audit"
            onClick={() => setActiveTab('audit')}
          >
            Audit Trail
          </button>
          <button
            className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
            id="tab-metrics"
            onClick={() => setActiveTab('metrics')}
          >
            Metrics
          </button>
        </nav>
      </header>

      {/* 3. Patient Picker & Temporal Slider Bar */}
      <div className="control-bar" id="control-bar">
        <div className="patient-picker-group">
          <input
            type="text"
            className="input-field"
            id="patient-search-input"
            placeholder="Search Name or Patient ID (e.g. p_2885b7c1d9a0)..."
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            style={{ width: '260px' }}
          />

          <select
            className="input-field"
            id="patient-select"
            value={selectedPatientId}
            onChange={(e) => setSelectedPatientId(e.target.value)}
            style={{ minWidth: '220px' }}
          >
            {filteredPatients.length === 0 && (
              <option value="">No matching patients</option>
            )}
            {filteredPatients.map((p) => (
              <option key={p.patient_id} value={p.patient_id}>
                {p.name || 'Unknown'} ({p.patient_id})
              </option>
            ))}
          </select>

          {/* Quick Demo Patient Selectors */}
          <div
            className="quick-patient-chips"
            style={{ display: 'flex', gap: '6px', alignItems: 'center' }}
          >
            {[
              { name: 'John Doe', id: 'p_2885b7c1d9a0' },
              { name: 'Jane Smith', id: 'p_3bdc39be7d69' },
              { name: 'Alice Williams', id: 'p_a1cdd3689525' },
            ].map((demo) => {
              const isSelected = selectedPatientId === demo.id;
              return (
                <button
                  key={demo.id}
                  type="button"
                  className="btn"
                  onClick={() => {
                    setSelectedPatientId(demo.id);
                    setSearchQuery('');
                  }}
                  style={{
                    fontSize: '11px',
                    padding: '3px 8px',
                    borderRadius: '999px',
                    backgroundColor: isSelected
                      ? 'var(--accent-blue)'
                      : 'rgba(255, 255, 255, 0.08)',
                    color: isSelected ? '#ffffff' : 'var(--text-secondary)',
                    border: isSelected
                      ? '1px solid var(--accent-blue)'
                      : '1px solid var(--border-subtle)',
                    cursor: 'pointer',
                  }}
                  title={`Select ${demo.name} (${demo.id})`}
                >
                  {demo.name}
                </button>
              );
            })}
          </div>

          {patients.find((p) => p.patient_id === selectedPatientId) && (
            <span
              id="active-patient-badge"
              style={{
                fontSize: '11px',
                padding: '4px 10px',
                borderRadius: '999px',
                background: 'rgba(59, 130, 246, 0.15)',
                border: '1px solid rgba(59, 130, 246, 0.35)',
                color: '#93c5fd',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <strong>
                {patients.find((p) => p.patient_id === selectedPatientId)?.name}
              </strong>
              <span style={{ opacity: 0.6 }}>|</span>
              <span style={{ fontFamily: 'monospace' }}>
                {selectedPatientId}
              </span>
            </span>
          )}

          <button
            className="btn btn-secondary"
            id="refresh-btn"
            onClick={() => {
              loadPatients();
              loadGraph();
            }}
            title="Refresh current graph"
          >
            Refresh
          </button>
        </div>

        {/* As-Of Timeline Slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            As-Of Date:
          </span>
          <input
            type="date"
            className="input-field"
            id="as-of-slider"
            value={asOfDate}
            onChange={(e) => setAsOfDate(e.target.value)}
            style={{ fontSize: '12px', padding: '4px 8px' }}
          />
          {asOfDate && (
            <button
              className="btn btn-secondary"
              id="as-of-reset-btn"
              onClick={() => setAsOfDate('')}
              style={{ fontSize: '11px', padding: '4px 8px' }}
            >
              Reset to Live
            </button>
          )}
          <span
            id="as-of-display"
            style={{
              fontSize: '11px',
              padding: '2px 8px',
              borderRadius: '999px',
              background: asOfDate ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)',
              color: asOfDate ? 'var(--accent-amber)' : 'var(--accent-emerald)',
              fontWeight: 600,
            }}
          >
            {asOfDate ? `As-Of: ${asOfDate}` : 'LIVE'}
          </span>
        </div>
      </div>

      {/* 4. Active Tab Content Container */}
      <main className="app-container" id="app-container">
        {activeTab === 'graph' && (
          <>
            <GraphCanvas
              graphData={graphData}
              selectedAnswerTrace={activeTrace}
              onNodeSelect={setSelectedNode}
            />

            <aside className="side-panel" id="graph-side-panel">
              <div className="panel-header">
                <div className="panel-title">Provenance &amp; Inspector</div>
                {activeTrace && (
                  <button
                    className="btn btn-secondary"
                    id="clear-trace-btn"
                    onClick={() => {
                      setActiveTrace(null);
                      setSelectedAnswerId('');
                    }}
                    style={{ fontSize: '11px', padding: '2px 8px' }}
                  >
                    Clear Highlight
                  </button>
                )}
              </div>

              <div className="panel-body">
                {/* Answer Selector for Subgraph Highlighting */}
                <div className="glass-card" id="trace-controller-card">
                  <h4
                    style={{
                      fontSize: '13px',
                      color: 'var(--accent-cyan)',
                      marginBottom: '8px',
                    }}
                  >
                    Trace Subgraph Highlighting
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <select
                      className="input-field"
                      id="answer-trace-selector"
                      value={selectedAnswerId}
                      onChange={(e) => handleSelectAnswer(e.target.value)}
                    >
                      <option value="">-- Choose Clinical Answer to Highlight --</option>
                      {recentAnswers.map((a) => (
                        <option key={a.id} value={a.id}>
                          {(a.summary || '').slice(0, 36)}... ({(a.id || '').slice(0, 8)})
                        </option>
                      ))}
                    </select>

                    <div style={{ display: 'flex', gap: '6px' }}>
                      <input
                        type="text"
                        className="input-field"
                        id="manual-answer-input"
                        placeholder="Or enter Answer ID directly..."
                        value={manualAnswerId}
                        onChange={(e) => setManualAnswerId(e.target.value)}
                        style={{ flex: 1, fontSize: '11px', fontFamily: 'monospace' }}
                      />
                      <button
                        className="btn btn-secondary"
                        id="load-trace-btn"
                        onClick={handleManualTrace}
                        style={{ fontSize: '11px' }}
                      >
                        Highlight
                      </button>
                    </div>
                  </div>

                  {activeTrace && (
                    <div
                      id="active-trace-summary"
                      style={{
                        marginTop: '10px',
                        padding: '8px',
                        background: 'rgba(0,0,0,0.25)',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '11px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '4px',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>
                          Used Nodes ({activeTrace.used?.length || 0}):
                        </span>
                        <span>Solid Border</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--accent-rose)', fontWeight: 600 }}>
                          Rejected ({activeTrace.rejected?.length || 0}):
                        </span>
                        <span>Red Dashed</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--accent-amber)', fontWeight: 600 }}>
                          Stale ({activeTrace.stale_nodes?.length || 0}):
                        </span>
                        <span>Amber Dotted</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Confidence:</span>
                        <span>{(activeTrace.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Node Inspector */}
                <NodeInspector
                  nodeData={selectedNode}
                  onClose={() => setSelectedNode(null)}
                  onWhyNotQuery={handleWhyNotFromNode}
                />

                {/* Why-Not Drawer toggle & panel */}
                <div className="glass-card">
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <h4 style={{ fontSize: '13px', color: 'var(--accent-rose)' }}>
                      Why-Not Analyzer
                    </h4>
                    <button
                      className="btn btn-secondary"
                      id="toggle-whynot-btn"
                      onClick={() => setShowWhyNot(!showWhyNot)}
                      style={{ fontSize: '11px', padding: '2px 8px' }}
                    >
                      {showWhyNot ? 'Hide Drawer' : 'Open Drawer'}
                    </button>
                  </div>

                  {showWhyNot && (
                    <div style={{ marginTop: '10px' }}>
                      <WhyNotDrawer
                        currentAnswerId={selectedAnswerId || manualAnswerId}
                        selectedNodeId={whyNotCandidateId}
                        onClose={() => setShowWhyNot(false)}
                      />
                    </div>
                  )}
                </div>
              </div>
            </aside>
          </>
        )}

        {activeTab === 'agent' && (
          <div style={{ flex: 1, padding: '20px', height: '100%', overflowY: 'auto' }}>
            <AgentConsole
              patientId={selectedPatientId}
              onTraceReceived={(trace) => {
                setActiveTrace(trace);
                setSelectedAnswerId(trace.answer_id);
              }}
              onAnswerCreated={handleAnswerCreated}
            />
          </div>
        )}

        {activeTab === 'handoffs' && (
          <div style={{ flex: 1, padding: '20px', height: '100%', overflowY: 'auto' }}>
            <HandoffBoard patientId={selectedPatientId} />
          </div>
        )}

        {activeTab === 'audit' && (
          <div style={{ flex: 1, padding: '20px', height: '100%', overflowY: 'auto' }}>
            <AuditTab patientId={selectedPatientId} />
          </div>
        )}

        {activeTab === 'metrics' && (
          <div style={{ flex: 1, padding: '20px', height: '100%', overflowY: 'auto' }}>
            <MetricsPanel />
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
