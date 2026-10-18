import React, { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
import type { Core, EventObject } from 'cytoscape';
import type { CytoscapeNodeData, GraphPayload } from '../api';

interface GraphCanvasProps {
  graphData: GraphPayload | null;
  selectedAnswerTrace?: {
    used: string[];
    rejected: string[];
    stale_nodes: string[];
  } | null;
  onNodeSelect: (nodeData: CytoscapeNodeData | null) => void;
}

/**
 * Maps raw Cypher relationship types to human-readable clinical labels.
 */
const formatEdgeLabel = (relType: string): string => {
  const edgeLabelMap: Record<string, string> = {
    HAS_FACT: 'has',
    ASSERTED_IN: 'documented in',
    HAS_EPISODE: 'visit',
    WITH_PROVIDER: 'treated by',
    AT_ORG: 'facility',
    NEXT: 'followed by',
    PRODUCED: 'observed',
    PERFORMED: 'procedure',
    HAS_SESSION: 'session',
    HAS_MESSAGE: 'message',
    USED: 'cited in answer',
    REJECTED: 'contraindicated',
    FLAGGED: 'safety flag',
    HANDOFF_TO: 'handoff to',
    SUPERSEDES: 'supersedes',
    DERIVED_FROM: 'derived from',
    CONTRADICTS: 'contradicts',
    FROM: 'from agent',
    TO: 'to agent',
    CARRIES: 'carries order',
    HAS_QUESTION: 'question',
    HAS_ACTION: 'action',
    HAS_NOTE: 'note',
    HAS_HANDOFF: 'handoff',
    ACCEPTED_IN: 'accepted in',
    CREATED_IN: 'created in',
  };
  return edgeLabelMap[relType] || relType.toLowerCase().replace(/_/g, ' ');
};

/**
 * Maps raw node identifiers and technical IDs into friendly clinical titles.
 */
const formatNodeDisplay = (data: Record<string, any>): string => {
  const nodeType = data.type || data.label || 'Node';
  const rawId = typeof data.id === 'string' ? data.id : '';

  // 1. Format Agent participants: agt_intake_01 -> "Intake Agent"
  if (nodeType === 'Agent' || rawId.startsWith('agt_')) {
    if (rawId.includes('intake') || data.role === 'intake') return 'Intake Agent';
    if (rawId.includes('specialist') || data.role === 'specialist') return 'Specialist Agent';
    if (rawId.includes('pharmacy') || data.role === 'pharmacy') return 'Pharmacy Agent';
    if (rawId.includes('triage') || data.role === 'triage') return 'Triage Agent';
    const roleMatch = rawId.match(/^agt_([a-zA-Z]+)/);
    if (roleMatch) {
      const r = roleMatch[1];
      return `${r.charAt(0).toUpperCase() + r.slice(1)} Agent`;
    }
    return 'Clinical Agent';
  }

  // 2. Format clinical episodes / encounters: epi_enc-01 -> "Visit 1 (Ambulatory)"
  if (nodeType === 'Episode' || rawId.startsWith('epi_')) {
    const seq = data.seq || rawId.match(/enc-0?(\d+)/)?.[1] || '';
    const rawType = data.episode_type || '';
    const epType = rawType ? rawType.charAt(0).toUpperCase() + rawType.slice(1) : '';
    if (seq && epType) {
      return `Visit ${seq} (${epType})`;
    }
    if (seq) {
      return `Visit ${seq}`;
    }
    if (epType) {
      return `${epType} Visit`;
    }
    return 'Clinic Visit';
  }

  // 3. Format handoffs
  if (nodeType === 'Handoff' || rawId.startsWith('hnd_')) {
    const role = data.to_role ? String(data.to_role).replace(/_/g, ' ') : '';
    return role ? `Handoff to ${role}` : 'Handoff';
  }

  // 4. Format Action nodes: act_... -> Action description
  if (nodeType === 'Action' || rawId.startsWith('act_')) {
    const desc = data.description || data.text || data.action || '';
    if (typeof desc === 'string' && desc.trim()) {
      return desc.length > 35 ? `${desc.slice(0, 32)}...` : desc;
    }
    return 'Action Item';
  }

  // 5. Format OpenQuestion nodes: qst_... -> Question text
  if (nodeType === 'OpenQuestion' || rawId.startsWith('qst_')) {
    const q = data.question || data.text || '';
    if (typeof q === 'string' && q.trim()) {
      return q.length > 35 ? `${q.slice(0, 32)}...` : q;
    }
    return 'Clinical Question';
  }

  // 6. Format Session nodes: ses_... -> "Clinical Session"
  if (nodeType === 'Session' || rawId.startsWith('ses_')) {
    return 'Clinical Session';
  }

  // 7. Format Message nodes: msg_... -> Message snippet
  if (nodeType === 'Message' || rawId.startsWith('msg_')) {
    const content = data.content || '';
    if (typeof content === 'string' && content.trim()) {
      return content.length > 30 ? `${content.slice(0, 27)}...` : content;
    }
    const role = data.role ? String(data.role).replace(/_/g, ' ') : '';
    return role ? `${role.charAt(0).toUpperCase() + role.slice(1)} Message` : 'Message';
  }

  // 8. Format Flags: flg_... -> Safety Flag
  if (nodeType === 'Flag' || rawId.startsWith('flg_')) {
    return data.category || 'Safety Flag';
  }

  // 9. Format Notes: not_... -> Clinical Note
  if (nodeType === 'Note' || rawId.startsWith('not_')) {
    const title = data.title || data.subject;
    return typeof title === 'string' && title ? title : 'Clinical Note';
  }

  const currentDisplay = data.display || data.name || data.text;
  if (typeof currentDisplay === 'string') {
    const epMatch = currentDisplay.match(/^epi_enc-0?(\d+)$/);
    if (epMatch) {
      return `Visit ${epMatch[1]}`;
    }
    return currentDisplay;
  }

  return rawId || 'Node';
};

export const GraphCanvas: React.FC<GraphCanvasProps> = ({
  graphData,
  selectedAnswerTrace,
  onNodeSelect,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const cy = cytoscape({
      container: containerRef.current,
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(display)',
            'color': '#f8fafc',
            'font-size': '11px',
            'text-valign': 'center',
            'text-halign': 'center',
            'text-outline-color': '#0a0e17',
            'text-outline-width': '2px',
            'background-color': '#334155',
            'border-width': '2px',
            'border-color': '#64748b',
            width: '46px',
            height: '46px',
            'text-max-width': '95px',
            'text-wrap': 'wrap',
            'transition-property': 'background-color, border-color, opacity',
            'transition-duration': 0.2,
          },
        },
        {
          selector: 'node[type = "Patient"]',
          style: {
            'background-color': '#2563eb',
            'border-color': '#60a5fa',
            shape: 'ellipse',
            width: '58px',
            height: '58px',
            'font-size': '12px',
            'font-weight': 'bold',
          },
        },
        {
          selector: 'node[type = "Session"]',
          style: {
            'background-color': '#7c3aed',
            'border-color': '#a78bfa',
            shape: 'hexagon',
            width: '46px',
            height: '46px',
          },
        },
        {
          selector: 'node[type = "Episode"]',
          style: {
            'background-color': '#0d9488',
            'border-color': '#2dd4bf',
            shape: 'round-rectangle',
            width: '58px',
            height: '58px',
            'font-size': '10px',
            'font-weight': 'bold',
          },
        },
        {
          selector: 'node[type = "Fact"]',
          style: {
            'background-color': '#059669',
            'border-color': '#34d399',
            shape: 'diamond',
            width: '46px',
            height: '46px',
            'font-size': '10px',
          },
        },
        {
          selector: 'node[type = "Observation"]',
          style: {
            'background-color': '#0284c7',
            'border-color': '#38bdf8',
            shape: 'round-rectangle',
            width: '40px',
            height: '40px',
          },
        },
        {
          selector: 'node[type = "Answer"]',
          style: {
            'background-color': '#9333ea',
            'border-color': '#c084fc',
            shape: 'star',
            width: '54px',
            height: '54px',
          },
        },
        {
          selector: 'node[type = "Handoff"]',
          style: {
            'background-color': '#ea580c',
            'border-color': '#fb923c',
            shape: 'vee',
            width: '46px',
            height: '46px',
          },
        },
        {
          selector: 'node[type = "Flag"]',
          style: {
            'background-color': '#e11d48',
            'border-color': '#fb7185',
            shape: 'triangle',
            width: '46px',
            height: '46px',
          },
        },
        {
          selector: 'node[type = "Organization"]',
          style: {
            'background-color': '#4f46e5',
            'border-color': '#818cf8',
            shape: 'round-rectangle',
            width: '46px',
            height: '46px',
          },
        },
        {
          selector: 'node[type = "Provider"]',
          style: {
            'background-color': '#0284c7',
            'border-color': '#38bdf8',
            shape: 'ellipse',
            width: '44px',
            height: '44px',
          },
        },
        {
          selector: 'node[type = "Agent"]',
          style: {
            'background-color': '#4338ca',
            'border-color': '#818cf8',
            shape: 'round-rectangle',
            width: '56px',
            height: '56px',
            'font-size': '10px',
            'font-weight': 'bold',
          },
        },
        {
          selector: 'node[type = "Action"]',
          style: {
            'background-color': '#0e7490',
            'border-color': '#38bdf8',
            shape: 'round-rectangle',
            width: '48px',
            height: '48px',
            'font-size': '9px',
          },
        },
        {
          selector: 'node[type = "OpenQuestion"]',
          style: {
            'background-color': '#b45309',
            'border-color': '#fbbf24',
            shape: 'round-rectangle',
            width: '48px',
            height: '48px',
            'font-size': '9px',
          },
        },
        {
          selector: 'node[type = "Message"]',
          style: {
            'background-color': '#334155',
            'border-color': '#94a3b8',
            shape: 'round-rectangle',
            width: '44px',
            height: '44px',
            'font-size': '9px',
          },
        },
        {
          selector: 'node[type = "Note"]',
          style: {
            'background-color': '#047857',
            'border-color': '#34d399',
            shape: 'round-rectangle',
            width: '48px',
            height: '48px',
            'font-size': '9px',
          },
        },
        {
          selector: 'node[?is_clinic]',
          style: {
            'background-color': '#d97706',
            'border-style': 'dashed',
            'border-width': '2.5px',
            'border-color': '#fde68a',
            shape: 'rectangle',
          },
        },
        {
          selector: 'edge',
          style: {
            width: 2,
            'line-color': '#475569',
            'target-arrow-color': '#64748b',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            label: 'data(label)',
            'font-size': '9px',
            color: '#cbd5e1',
            'text-rotation': 'autorotate',
            'text-background-opacity': 0.88,
            'text-background-color': '#0a0e17',
            'text-background-padding': '3px',
            'text-background-shape': 'roundrectangle',
          },
        },
        // Trace highlight classes
        {
          selector: '.used',
          style: {
            'border-color': '#10b981',
            'border-width': '4px',
            'border-style': 'solid',
            opacity: 1.0,
            'line-color': '#10b981',
            'target-arrow-color': '#10b981',
            width: 3.5,
          },
        },
        {
          selector: '.rejected',
          style: {
            'border-color': '#ef4444',
            'border-width': '4px',
            'border-style': 'dashed',
            opacity: 1.0,
            'line-color': '#ef4444',
            'target-arrow-color': '#ef4444',
            'line-style': 'dashed',
          },
        },
        {
          selector: '.stale',
          style: {
            'border-color': '#f59e0b',
            'border-width': '4px',
            'border-style': 'dotted',
            opacity: 1.0,
          },
        },
        {
          selector: '.dimmed',
          style: {
            opacity: 0.18,
          },
        },
      ],
      elements: [],
      layout: { name: 'cose', animate: false },
    });

    cy.on('tap select', 'node', (evt: EventObject) => {
      const data = evt.target.data() as CytoscapeNodeData;
      onNodeSelect(data);
    });

    cy.on('tap', (evt: EventObject) => {
      if (evt.target === cy) {
        onNodeSelect(null);
      }
    });

    cyRef.current = cy;
    (window as any)._cy = cy;

    const resizeObserver = new ResizeObserver(() => {
      cy.resize();
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      cy.destroy();
    };
  }, [onNodeSelect]);

  // Update elements when graphData changes
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !graphData) return;

    cy.elements().remove();
    const rawNodes = (graphData.elements.nodes || []).filter(
      (n) => n.data.label !== 'AuditEvent' && n.data.type !== 'AuditEvent'
    );
    const rawEdges = graphData.elements.edges || [];

    // Normalize node data for display and label matching
    const nodes = rawNodes.map((n) => {
      const d = { ...n.data };
      const nodeType = d.type || d.label || 'Node';
      return {
        ...n,
        data: {
          ...d,
          type: nodeType,
          label: nodeType,
          display: formatNodeDisplay({ ...d, type: nodeType, label: nodeType }),
        },
      };
    });

    const edges = rawEdges.map((e) => {
      const d = { ...e.data };
      const rawRel = String(d.label || d.rel_type || 'RELATED');
      return {
        ...e,
        data: {
          ...d,
          rel_type: rawRel,
          label: formatEdgeLabel(rawRel),
        },
      };
    });

    cy.add([...nodes, ...edges]);

    const layout = cy.layout({
      name: 'cose',
      animate: false,
      idealEdgeLength: () => 95,
      nodeOverlap: 25,
    });
    layout.run();
    cy.fit(undefined, 40);
  }, [graphData]);

  // Apply trace highlighting
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    cy.elements().removeClass('used rejected stale dimmed');

    if (!selectedAnswerTrace) return;

    const usedSet = new Set(selectedAnswerTrace.used || []);
    const rejectedSet = new Set(selectedAnswerTrace.rejected || []);
    const staleSet = new Set(selectedAnswerTrace.stale_nodes || []);

    const hasHighlight = usedSet.size > 0 || rejectedSet.size > 0 || staleSet.size > 0;
    if (!hasHighlight) return;

    cy.elements().addClass('dimmed');

    // Highlight used nodes
    usedSet.forEach((id) => {
      const ele = cy.getElementById(id);
      if (ele.length) {
        ele.removeClass('dimmed').addClass('used');
      }
    });

    // Highlight rejected nodes
    rejectedSet.forEach((id) => {
      const ele = cy.getElementById(id);
      if (ele.length) {
        ele.removeClass('dimmed').addClass('rejected');
      }
    });

    // Highlight stale nodes
    staleSet.forEach((id) => {
      const ele = cy.getElementById(id);
      if (ele.length) {
        ele.removeClass('dimmed').addClass('stale');
      }
    });

    // Highlight edges between used nodes
    cy.edges().forEach((edge) => {
      const src = edge.source().id();
      const tgt = edge.target().id();
      if (usedSet.has(src) && usedSet.has(tgt)) {
        edge.removeClass('dimmed').addClass('used');
      }
    });
  }, [selectedAnswerTrace]);

  const handleFit = () => cyRef.current?.fit(undefined, 35);
  const handleZoomIn = () => cyRef.current?.zoom(cyRef.current.zoom() * 1.25);
  const handleZoomOut = () => cyRef.current?.zoom(cyRef.current.zoom() * 0.8);
  const handleLayoutCose = () => {
    cyRef.current?.layout({ name: 'cose', animate: true }).run();
  };
  const handleLayoutConcentric = () => {
    cyRef.current?.layout({ name: 'concentric', animate: true }).run();
  };

  return (
    <div className="graph-viewport" id="graph-viewport">
      <div className="graph-controls" id="graph-controls">
        <button
          className="btn btn-secondary"
          id="graph-fit-btn"
          onClick={handleFit}
          title="Fit Graph"
        >
          Fit
        </button>
        <button
          className="btn btn-secondary"
          id="graph-zoom-in-btn"
          onClick={handleZoomIn}
          title="Zoom In"
        >
          +
        </button>
        <button
          className="btn btn-secondary"
          id="graph-zoom-out-btn"
          onClick={handleZoomOut}
          title="Zoom Out"
        >
          -
        </button>
        <button
          className="btn btn-secondary"
          id="graph-cose-btn"
          onClick={handleLayoutCose}
          title="Force Layout"
        >
          Force
        </button>
        <button
          className="btn btn-secondary"
          id="graph-concentric-btn"
          onClick={handleLayoutConcentric}
          title="Concentric Layout"
        >
          Ring
        </button>
      </div>

      <div id="cy-graph-container" ref={containerRef} />

      {(!graphData || graphData.elements.nodes.length === 0) && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-muted)',
            pointerEvents: 'none',
            gap: '8px',
          }}
        >
          <span
            style={{
              fontSize: '15px',
              fontWeight: 600,
              color: 'var(--text-primary)',
            }}
          >
            No Clinical Graph Nodes
          </span>
          <span style={{ fontSize: '12px' }}>
            Select a patient or click a demo patient above to view graph.
          </span>
        </div>
      )}

      <div className="graph-legend" id="graph-legend">
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#2563eb' }} />
          <span>Patient</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#0d9488' }} />
          <span>Episode</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#059669' }} />
          <span>Fact</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#0284c7' }} />
          <span>Observation</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#ea580c' }} />
          <span>Handoff</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#e11d48' }} />
          <span>Flag</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#d97706' }} />
          <span>Clinic Rule</span>
        </div>
      </div>
    </div>
  );
};
