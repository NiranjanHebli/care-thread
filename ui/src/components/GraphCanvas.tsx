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
            width: '44px',
            height: '44px',
            'text-max-width': '95px',
            'text-wrap': 'ellipsis',
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
            width: '48px',
            height: '48px',
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
            color: '#94a3b8',
            'text-background-opacity': 0.85,
            'text-background-color': '#0a0e17',
            'text-background-padding': '2px',
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
    const rawNodes = graphData.elements.nodes || [];
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
          display: d.display || d.name || d.id,
        },
      };
    });

    cy.add([...nodes, ...rawEdges]);

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
