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
            label: 'data(label)',
            'color': '#f8fafc',
            'font-size': '11px',
            'text-valign': 'center',
            'text-halign': 'center',
            'text-outline-color': '#0f172a',
            'text-outline-width': '2px',
            'background-color': '#475569',
            'border-width': '1.5px',
            'border-color': '#94a3b8',
            width: '40px',
            height: '40px',
            'transition-property': 'background-color, border-color, opacity, width, height',
            'transition-duration': 0.2,
          },
        },
        {
          selector: 'node[type = "Patient"]',
          style: {
            'background-color': '#2563eb',
            shape: 'ellipse',
            width: '54px',
            height: '54px',
            'font-weight': 'bold',
          },
        },
        {
          selector: 'node[type = "Session"]',
          style: {
            'background-color': '#7c3aed',
            shape: 'hexagon',
            width: '46px',
            height: '46px',
          },
        },
        {
          selector: 'node[type = "Episode"]',
          style: {
            'background-color': '#0d9488',
            shape: 'round-rectangle',
            width: '44px',
            height: '44px',
          },
        },
        {
          selector: 'node[type = "Fact"]',
          style: {
            'background-color': '#059669',
            shape: 'diamond',
            width: '42px',
            height: '42px',
          },
        },
        {
          selector: 'node[type = "Observation"]',
          style: {
            'background-color': '#0284c7',
            shape: 'round-rectangle',
            width: '38px',
            height: '38px',
          },
        },
        {
          selector: 'node[type = "Answer"]',
          style: {
            'background-color': '#9333ea',
            shape: 'star',
            width: '50px',
            height: '50px',
            'border-color': '#c084fc',
          },
        },
        {
          selector: 'node[type = "Handoff"]',
          style: {
            'background-color': '#ea580c',
            shape: 'vee',
            width: '44px',
            height: '44px',
          },
        },
        {
          selector: 'node[type = "Flag"]',
          style: {
            'background-color': '#e11d48',
            shape: 'triangle',
            width: '42px',
            height: '42px',
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
            'text-background-opacity': 0.8,
            'text-background-color': '#0f172a',
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

    cy.on('tap', 'node', (evt: EventObject) => {
      const data = evt.target.data() as CytoscapeNodeData;
      onNodeSelect(data);
    });

    cy.on('tap', (evt: EventObject) => {
      if (evt.target === cy) {
        onNodeSelect(null);
      }
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [onNodeSelect]);

  // Update elements when graphData changes
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !graphData) return;

    cy.elements().remove();
    const nodes = graphData.elements.nodes || [];
    const edges = graphData.elements.edges || [];

    cy.add([...nodes, ...edges]);

    const layout = cy.layout({
      name: 'cose',
      animate: false,
      idealEdgeLength: () => 90,
      nodeOverlap: 20,
    });
    layout.run();
    cy.fit(undefined, 35);
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

      <div className="graph-legend" id="graph-legend">
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#2563eb' }} />
          <span>Patient</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ backgroundColor: '#7c3aed' }} />
          <span>Session</span>
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
          <span>Clinic Guideline</span>
        </div>
      </div>
    </div>
  );
};
