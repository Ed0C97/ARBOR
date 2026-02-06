import { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import * as d3 from 'd3';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Loader2,
  Network,
  Info,
} from 'lucide-react';

import { apiGet } from '@/lib/api';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

const NODE_COLORS = {
  brand: 'hsl(172, 66%, 50%)',
  venue: 'hsl(262, 83%, 58%)',
  category: 'hsl(38, 92%, 50%)',
  city: 'hsl(199, 89%, 48%)',
  default: 'hsl(215, 20%, 65%)',
};

const NODE_RADIUS = {
  brand: 8,
  venue: 8,
  category: 6,
  city: 6,
  default: 5,
};

export default function GraphPage() {
  const [searchParams] = useSearchParams();
  const entityId = searchParams.get('entity');
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [graphType, setGraphType] = useState('full');

  const fetchGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let data;
      if (entityId) {
        data = await apiGet(`/api/v1/graph/related`, { entity_id: entityId });
      } else if (graphType === 'brand-retailers') {
        data = await apiGet('/api/v1/graph/brand-retailers');
      } else {
        data = await apiGet('/api/v1/graph/full');
      }
      setGraphData(data);
    } catch (err) {
      setError(err.message || 'Failed to load graph');
    } finally {
      setLoading(false);
    }
  }, [entityId, graphType]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

  useEffect(() => {
    if (!graphData || !svgRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g');

    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    const nodes = (graphData.nodes || []).map((d) => ({ ...d }));
    const links = (graphData.edges || graphData.links || []).map((d) => ({
      ...d,
      source: d.source_id || d.source,
      target: d.target_id || d.target,
    }));

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d) => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(15));

    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', 'hsl(var(--border))')
      .attr('stroke-opacity', 0.5)
      .attr('stroke-width', 1);

    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', (d) => NODE_RADIUS[d.type || d.entity_type] || NODE_RADIUS.default)
      .attr('fill', (d) => NODE_COLORS[d.type || d.entity_type] || NODE_COLORS.default)
      .attr('stroke', 'hsl(var(--background))')
      .attr('stroke-width', 1.5)
      .attr('cursor', 'pointer')
      .on('click', (event, d) => {
        setSelectedNode(d);
      })
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }),
      );

    const labels = g.append('g')
      .selectAll('text')
      .data(nodes)
      .join('text')
      .text((d) => d.name || d.label || d.id)
      .attr('font-size', 10)
      .attr('dx', 12)
      .attr('dy', 4)
      .attr('fill', 'hsl(var(--muted-foreground))')
      .attr('pointer-events', 'none')
      .style('font-family', 'var(--font-sans)');

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);

      node
        .attr('cx', (d) => d.x)
        .attr('cy', (d) => d.y);

      labels
        .attr('x', (d) => d.x)
        .attr('y', (d) => d.y);
    });

    svg.zoomHandler = zoom;

    return () => simulation.stop();
  }, [graphData]);

  function handleZoom(direction) {
    const svg = d3.select(svgRef.current);
    const zoom = svg.zoomHandler;
    if (!zoom) return;

    if (direction === 'in') {
      svg.transition().duration(300).call(zoom.scaleBy, 1.3);
    } else if (direction === 'out') {
      svg.transition().duration(300).call(zoom.scaleBy, 0.7);
    } else {
      svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
    }
  }

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b px-4 py-2">
        <div className="flex items-center gap-3">
          <Select value={graphType} onValueChange={setGraphType}>
            <SelectTrigger className="w-[180px] h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="full">Full Graph</SelectItem>
              <SelectItem value="brand-retailers">Brand-Retailers</SelectItem>
            </SelectContent>
          </Select>

          {graphData && (
            <span className="text-xs text-muted-foreground">
              {graphData.nodes?.length || 0} nodes &middot; {(graphData.edges || graphData.links)?.length || 0} edges
            </span>
          )}
        </div>

        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" className="size-8" onClick={() => handleZoom('in')}>
            <ZoomIn className="size-4" />
          </Button>
          <Button variant="ghost" size="icon" className="size-8" onClick={() => handleZoom('out')}>
            <ZoomOut className="size-4" />
          </Button>
          <Button variant="ghost" size="icon" className="size-8" onClick={() => handleZoom('reset')}>
            <Maximize2 className="size-4" />
          </Button>
        </div>
      </div>

      {/* Graph Canvas */}
      <div ref={containerRef} className="relative flex-1">
        {loading ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="size-6 animate-spin text-muted-foreground" />
          </div>
        ) : error ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <Network className="mb-4 size-10 text-muted-foreground/50" />
            <p className="text-sm text-muted-foreground">{error}</p>
            <Button variant="outline" size="sm" className="mt-4" onClick={fetchGraph}>
              Retry
            </Button>
          </div>
        ) : (
          <svg
            ref={svgRef}
            className="h-full w-full"
            style={{ background: 'hsl(var(--background))' }}
          />
        )}

        {/* Legend */}
        <div className="absolute bottom-4 left-4 flex flex-wrap gap-3 rounded-lg border bg-background/80 px-3 py-2 text-xs backdrop-blur-sm">
          {Object.entries(NODE_COLORS).filter(([k]) => k !== 'default').map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <div
                className="size-2.5 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="capitalize text-muted-foreground">{type}</span>
            </div>
          ))}
        </div>

        {/* Selected node panel */}
        {selectedNode && (
          <Card className="absolute right-4 top-4 w-64">
            <CardHeader className="p-3">
              <CardTitle className="flex items-center justify-between text-sm">
                {selectedNode.name || selectedNode.id}
                <button
                  onClick={() => setSelectedNode(null)}
                  className="text-muted-foreground hover:text-foreground"
                >
                  &times;
                </button>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-3 pt-0 text-xs text-muted-foreground">
              {(selectedNode.type || selectedNode.entity_type) && (
                <Badge variant="secondary" className="mb-2 text-[10px] capitalize">
                  {selectedNode.type || selectedNode.entity_type}
                </Badge>
              )}
              {selectedNode.category && <p>Category: {selectedNode.category}</p>}
              {selectedNode.city && <p>City: {selectedNode.city}</p>}
              {selectedNode.id && (
                <Button asChild variant="link" size="sm" className="mt-2 h-auto p-0 text-xs">
                  <a href={`/entity/${selectedNode.id}`}>View details &rarr;</a>
                </Button>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
