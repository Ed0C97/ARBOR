"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Search, ZoomIn, ZoomOut, Maximize2, Loader2 } from "lucide-react";
import dynamic from "next/dynamic";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
});

interface GraphNode {
  id: string;
  name: string;
  category: string;
  entity_type?: string;
  x?: number;
  y?: number;
}
interface GraphLink {
  source: string;
  target: string;
  type?: string;
}
interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

const CATEGORY_COLORS: Record<string, string> = {
  restaurant: "#3B82F6",
  bar: "#7C3AED",
  hotel: "#F59E0B",
  shop: "#10B981",
  cafe: "#F97316",
  gallery: "#4353FF",
  tailoring: "#64748B",
  accessories: "#6B7280",
  clothing: "#10B981",
  footwear: "#F59E0B",
  jewelry: "#EC4899",
  food_drink: "#7C3AED",
  default: "#9CA3AF",
};

function getCategoryColor(category: string): string {
  return CATEGORY_COLORS[category] ?? CATEGORY_COLORS.default;
}

function generateDemoData(): GraphData {
  const categories = Object.keys(CATEGORY_COLORS).filter(
    (k) => k !== "default"
  );
  const names = [
    "Ristorante Porfido", "Bar Centrale", "Hotel Palazzo", "Boutique Nera",
    "Cafe del Corso", "Galleria Arte", "Sartoria Toscana", "Accessori Lusso",
    "Emporio Stile", "Calzature Fiorentine", "Gioielleria Oro", "Trattoria Vino",
    "Lounge Noir", "Residenza Flora", "Concept Store Lux", "Bistrot Moderno",
    "Osteria del Porto", "Cocktail Lab",
  ];
  const nodes = names.map((name, i) => ({
    id: String(i),
    name,
    category: categories[i % categories.length],
  }));
  const links: GraphLink[] = [];
  for (let i = 0; i < nodes.length; i++) {
    const numEdges = 1 + Math.floor(Math.random() * 3);
    for (let j = 0; j < numEdges; j++) {
      const target = Math.floor(Math.random() * nodes.length);
      if (target !== i)
        links.push({
          source: nodes[i].id,
          target: nodes[target].id,
          type: "related",
        });
    }
  }
  return { nodes, links };
}

export default function MapPage() {
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [graphData, setGraphData] = useState<GraphData>({
    nodes: [],
    links: [],
  });
  const [loading, setLoading] = useState(true);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [query, setQuery] = useState("");

  useEffect(() => {
    async function loadGraph() {
      try {
        const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
        const res = await fetch(`${API}/api/v1/graph/full?limit=200`);
        if (!res.ok) throw new Error("API error");
        const data = await res.json();
        const nodes = (data.nodes ?? []).map((n: any) => ({
          id: n.id,
          name: n.name,
          category: n.category ?? "default",
          entity_type: n.entity_type,
        }));
        const rawEdges = data.links ?? data.edges ?? [];
        const links = rawEdges.map((e: any) => ({
          source: e.source,
          target: e.target,
          type: e.type ?? "related",
        }));
        setGraphData({ nodes, links });
      } catch {
        setGraphData(generateDemoData());
      } finally {
        setLoading(false);
      }
    }
    loadGraph();
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () =>
      setDimensions({ width: el.clientWidth, height: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (loading || !graphRef.current) return;
    setTimeout(() => {
      const fg = graphRef.current;
      if (!fg) return;
      fg.d3Force("charge")?.strength(-200);
      fg.d3Force("link")?.distance(80);
      fg.d3Force("center")?.strength(0.05);
    }, 100);
  }, [loading, graphData]);

  const matchingIds = useMemo(() => {
    if (!query.trim()) return null;
    const q = query.toLowerCase();
    const ids = new Set<string>();
    for (const node of graphData.nodes) {
      if (
        node.name.toLowerCase().includes(q) ||
        node.category.toLowerCase().includes(q)
      ) {
        ids.add(node.id);
      }
    }
    return ids;
  }, [query, graphData.nodes]);

  const hoveredConnections = useMemo(() => {
    if (!hoveredNode) return new Set<string>();
    const ids = new Set<string>();
    ids.add(hoveredNode.id);
    for (const link of graphData.links) {
      const srcId =
        typeof link.source === "object" ? (link.source as any).id : link.source;
      const tgtId =
        typeof link.target === "object" ? (link.target as any).id : link.target;
      if (srcId === hoveredNode.id) ids.add(tgtId);
      if (tgtId === hoveredNode.id) ids.add(srcId);
    }
    return ids;
  }, [hoveredNode, graphData.links]);

  const legendCategories = useMemo(() => {
    const cats = new Set(graphData.nodes.map((n) => n.category));
    return Object.entries(CATEGORY_COLORS)
      .filter(([k]) => k !== "default" && cats.has(k))
      .sort(([a], [b]) => a.localeCompare(b));
  }, [graphData.nodes]);

  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const id = node.id;
      const name = node.name ?? "";
      const category = node.category ?? "default";
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const isHovered = hoveredNode?.id === id;
      const isSearchMatch = matchingIds === null || matchingIds.has(id);
      const isDimmed = matchingIds !== null && !matchingIds.has(id);

      const radius = isHovered ? 7 : 5;
      const color = getCategoryColor(category);

      ctx.save();
      if (isDimmed) ctx.globalAlpha = 0.1;
      else if (isSearchMatch) ctx.globalAlpha = 1;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      if (isHovered) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
      }

      ctx.shadowBlur = 0;
      if (!isDimmed) {
        const fontSize = Math.max(10 / globalScale, 3);
        ctx.font = `${fontSize}px Inter, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = isHovered ? "#111827" : "rgba(107, 114, 128, 0.8)";
        ctx.fillText(name, x, y + radius + 3);
      }
      ctx.restore();
    },
    [hoveredNode, matchingIds]
  );

  const linkCanvasObject = useCallback(
    (link: any, ctx: CanvasRenderingContext2D) => {
      const src = link.source;
      const tgt = link.target;
      if (!src || !tgt) return;
      const srcX = typeof src === "object" ? src.x : 0;
      const srcY = typeof src === "object" ? src.y : 0;
      const tgtX = typeof tgt === "object" ? tgt.x : 0;
      const tgtY = typeof tgt === "object" ? tgt.y : 0;

      const srcId = typeof src === "object" ? src.id : src;
      const tgtId = typeof tgt === "object" ? tgt.id : tgt;
      const isHighlighted =
        hoveredConnections.has(srcId) && hoveredConnections.has(tgtId);

      ctx.beginPath();
      ctx.moveTo(srcX, srcY);
      ctx.lineTo(tgtX, tgtY);
      ctx.strokeStyle = isHighlighted
        ? "#6B7280"
        : "rgba(209, 213, 219, 0.3)";
      ctx.lineWidth = isHighlighted ? 1.5 : 0.5;
      ctx.stroke();
    },
    [hoveredConnections]
  );

  const handleNodeClick = useCallback((node: any) => {
    if (node?.id) window.location.href = `/entity/${node.id}`;
  }, []);

  const handleNodeHover = useCallback((node: any | null) => {
    setHoveredNode(node ?? null);
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (hoveredNode) setTooltipPos({ x: e.clientX, y: e.clientY });
    },
    [hoveredNode]
  );

  return (
    <div className="flex h-[calc(100vh-7rem)] flex-col gap-4">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">
          Knowledge Graph
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          Visualizing {graphData.nodes.length} entities and their relationships
        </p>
      </div>

      <div
        ref={containerRef}
        className="relative flex-1 overflow-hidden border border-gray-200 bg-white shadow-render"
        style={{ minHeight: 500 }}
        onMouseMove={handleMouseMove}
      >
        {loading ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="h-5 w-5 animate-spin text-gray-400" />
            <span className="ml-3 text-sm text-gray-500">Loading graph...</span>
          </div>
        ) : (
          <ForceGraph2D
            ref={graphRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={graphData}
            backgroundColor="#ffffff"
            d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
            nodeCanvasObject={nodeCanvasObject}
            linkCanvasObject={linkCanvasObject}
            onNodeClick={handleNodeClick}
            onNodeHover={handleNodeHover}
            enableNodeDrag={true}
          />
        )}

        {/* Search */}
        <div className="absolute right-4 top-4 z-10 flex items-center gap-2 border border-gray-200 bg-white p-2 shadow-render-md">
          <Search className="h-4 w-4 text-gray-400 ml-1" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter entities..."
            className="h-7 w-52 bg-transparent text-sm text-gray-900 focus:outline-none placeholder:text-gray-400"
          />
        </div>

        {/* Legend */}
        <div className="absolute left-4 top-4 z-10 border border-gray-200 bg-white p-4 shadow-render-md max-h-[80%] overflow-y-auto scrollbar-thin">
          <div className="mb-3 text-xs font-medium uppercase tracking-wider text-gray-400">
            Categories
          </div>
          <div className="space-y-2">
            {legendCategories.map(([cat, color]) => (
              <div key={cat} className="flex items-center gap-3">
                <div
                  className="h-3 w-3 shrink-0 rounded"
                  style={{ backgroundColor: color }}
                />
                <span className="text-xs text-gray-600 capitalize">
                  {cat.replace(/_/g, " ")}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-1">
          {[
            {
              icon: ZoomIn,
              fn: () => {
                const fg = graphRef.current;
                if (fg) fg.zoom(fg.zoom() * 1.4, 300);
              },
            },
            {
              icon: ZoomOut,
              fn: () => {
                const fg = graphRef.current;
                if (fg) fg.zoom(fg.zoom() / 1.4, 300);
              },
            },
            {
              icon: Maximize2,
              fn: () => {
                const fg = graphRef.current;
                if (fg) fg.zoomToFit(400, 40);
              },
            },
          ].map((btn, i) => (
            <button
              key={i}
              onClick={btn.fn}
              className="flex h-8 w-8 items-center justify-center border border-gray-200 bg-white text-gray-500 hover:bg-gray-50 hover:text-gray-700 shadow-sm transition-colors"
            >
              <btn.icon className="h-4 w-4" />
            </button>
          ))}
        </div>

        {/* Tooltip */}
        {hoveredNode && (
          <div
            className="absolute z-50 border border-gray-200 bg-white px-3 py-2 text-xs shadow-render-md"
            style={{
              left:
                tooltipPos.x -
                containerRef.current!.getBoundingClientRect().left +
                10,
              top:
                tooltipPos.y -
                containerRef.current!.getBoundingClientRect().top -
                10,
            }}
          >
            <div className="font-medium text-gray-900">{hoveredNode.name}</div>
            <div className="text-gray-500 capitalize">
              {hoveredNode.category}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
