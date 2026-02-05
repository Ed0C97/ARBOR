interface LegendItem {
  label: string;
  color: string;
}

interface GraphLegendProps {
  items: LegendItem[];
  className?: string;
}

export default function GraphLegend({ items, className = "" }: GraphLegendProps) {
  return (
    <div
      className={`glass-panel p-3 text-xs ${className}`}
    >
      <div className="mb-2 font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
        Categories
      </div>
      <div className="space-y-1.5">
        {items.map((item) => (
          <div key={item.label} className="flex items-center gap-2">
            <div
              className="h-3 w-3 shrink-0"
              style={{ backgroundColor: item.color }}
            />
            <span className="font-mono text-[11px] uppercase text-muted-foreground">
              {item.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
