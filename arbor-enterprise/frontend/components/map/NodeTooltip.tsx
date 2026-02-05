interface NodeTooltipProps {
  name: string;
  category: string;
  extra?: Record<string, string | number>;
  className?: string;
}

export default function NodeTooltip({
  name,
  category,
  extra,
  className = "",
}: NodeTooltipProps) {
  return (
    <div
      className={`glass-panel px-4 py-2.5 shadow-xl ${className}`}
    >
      <div className="font-serif text-sm text-foreground">{name}</div>
      <div className="font-mono text-[11px] uppercase text-muted-foreground">
        {category}
      </div>
      {extra &&
        Object.entries(extra).map(([key, val]) => (
          <div key={key} className="mt-1 font-mono text-[11px] text-muted-foreground">
            <span className="uppercase text-foreground/60">{key}:</span> {val}
          </div>
        ))}
    </div>
  );
}
