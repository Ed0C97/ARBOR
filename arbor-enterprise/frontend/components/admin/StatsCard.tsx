import { type LucideIcon } from "lucide-react";

interface StatsCardProps {
  title: string;
  value: string | number;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: LucideIcon;
  borderColor?: string;
}

export default function StatsCard({
  title,
  value,
  change,
  changeType = "neutral",
  icon: Icon,
  borderColor = "border-l-primary",
}: StatsCardProps) {
  const changeColor = {
    positive: "text-primary",
    negative: "text-destructive",
    neutral: "text-muted-foreground",
  };

  return (
    <div
      className={`border border-border/50 bg-card p-5 border-l-4 ${borderColor}`}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-muted-foreground">{title}</p>
          <p className="mt-1 text-3xl font-mono font-bold text-foreground">
            {value}
          </p>
          {change && (
            <p className={`mt-1 text-xs ${changeColor[changeType]}`}>
              {change}
            </p>
          )}
        </div>
        <div className="bg-accent p-2">
          <Icon className="h-5 w-5 text-muted-foreground" />
        </div>
      </div>
    </div>
  );
}
