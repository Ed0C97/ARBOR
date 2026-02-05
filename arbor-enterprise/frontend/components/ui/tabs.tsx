"use client";

import { createContext, useContext, useState } from "react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Tabs component (lightweight, no Radix dependency)
// ---------------------------------------------------------------------------

const TabsContext = createContext<{
  value: string;
  onValueChange: (v: string) => void;
}>({ value: "", onValueChange: () => {} });

interface TabsProps {
  defaultValue: string;
  children: React.ReactNode;
  className?: string;
}

export function Tabs({ defaultValue, children, className }: TabsProps) {
  const [value, setValue] = useState(defaultValue);

  return (
    <TabsContext.Provider value={{ value, onValueChange: setValue }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  );
}

export function TabsList({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-0 bg-transparent border-b border-border",
        className,
      )}
    >
      {children}
    </div>
  );
}

interface TabsTriggerProps {
  value: string;
  children: React.ReactNode;
  className?: string;
}

export function TabsTrigger({ value, children, className }: TabsTriggerProps) {
  const { value: active, onValueChange } = useContext(TabsContext);
  const isActive = active === value;

  return (
    <button
      onClick={() => onValueChange(value)}
      className={cn(
        "inline-flex items-center justify-center px-4 py-2 text-sm font-medium font-mono transition-colors border-b-2 border-transparent",
        isActive
          ? "text-primary border-primary"
          : "text-muted-foreground hover:text-foreground hover:border-border",
        className,
      )}
    >
      {children}
    </button>
  );
}

interface TabsContentProps {
  value: string;
  children: React.ReactNode;
  className?: string;
}

export function TabsContent({ value, children, className }: TabsContentProps) {
  const { value: active } = useContext(TabsContext);
  if (active !== value) return null;

  return <div className={cn("mt-4", className)}>{children}</div>;
}
