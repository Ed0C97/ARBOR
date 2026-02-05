"use client";

import { useEffect, useRef } from "react";
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";
import { Radar } from "react-chartjs-2";
import type { VibeDimensions } from "@/lib/types";

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
);

/** Human-readable labels for each Vibe DNA dimension. */
const DIMENSION_LABELS: Record<string, string> = {
  formality: "Formality",
  craftsmanship: "Craftsmanship",
  price_value: "Price / Value",
  atmosphere: "Atmosphere",
  service_quality: "Service",
  exclusivity: "Exclusivity",
};

/** Canonical dimension order for consistent chart rendering. */
const DIMENSION_ORDER = [
  "formality",
  "craftsmanship",
  "price_value",
  "atmosphere",
  "service_quality",
  "exclusivity",
];

interface VibeRadarProps {
  dimensions: VibeDimensions;
  className?: string;
}

export function VibeRadar({ dimensions, className }: VibeRadarProps) {
  const chartRef = useRef<ChartJS<"radar"> | null>(null);

  const labels = DIMENSION_ORDER.map(
    (key) => DIMENSION_LABELS[key] ?? key,
  );

  const values = DIMENSION_ORDER.map((key) => dimensions[key] ?? 0);

  const data = {
    labels,
    datasets: [
      {
        label: "Vibe DNA",
        data: values,
        backgroundColor: "rgba(17, 212, 160, 0.15)",
        borderColor: "rgba(17, 212, 160, 0.8)",
        borderWidth: 2,
        pointBackgroundColor: "rgba(17, 212, 160, 1)",
        pointBorderColor: "rgba(17, 212, 160, 1)",
        pointRadius: 4,
        pointHoverRadius: 6,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: "hsl(222, 40%, 4%)",
        titleColor: "hsl(0, 0%, 98%)",
        bodyColor: "hsl(0, 0%, 98%)",
        borderColor: "hsl(213, 18%, 20%)",
        borderWidth: 1,
        padding: 10,
        cornerRadius: 0,
        callbacks: {
          label: (context: { parsed: { r: number } }) =>
            `${context.parsed.r}/100`,
        },
      },
    },
    scales: {
      r: {
        beginAtZero: true,
        max: 100,
        ticks: {
          stepSize: 25,
          display: false,
        },
        grid: {
          color: "rgba(255, 255, 255, 0.08)",
        },
        angleLines: {
          color: "rgba(255, 255, 255, 0.08)",
        },
        pointLabels: {
          font: {
            size: 11,
            family: '"JetBrains Mono", monospace',
          },
          color: "rgba(255, 255, 255, 0.5)",
        },
      },
    },
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      chartRef.current?.destroy();
    };
  }, []);

  return (
    <div className={className}>
      <Radar ref={chartRef} data={data} options={options} />
    </div>
  );
}
