"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { ArrowLeft, Loader2 } from "lucide-react";
import Link from "next/link";
import { getEntity } from "@/lib/api";
import type { Entity } from "@/lib/types";
import { EntityDetail } from "@/components/entity/EntityDetail";

export default function EntityPage() {
  const params = useParams<{ id: string }>();
  const [entity, setEntity] = useState<Entity | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!params.id) return;
    setLoading(true);
    setError(null);
    getEntity(params.id)
      .then((data) => setEntity(data))
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load entity")
      )
      .finally(() => setLoading(false));
  }, [params.id]);

  if (loading) {
    return (
      <div className="flex min-h-[400px] items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-gray-400" />
      </div>
    );
  }

  if (error || !entity) {
    return (
      <div className="flex min-h-[400px] flex-col items-center justify-center gap-4 text-center">
        <p className="text-sm font-medium text-red-600">
          {error ?? "Entity not found"}
        </p>
        <Link
          href="/discover"
          className="inline-flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700 transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Discover
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Link
        href="/discover"
        className="inline-flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700 transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Discover
      </Link>
      <EntityDetail entity={entity} />
    </div>
  );
}
