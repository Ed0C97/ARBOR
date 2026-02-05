"use client";

import Link from "next/link";
import {
  Search,
  Network,
  Sparkles,
  Shield,
  Zap,
  ArrowRight,
  ChevronRight,
  Bot,
  CheckCircle,
} from "lucide-react";

const NAV_LINKS = [
  { label: "Product", href: "#features" },
  { label: "Docs", href: "#how-it-works" },
  { label: "Pricing", href: "#pricing" },
  { label: "Blog", href: "#" },
];

const FEATURES = [
  {
    icon: Search,
    title: "Semantic Discovery",
    description:
      "Query your knowledge base with natural language. ARBOR understands intent, context, and nuance to surface the most relevant entities.",
  },
  {
    icon: Network,
    title: "Knowledge Graph",
    description:
      "Entities connected through rich relationships. Explore brands, venues, and styles through an intelligent, traversable graph.",
  },
  {
    icon: Bot,
    title: "Agentic AI Pipeline",
    description:
      "Multi-agent orchestration with LangGraph. Vector, metadata, and graph agents work in parallel to deliver comprehensive results.",
  },
  {
    icon: Sparkles,
    title: "Vibe DNA Enrichment",
    description:
      "Automated multi-dimensional scoring across formality, craftsmanship, atmosphere, and more. Every entity gets a unique vibe fingerprint.",
  },
  {
    icon: Shield,
    title: "Guardrails & Safety",
    description:
      "NeMo-powered input and output validation. Professional, factual responses with no fabricated entities or unsupported claims.",
  },
  {
    icon: Zap,
    title: "Semantic Caching",
    description:
      "Two-tier caching with Redis and Qdrant. Similar queries resolve instantly through embedding-based semantic matching.",
  },
];

const STATS = [
  { value: "50ms", label: "Avg. cache hit latency" },
  { value: "99.9%", label: "API uptime SLA" },
  { value: "6", label: "Vibe dimensions scored" },
  { value: "3", label: "Parallel AI agents" },
];

const STEPS = [
  {
    step: "01",
    title: "Ingest your data",
    description:
      "Connect your entity sources. ARBOR scrapes, enriches, and indexes brands, venues, and products into a unified knowledge layer.",
  },
  {
    step: "02",
    title: "Enrich with AI",
    description:
      "Our pipeline extracts vibe DNA, generates embeddings, builds graph relationships, and calibrates scores against gold standards.",
  },
  {
    step: "03",
    title: "Discover & explore",
    description:
      "Query with natural language. Get curated, context-aware recommendations powered by parallel agent execution and semantic search.",
  },
];

const LOGOS = [
  "PostgreSQL",
  "Neo4j",
  "Qdrant",
  "Redis",
  "LangGraph",
  "FastAPI",
  "Next.js",
  "Temporal",
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white">
      {/* ─── Navbar ─── */}
      <nav className="sticky top-0 z-50 border-b border-gray-200/80 bg-white/80 backdrop-blur-lg">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-8">
            <Link href="/" className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-[#4353FF] to-[#7C3AED] text-white text-sm font-bold">
                A
              </div>
              <span className="text-lg font-semibold text-gray-900 tracking-tight">
                ARBOR
              </span>
            </Link>
            <div className="hidden md:flex items-center gap-6">
              {NAV_LINKS.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/login"
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors px-3 py-2"
            >
              Sign In
            </Link>
            <Link
              href="/login"
              className="inline-flex items-center gap-2 rounded-lg bg-[#4353FF] px-4 py-2 text-sm font-medium text-white hover:bg-[#3643E0] transition-colors"
            >
              Get Started
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </nav>

      {/* ─── Hero ─── */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-50/50 via-purple-50/30 to-white" />
        <div className="relative mx-auto max-w-7xl px-6 pb-24 pt-20 md:pt-32 md:pb-32">
          <div className="mx-auto max-w-3xl text-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-4 py-1.5 text-sm text-blue-700">
              <Sparkles className="h-3.5 w-3.5" />
              Powered by agentic AI and knowledge graphs
            </div>
            <h1 className="text-5xl font-bold tracking-tight text-gray-900 md:text-6xl lg:text-7xl">
              Your fastest path to{" "}
              <span className="gradient-text">discovery</span>
            </h1>
            <p className="mt-6 text-lg text-gray-600 leading-relaxed md:text-xl">
              ARBOR is the contextual discovery engine that connects semantic
              search, knowledge graphs, and AI agents to deliver curated,
              context-aware recommendations at scale.
            </p>
            <div className="mt-10 flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
              <Link
                href="/login"
                className="inline-flex items-center gap-2 rounded-lg bg-[#4353FF] px-6 py-3 text-base font-medium text-white hover:bg-[#3643E0] transition-colors shadow-lg shadow-blue-500/20"
              >
                Start discovering
                <ArrowRight className="h-4 w-4" />
              </Link>
              <a
                href="#features"
                className="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-6 py-3 text-base font-medium text-gray-700 hover:bg-gray-50 transition-colors"
              >
                See how it works
              </a>
            </div>
          </div>

          {/* Hero visual */}
          <div className="mx-auto mt-16 max-w-5xl">
            <div className="rounded-xl border border-gray-200 bg-white p-1 shadow-render-lg">
              <div className="rounded-lg bg-gradient-to-br from-gray-50 to-gray-100 p-8 md:p-12">
                <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                  {[
                    {
                      label: "Vector Search",
                      desc: "Semantic similarity",
                      icon: Search,
                      color: "text-blue-600 bg-blue-50",
                    },
                    {
                      label: "Graph Traversal",
                      desc: "Relationship reasoning",
                      icon: Network,
                      color: "text-purple-600 bg-purple-50",
                    },
                    {
                      label: "AI Synthesis",
                      desc: "Natural language output",
                      icon: Bot,
                      color: "text-pink-600 bg-pink-50",
                    },
                  ].map((item) => (
                    <div
                      key={item.label}
                      className="rounded-lg border border-gray-200 bg-white p-5 shadow-render"
                    >
                      <div
                        className={`mb-3 inline-flex h-10 w-10 items-center justify-center rounded-lg ${item.color}`}
                      >
                        <item.icon className="h-5 w-5" />
                      </div>
                      <h3 className="font-semibold text-gray-900">
                        {item.label}
                      </h3>
                      <p className="mt-1 text-sm text-gray-500">{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ─── Tech Stack ─── */}
      <section className="border-y border-gray-200 bg-gray-50 py-12">
        <div className="mx-auto max-w-7xl px-6">
          <p className="mb-8 text-center text-sm font-medium uppercase tracking-wider text-gray-500">
            Built on industry-leading technologies
          </p>
          <div className="flex flex-wrap items-center justify-center gap-x-12 gap-y-4">
            {LOGOS.map((logo) => (
              <span
                key={logo}
                className="text-sm font-medium text-gray-400 hover:text-gray-600 transition-colors"
              >
                {logo}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Features Grid ─── */}
      <section id="features" className="py-24">
        <div className="mx-auto max-w-7xl px-6">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 md:text-4xl">
              Everything you need for intelligent discovery
            </h2>
            <p className="mt-4 text-lg text-gray-600">
              A complete platform for ingesting, enriching, and querying
              curated entities with AI-powered precision.
            </p>
          </div>
          <div className="mt-16 grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((feature) => (
              <div
                key={feature.title}
                className="group rounded-xl border border-gray-200 bg-white p-6 transition-all hover:border-gray-300 hover:shadow-render-md"
              >
                <div className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-lg bg-blue-50 text-blue-600 group-hover:bg-blue-100 transition-colors">
                  <feature.icon className="h-5 w-5" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">
                  {feature.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-gray-600">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Stats ─── */}
      <section className="border-y border-gray-200 bg-gray-50 py-16">
        <div className="mx-auto max-w-7xl px-6">
          <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
            {STATS.map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-3xl font-bold text-gray-900 md:text-4xl">
                  {stat.value}
                </div>
                <div className="mt-2 text-sm text-gray-500">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── How It Works ─── */}
      <section id="how-it-works" className="py-24">
        <div className="mx-auto max-w-7xl px-6">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 md:text-4xl">
              From raw data to curated discovery
            </h2>
            <p className="mt-4 text-lg text-gray-600">
              Three steps to transform your entity data into an intelligent,
              queryable knowledge base.
            </p>
          </div>
          <div className="mt-16 grid grid-cols-1 gap-8 md:grid-cols-3">
            {STEPS.map((step) => (
              <div key={step.step} className="relative">
                <div className="mb-4 text-5xl font-bold text-gray-200">
                  {step.step}
                </div>
                <h3 className="text-xl font-semibold text-gray-900">
                  {step.title}
                </h3>
                <p className="mt-3 text-sm leading-relaxed text-gray-600">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Pricing ─── */}
      <section id="pricing" className="bg-gray-50 py-24">
        <div className="mx-auto max-w-7xl px-6">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 md:text-4xl">
              Simple, transparent pricing
            </h2>
            <p className="mt-4 text-lg text-gray-600">
              Start free, scale as you grow. No surprises.
            </p>
          </div>
          <div className="mt-16 grid grid-cols-1 gap-8 md:grid-cols-3">
            {[
              {
                name: "Free",
                price: "$0",
                description: "For exploration and prototyping",
                features: [
                  "10 queries/minute",
                  "100 queries/day",
                  "Basic semantic search",
                  "Community support",
                ],
                cta: "Get started",
                highlighted: false,
              },
              {
                name: "Pro",
                price: "$29",
                description: "For teams and power users",
                features: [
                  "60 queries/minute",
                  "1,000 queries/day",
                  "Full knowledge graph",
                  "Vibe DNA enrichment",
                  "Priority support",
                ],
                cta: "Start free trial",
                highlighted: true,
              },
              {
                name: "Enterprise",
                price: "Custom",
                description: "For large-scale deployments",
                features: [
                  "Unlimited queries",
                  "Custom domain configs",
                  "Dedicated infrastructure",
                  "SLA guarantees",
                  "24/7 support",
                ],
                cta: "Contact sales",
                highlighted: false,
              },
            ].map((plan) => (
              <div
                key={plan.name}
                className={`rounded-xl border p-8 ${
                  plan.highlighted
                    ? "border-[#4353FF] bg-white shadow-render-lg ring-1 ring-[#4353FF]/10"
                    : "border-gray-200 bg-white"
                }`}
              >
                <h3 className="text-lg font-semibold text-gray-900">
                  {plan.name}
                </h3>
                <div className="mt-2">
                  <span className="text-4xl font-bold text-gray-900">
                    {plan.price}
                  </span>
                  {plan.price !== "Custom" && (
                    <span className="text-gray-500">/month</span>
                  )}
                </div>
                <p className="mt-2 text-sm text-gray-500">
                  {plan.description}
                </p>
                <ul className="mt-6 space-y-3">
                  {plan.features.map((feature) => (
                    <li
                      key={feature}
                      className="flex items-center gap-2 text-sm text-gray-600"
                    >
                      <CheckCircle className="h-4 w-4 shrink-0 text-green-500" />
                      {feature}
                    </li>
                  ))}
                </ul>
                <Link
                  href="/login"
                  className={`mt-8 flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors ${
                    plan.highlighted
                      ? "bg-[#4353FF] text-white hover:bg-[#3643E0]"
                      : "border border-gray-300 text-gray-700 hover:bg-gray-50"
                  }`}
                >
                  {plan.cta}
                  <ChevronRight className="h-4 w-4" />
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── CTA ─── */}
      <section className="py-24">
        <div className="mx-auto max-w-7xl px-6">
          <div className="rounded-2xl bg-gradient-to-r from-[#4353FF] to-[#7C3AED] px-8 py-16 text-center md:px-16">
            <h2 className="text-3xl font-bold text-white md:text-4xl">
              Ready to transform your discovery experience?
            </h2>
            <p className="mx-auto mt-4 max-w-xl text-lg text-blue-100">
              Join teams using ARBOR to build intelligent, context-aware
              discovery systems.
            </p>
            <div className="mt-8 flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
              <Link
                href="/login"
                className="inline-flex items-center gap-2 rounded-lg bg-white px-6 py-3 text-base font-medium text-[#4353FF] hover:bg-blue-50 transition-colors"
              >
                Get started for free
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ─── Footer ─── */}
      <footer className="border-t border-gray-200 bg-white py-12">
        <div className="mx-auto max-w-7xl px-6">
          <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
            <div>
              <div className="flex items-center gap-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-[#4353FF] to-[#7C3AED] text-white text-xs font-bold">
                  A
                </div>
                <span className="font-semibold text-gray-900">ARBOR</span>
              </div>
              <p className="mt-3 text-sm text-gray-500">
                Advanced Reasoning By Ontological Rules
              </p>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-900">Product</h4>
              <ul className="mt-3 space-y-2">
                {["Features", "Pricing", "Changelog", "Status"].map((item) => (
                  <li key={item}>
                    <a href="#" className="text-sm text-gray-500 hover:text-gray-700 transition-colors">
                      {item}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-900">Resources</h4>
              <ul className="mt-3 space-y-2">
                {["Documentation", "API Reference", "Guides", "Blog"].map((item) => (
                  <li key={item}>
                    <a href="#" className="text-sm text-gray-500 hover:text-gray-700 transition-colors">
                      {item}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-900">Company</h4>
              <ul className="mt-3 space-y-2">
                {["About", "Careers", "Contact", "Legal"].map((item) => (
                  <li key={item}>
                    <a href="#" className="text-sm text-gray-500 hover:text-gray-700 transition-colors">
                      {item}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <div className="mt-12 border-t border-gray-200 pt-8 text-center text-sm text-gray-400">
            &copy; {new Date().getFullYear()} ARBOR. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
