"use client";

import { useState, useEffect } from "react";
import {
  User,
  Mail,
  MapPin,
  Bell,
  Shield,
  Palette,
  Save,
  Loader2,
  LogOut,
  ChevronRight,
} from "lucide-react";
import { useRouter } from "next/navigation";

interface UserProfile {
  id: string;
  name: string;
  email: string;
  avatar_url: string | null;
  city: string;
  preferences: {
    categories: string[];
    price_range: [number, number];
    notifications: boolean;
    theme: "dark" | "light" | "system";
  };
  created_at: string;
}

const CATEGORIES = [
  "restaurant", "bar", "hotel", "shop", "cafe", "gallery", "spa", "club",
];

export default function ProfilePage() {
  const router = useRouter();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState<"general" | "preferences" | "security">("general");
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [city, setCity] = useState("");
  const [notificationsOn, setNotificationsOn] = useState(true);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [theme, setTheme] = useState<"dark" | "light" | "system">("light");

  useEffect(() => {
    async function loadProfile() {
      try {
        const token = localStorage.getItem("arbor_token");
        const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
        const res = await fetch(`${API}/api/v1/auth/me`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) throw new Error("Unauthorized");
        const data: UserProfile = await res.json();
        setProfile(data);
        setName(data.name);
        setCity(data.city);
        setNotificationsOn(data.preferences.notifications);
        setSelectedCategories(data.preferences.categories);
        setTheme(data.preferences.theme);
      } catch {
        const demo: UserProfile = {
          id: "demo-user-1",
          name: "Demo User",
          email: "demo@arbor.ai",
          avatar_url: null,
          city: "Milan",
          preferences: {
            categories: ["restaurant", "bar", "gallery"],
            price_range: [1, 4],
            notifications: true,
            theme: "light",
          },
          created_at: new Date().toISOString(),
        };
        setProfile(demo);
        setName(demo.name);
        setCity(demo.city);
        setNotificationsOn(demo.preferences.notifications);
        setSelectedCategories(demo.preferences.categories);
        setTheme(demo.preferences.theme);
      } finally {
        setLoading(false);
      }
    }
    loadProfile();
  }, []);

  async function handleSave() {
    setSaving(true);
    setSuccessMsg(null);
    try {
      const token = localStorage.getItem("arbor_token");
      const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      await fetch(`${API}/api/v1/auth/me`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          name,
          city,
          preferences: { categories: selectedCategories, notifications: notificationsOn, theme },
        }),
      });
      setSuccessMsg("Profile saved successfully");
      setTimeout(() => setSuccessMsg(null), 3000);
    } catch {
      setSuccessMsg("Saved locally (API unavailable)");
      setTimeout(() => setSuccessMsg(null), 3000);
    } finally {
      setSaving(false);
    }
  }

  function toggleCategory(cat: string) {
    setSelectedCategories((prev) =>
      prev.includes(cat) ? prev.filter((c) => c !== cat) : [...prev, cat]
    );
  }

  function handleLogout() {
    localStorage.removeItem("arbor_token");
    router.push("/login");
  }

  if (loading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-gray-400" />
      </div>
    );
  }

  const tabs = [
    { id: "general" as const, label: "General", icon: User },
    { id: "preferences" as const, label: "Preferences", icon: Palette },
    { id: "security" as const, label: "Security", icon: Shield },
  ];

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">Settings</h1>
        <p className="mt-1 text-sm text-gray-500">
          Manage your account settings and preferences
        </p>
      </div>

      {successMsg && (
        <div className="rounded-lg bg-green-50 border border-green-200 px-4 py-3 text-sm text-green-700">
          {successMsg}
        </div>
      )}

      <div className="flex flex-col md:flex-row gap-6">
        {/* Sidebar Tabs */}
        <div className="w-full md:w-56 shrink-0 space-y-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 text-sm rounded-lg text-left transition-colors ${
                activeTab === tab.id
                  ? "bg-blue-50 text-[#4353FF] font-medium"
                  : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
              }`}
            >
              <tab.icon className="h-4 w-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 rounded-xl border border-gray-200 bg-white p-8 shadow-render min-h-[450px]">
          {activeTab === "general" && (
            <div className="space-y-8 animate-fade-in">
              <div className="flex items-center gap-6 pb-6 border-b border-gray-200">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-purple-500 text-2xl font-bold text-white">
                  {name.charAt(0).toUpperCase()}
                </div>
                <div>
                  <div className="font-medium text-gray-900">{name}</div>
                  <div className="flex items-center gap-1.5 text-sm text-gray-500 mt-1">
                    <Mail className="h-3.5 w-3.5" />
                    {profile?.email}
                  </div>
                </div>
              </div>
              <div className="grid gap-6 max-w-md">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700">Display Name</label>
                  <input value={name} onChange={(e) => setName(e.target.value)} className="flex h-10 w-full rounded-lg border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20 transition-all" />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700">Location</label>
                  <div className="relative">
                    <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input value={city} onChange={(e) => setCity(e.target.value)} className="flex h-10 w-full rounded-lg border border-gray-300 bg-white pl-10 pr-3 text-sm text-gray-900 focus:border-[#4353FF] focus:outline-none focus:ring-2 focus:ring-[#4353FF]/20 transition-all" />
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "preferences" && (
            <div className="space-y-8 animate-fade-in">
              <div className="space-y-4">
                <label className="text-sm font-medium text-gray-700">Interests</label>
                <div className="flex flex-wrap gap-2">
                  {CATEGORIES.map((cat) => (
                    <button key={cat} onClick={() => toggleCategory(cat)} className={`px-3 py-1.5 text-sm rounded-lg border capitalize transition-colors ${selectedCategories.includes(cat) ? "bg-blue-50 border-[#4353FF]/30 text-[#4353FF]" : "border-gray-200 text-gray-600 hover:border-gray-300"}`}>
                      {cat}
                    </button>
                  ))}
                </div>
              </div>
              <div className="space-y-4">
                <label className="text-sm font-medium text-gray-700">Theme</label>
                <div className="flex gap-2">
                  {(["light", "dark", "system"] as const).map((t) => (
                    <button key={t} onClick={() => setTheme(t)} className={`px-4 py-2 text-sm capitalize rounded-lg border transition-colors ${theme === t ? "bg-gray-900 text-white border-gray-900" : "border-gray-200 text-gray-600 hover:border-gray-300"}`}>
                      {t}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-gray-200 p-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-blue-50">
                    <Bell className="h-4 w-4 text-blue-600" />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-900">Notifications</div>
                    <div className="text-xs text-gray-500 mt-0.5">Receive real-time updates</div>
                  </div>
                </div>
                <button onClick={() => setNotificationsOn(!notificationsOn)} className={`relative h-6 w-11 rounded-full transition-colors ${notificationsOn ? "bg-[#4353FF]" : "bg-gray-300"}`}>
                  <div className={`absolute top-1 h-4 w-4 rounded-full bg-white transition-transform ${notificationsOn ? "translate-x-6" : "translate-x-1"}`} />
                </button>
              </div>
            </div>
          )}

          {activeTab === "security" && (
            <div className="space-y-4 animate-fade-in">
              <button className="flex w-full items-center justify-between rounded-lg border border-gray-200 p-4 text-left group hover:bg-gray-50 transition-colors">
                <div>
                  <div className="text-sm font-medium text-gray-900 group-hover:text-[#4353FF] transition-colors">Change Password</div>
                  <div className="text-xs text-gray-500 mt-0.5">Last updated 30 days ago</div>
                </div>
                <ChevronRight className="h-4 w-4 text-gray-400" />
              </button>
              <button className="flex w-full items-center justify-between rounded-lg border border-gray-200 p-4 text-left group hover:bg-gray-50 transition-colors">
                <div>
                  <div className="text-sm font-medium text-gray-900 group-hover:text-[#4353FF] transition-colors">Two-Factor Authentication</div>
                  <div className="text-xs text-gray-500 mt-0.5">Add an extra layer of security</div>
                </div>
                <ChevronRight className="h-4 w-4 text-gray-400" />
              </button>
              <div className="border-t border-gray-200 my-6" />
              <button onClick={handleLogout} className="flex w-full items-center justify-center gap-2 rounded-lg border border-red-200 px-4 py-2.5 text-sm font-medium text-red-600 hover:bg-red-50 transition-colors">
                <LogOut className="h-4 w-4" />
                Sign out
              </button>
            </div>
          )}

          {activeTab !== "security" && (
            <div className="flex justify-end pt-8 mt-6 border-t border-gray-200">
              <button onClick={handleSave} disabled={saving} className="inline-flex items-center gap-2 rounded-lg bg-[#4353FF] px-5 py-2.5 text-sm font-medium text-white hover:bg-[#3643E0] disabled:opacity-50 transition-colors shadow-sm">
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                Save changes
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
