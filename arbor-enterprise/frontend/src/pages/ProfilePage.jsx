import { useState } from 'react';
import { User, Mail, Shield, Palette, Save } from 'lucide-react';

import { useAuthStore } from '@/stores/authStore';
import { useTheme } from '@/components/ThemeProvider';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

export default function ProfilePage() {
  const { user } = useAuthStore();
  const { theme, setTheme } = useTheme();

  if (!user) return null;

  const roleColors = {
    admin: 'bg-red-500/10 text-red-600 border-red-500/20',
    curator: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
    viewer: 'bg-green-500/10 text-green-600 border-green-500/20',
    api_client: 'bg-purple-500/10 text-purple-600 border-purple-500/20',
  };

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <div>
        <h2 className="text-lg font-semibold">Profile</h2>
        <p className="text-sm text-muted-foreground">
          Manage your account settings and preferences
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Account Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <User className="size-8" />
            </div>
            <div>
              <h3 className="font-semibold">{user.name}</h3>
              <p className="text-sm text-muted-foreground">{user.email}</p>
              <Badge className={`mt-1 capitalize ${roleColors[user.role] || ''}`}>
                {user.role}
              </Badge>
            </div>
          </div>

          <Separator />

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label>Name</Label>
              <Input value={user.name || ''} disabled />
            </div>
            <div className="space-y-2">
              <Label>Email</Label>
              <Input value={user.email || ''} disabled />
            </div>
            <div className="space-y-2">
              <Label>Role</Label>
              <Input value={user.role || ''} disabled className="capitalize" />
            </div>
            <div className="space-y-2">
              <Label>User ID</Label>
              <Input value={user.id || ''} disabled className="font-mono text-xs" />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Preferences</CardTitle>
          <CardDescription>Customize your experience</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <Label>Theme</Label>
              <p className="text-xs text-muted-foreground">
                Choose your preferred color scheme
              </p>
            </div>
            <Select value={theme} onValueChange={setTheme}>
              <SelectTrigger className="w-[120px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="dark">Dark</SelectItem>
                <SelectItem value="light">Light</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
