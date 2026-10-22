"use client";

import { ChartThemeProvider } from "@/components/providers/chart-theme-provider";
import { ModeThemeProvider } from "@/components/providers/mode-theme-provider";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ModeThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
      <ChartThemeProvider>{children}</ChartThemeProvider>
    </ModeThemeProvider>
  );
}
