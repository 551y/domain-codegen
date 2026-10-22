"use client";

import { useTheme } from "next-themes";
import { createContext, useContext, useEffect, useState } from "react";
import { type ITheme, ThemeManager } from "@visactor/vchart";
import { customDarkTheme, customLightTheme } from "@/config/chart-theme";

type ChartTheme = "light" | "dark" | "system";

interface ChartThemeContextI {
  theme: ChartTheme | undefined;
}

export const ChartThemeContext = createContext<ChartThemeContextI>({ theme: undefined });

export function ChartThemeProvider({ children }: { children: React.ReactNode }) {
  const { theme: modeTheme } = useTheme();
  const [theme, setTheme] = useState<ChartTheme>("system");

  useEffect(() => {
    const font = window.getComputedStyle(document.body).getPropertyValue("--font-sans").trim();
    ThemeManager.registerTheme("light", { ...customLightTheme, fontFamily: font || "Outfit, sans-serif" });
    ThemeManager.registerTheme("dark", { ...customDarkTheme, fontFamily: font || "Outfit, sans-serif" });
  }, []);

  useEffect(() => {
    const updateTheme = () => {
      let resolved = modeTheme;
      if (modeTheme === "system") {
        resolved = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      }
      setTheme(resolved as ChartTheme);
      ThemeManager.setCurrentTheme(resolved === "dark" ? "dark" : "light");
    };
    updateTheme();
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => { if (modeTheme === "system") updateTheme(); };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [modeTheme]);

  return <ChartThemeContext.Provider value={{ theme }}>{children}</ChartThemeContext.Provider>;
}

export function useChartTheme() {
  const context = useContext(ChartThemeContext);
  if (!context) throw new Error("useChartTheme must be used within ChartThemeProvider");
  return context;
}
