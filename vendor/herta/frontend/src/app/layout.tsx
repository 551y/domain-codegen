import type { Metadata } from "next";
import { Outfit, JetBrains_Mono } from "next/font/google";
import { SideNav } from "@/components/nav";
import { siteConfig } from "@/config/site";
import { cn } from "@/lib/utils";
import "@/style/globals.css";
import { Providers } from "./providers";

const outfit = Outfit({ subsets: ["latin", "latin-ext"], variable: "--font-sans" });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: siteConfig.title,
  description: siteConfig.description,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className={cn("bg-background font-sans", outfit.variable, jetbrains.variable)}>
        <Providers>
          <div className="flex min-h-[100dvh]">
            <SideNav />
            <div className="flex-grow overflow-auto">{children}</div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
