import type { Metadata } from "next";
import { Sora, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import TopBar from "@/components/TopBar";

const sora = Sora({ subsets: ["latin"], variable: "--font-sans" });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "领域知识增强的无人系统代码智能适配平台",
  description: "一套检索增强（RAG）代码生成框架，接入两套领域知识库：卫星协议代码生成与无人机模型 ONNX 部署适配。",
  icons: { icon: "/icon.svg" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body className={`${sora.variable} ${jetbrains.variable} font-sans`}>
        <TopBar />
        <main className="min-h-[calc(100vh-3.5rem)]">{children}</main>
      </body>
    </html>
  );
}
