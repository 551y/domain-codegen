/** @type {import('next').NextConfig} */

// 编排后端地址（可用环境变量覆盖）
const DEMO_API = process.env.DEMO_API_URL || "http://localhost:8200";

const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${DEMO_API}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
