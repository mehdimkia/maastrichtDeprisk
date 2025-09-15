/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: { ignoreDuringBuilds: true },   // let production build succeed despite ESLint errors
  // typescript: { ignoreBuildErrors: true }, // optional; only if TS errors ever block builds
};
export default nextConfig;
