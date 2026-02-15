import type { Metadata } from "next";
import Navbar from "../components/Navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Image Generator - Powered by Stable Diffusion",
  description: "Generate stunning AI images with our custom trained LoRA model. Transform text into beautiful artwork, portraits, and creative visuals."
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <main>{children}</main>
      </body>
    </html>
  );
}
