import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SpamDetector - AI-Powered Email & Message Analysis",
  description: "Instantly detect spam emails and messages using advanced machine learning. Real-time analysis with confidence scores.",
  keywords: ["spam detector", "email filter", "message analysis", "AI", "machine learning"],
  authors: [{ name: "SpamDetector Team" }],
  openGraph: {
    title: "SpamDetector - Protect Your Inbox",
    description: "Advanced AI-powered spam detection for emails and messages",
    type: "website",
  },
  viewport: "width=device-width, initial-scale=1.0",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
