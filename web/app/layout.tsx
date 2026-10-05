import type { Metadata, Viewport } from "next";
import { Geist, Noto_Kufi_Arabic } from "next/font/google";
import "./globals.css";

const geist = Geist({ variable: "--font-geist", subsets: ["latin"] });
const kufi = Noto_Kufi_Arabic({ variable: "--font-kufi", subsets: ["arabic"], weight: ["600", "800"] });

export const metadata: Metadata = {
  title: "Qima — what's your laptop worth?",
  description:
    "Fair asking prices for used laptops in Algeria, learned from 16,000 real listings. A tight price range, the reasons behind it, and the deals priced below it.",
};

export const viewport: Viewport = { themeColor: "#ffffff" };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geist.variable} ${kufi.variable}`}>
      <body>{children}</body>
    </html>
  );
}
