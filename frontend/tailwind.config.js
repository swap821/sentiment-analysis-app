/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: { bg: '#0a0b10', card: '#10121a' },
        primary: '#4d7dff',
        positive: '#10b981',
        negative: '#ef4444',
      },
    },
  },
  plugins: [],
}