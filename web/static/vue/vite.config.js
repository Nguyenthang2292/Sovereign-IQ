import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'
import viteImagemin from 'vite-plugin-imagemin'

// Determine sourcemap configuration based on environment variable
let sourcemap;
if (process.env.GENERATE_SOURCEMAPS === 'true') {
  sourcemap = 'inline';
} else if (process.env.GENERATE_SOURCEMAPS === 'false') {
  sourcemap = false;
} else {
  sourcemap = 'hidden'; // Default: hidden (safe for production, allows server-side debugging)
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    viteImagemin({
      // Disable image optimization in development for faster builds
      disable: process.env.NODE_ENV === 'development',      
      gifsicle: {
        optimizationLevel: 7,
        interlaced: false,
      },
      optipng: {
        optimizationLevel: 7,
      },
      mozjpeg: {
        quality: 80,
      },
      pngquant: {
        quality: [0.7, 0.9],
        speed: 4,
      },
      svg: {
        plugins: [
          {
            name: 'removeViewBox',
            active: false,
          },          
          {
            name: 'removeEmptyAttrs',
            active: false,
          },
        ],
      },
      // Tạo WebP version nếu có thể
      webp: {
        quality: 80,
      },
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    // Sourcemap configuration: 'hidden' by default (generates sourcemaps but doesn't expose them to browser)
    // Set GENERATE_SOURCEMAPS='true' for inline sourcemaps, or 'false' to disable completely
    sourcemap,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/static': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})

