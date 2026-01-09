import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

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
export default defineConfig(async ({ mode }) => {
  const plugins = [vue()]
  
  // Only load imagemin plugin in production builds to avoid import issues in dev
  if (mode === 'production') {
    try {
      const imageminModule = await import('vite-plugin-imagemin')
      const viteImagemin = imageminModule.default || imageminModule
      plugins.push(
        viteImagemin({
          gifsicle: {
            optimizationLevel: 3, // Valid range: 1-3 (3 is maximum)
            interlaced: false,
          },
          optipng: {
            optimizationLevel: 7, // Valid range: 0-7 (7 is maximum)
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
          webp: {
            quality: 80,
          },
        })
      )
    } catch (error) {
      console.warn(
        'vite-plugin-imagemin not available, skipping image optimization:',
        error?.message || error
      )
    }
  }
  
  return {    
    plugins,
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
  }
})
