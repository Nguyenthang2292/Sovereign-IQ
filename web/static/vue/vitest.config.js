import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [vue()],
  test: {
    globals: false,
    environment: 'jsdom',
    // Ensure that the setup file exists at ./tests/setup.js relative to this config file.
    // If it does not exist, please create it with any necessary test setup code.
    setupFiles: ['./tests/setup.js', './tests/setup-mermaid-global.js'],
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
})

