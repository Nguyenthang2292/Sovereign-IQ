<template>
  <div id="app" class="min-h-screen flex flex-col">
    <!-- Background Layer -->
    <div class="app-background"></div>
    <div class="app-overlay"></div>
    
    <!-- Content -->
    <div class="app-content flex flex-col min-h-screen">
      <!-- Navigation -->
      <nav class="glass-nav sticky top-0 z-50">
      <div class="max-w-7xl mx-auto px-6 py-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="text-2xl">📊</span>
            <h1 class="text-xl font-bold text-white">{{ $t('app.title') }}</h1>
          </div>
          <div class="flex gap-4 items-center">
            <router-link
              to="/scanner"
              :class="[
                'px-4 py-2 rounded-lg font-medium transition-all duration-300',
                $route.name === 'batch-scanner'
                  ? 'btn-gradient text-white hover:shadow-glow-purple hover:scale-[1.05] active:scale-[0.95]'
                  : 'text-gray-300 hover:bg-gray-700/50 hover:text-white backdrop-blur-sm border border-transparent hover:border-gray-600/50'
              ]"
            >
              {{ $t('nav.batchScanner') }}
            </router-link>
            <router-link
              to="/analyzer"
              :class="[
                'px-4 py-2 rounded-lg font-medium transition-all duration-300',
                $route.name === 'chart-analyzer'
                  ? 'btn-gradient text-white hover:shadow-glow-purple hover:scale-[1.05] active:scale-[0.95]'
                  : 'text-gray-300 hover:bg-gray-700/50 hover:text-white backdrop-blur-sm border border-transparent hover:border-gray-600/50'
              ]"
            >
              {{ $t('nav.chartAnalyzer') }}
            </router-link>
            <button
              @click="toggleLanguage"
              class="flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-all duration-300 text-gray-300 hover:bg-gray-700/50 hover:text-white border border-gray-600/50 hover:border-gray-500/50 backdrop-blur-sm hover:scale-105 active:scale-95"
              :title="currentLocale === 'vi' ? 'Switch to English' : 'Chuyển sang Tiếng Việt'"
            >
              <img 
                :src="currentLocale === 'vi' ? flagVi : flagEn" 
                :alt="currentLocale === 'vi' ? 'Vietnam flag' : 'UK flag'"
                class="w-5 h-4 object-cover rounded-sm"
              />
              <span class="text-sm font-semibold">{{ currentLocale === 'vi' ? 'VI' : 'EN' }}</span>
            </button>
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="py-6 flex-1">
      <router-view @symbol-click="handleSymbolClick" />
    </main>

    <!-- Footer -->
    <footer class="glass-panel border-t border-gray-700/50 mt-auto">
      <div class="max-w-7xl mx-auto px-6 py-4 text-center text-gray-400 text-sm">
        <p>{{ $t('footer.text') }}</p>
      </div>
    </footer>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { setLocale } from './i18n'
import flagVi from './assets/icons/flag-vi.svg'
import flagEn from './assets/icons/flag-en.svg'

const router = useRouter()
const { locale } = useI18n()

// Use reactive locale from vue-i18n for proper reactivity in UI
const currentLocale = computed(() => locale.value)

function toggleLanguage() {
  const newLocale = currentLocale.value === 'vi' ? 'en' : 'vi'
  setLocale(newLocale)
}

function handleSymbolClick(symbol) {
  // Navigate to analyzer with symbol pre-filled
  router.push({
    name: 'chart-analyzer',
    query: { symbol }
  })
}
</script>

