<template>
  <div id="app" class="min-h-screen flex flex-col">
    <!-- Background Layer -->
    <div class="app-background"></div>
    <div class="app-overlay"></div>
    
    <!-- Content -->
    <div class="app-content flex min-h-screen">
      <!-- Sidebar -->
      <aside 
        :class="[
          'sidebar',
          { 'collapsed': sidebarCollapsed },
          { 'mobile-open': mobileMenuOpen }
        ]"
      >
        <!-- Logo/Title Section -->
        <div class="sidebar-header">
          <div class="flex items-center gap-3" v-if="!sidebarCollapsed">
            <div class="sidebar-logo">
              <span class="text-2xl">📊</span>
            </div>
            <h1 class="text-lg font-semibold text-white">{{ $t('app.title') }}</h1>
          </div>
          <div class="sidebar-logo" v-else>
            <span class="text-2xl">📊</span>
          </div>
        </div>

        <!-- Navigation Links -->
        <nav class="sidebar-nav">
          <router-link
            to="/scanner"
            :class="[
              'sidebar-link',
              $route.name === 'batch-scanner' ? 'active' : ''
            ]"
            @click="closeMobileMenu"
          >
            <span class="sidebar-icon">📋</span>
            <span class="sidebar-text" v-if="!sidebarCollapsed">{{ $t('nav.batchScanner') }}</span>
          </router-link>
          <router-link
            to="/analyzer"
            :class="[
              'sidebar-link',
              $route.name === 'chart-analyzer' ? 'active' : ''
            ]"
            @click="closeMobileMenu"
          >
            <span class="sidebar-icon">📈</span>
            <span class="sidebar-text" v-if="!sidebarCollapsed">{{ $t('nav.chartAnalyzer') }}</span>
          </router-link>
          <router-link
            to="/workflow"
            :class="[
              'sidebar-link',
              $route.name === 'workflow-diagrams' ? 'active' : ''
            ]"
            @click="closeMobileMenu"
          >
            <span class="sidebar-icon">🔄</span>
            <span class="sidebar-text" v-if="!sidebarCollapsed">{{ $t('nav.workflowDiagrams') }}</span>
          </router-link>        
        </nav>

        <!-- Footer Section -->
        <div class="sidebar-footer">
          <!-- Language Selector -->
          <button
            @click="toggleLanguage"
            class="sidebar-footer-btn"
            :title="currentLocale === 'vi' ? 'Switch to English' : 'Chuyển sang Tiếng Việt'"
          >
            <img 
              :src="currentLocale === 'vi' ? flagVi : flagEn" 
              :alt="currentLocale === 'vi' ? 'Vietnam flag' : 'UK flag'"
              class="footer-icon"
            />
            <span class="sidebar-text" v-if="!sidebarCollapsed">
              {{ currentLocale === 'vi' ? 'Tiếng Việt' : 'English' }}
            </span>
          </button>
        </div>
      </aside>

      <!-- Collapse Toggle Button (Outside sidebar, moves with it) -->
      <button
        @click="toggleSidebar"
        :class="[
          'sidebar-collapse-btn',
          { 'collapsed': sidebarCollapsed }
        ]"
        v-if="!isMobile"
        :title="sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'"
        :aria-label="sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'"
      >
        <span class="collapse-icon" :class="{ 'rotated': sidebarCollapsed }">◀</span>
      </button>

      <!-- Mobile Menu Button (floating) -->
      <button
        @click="toggleMobileMenu"
        class="mobile-menu-button"
        v-if="isMobile"
        :aria-label="mobileMenuOpen ? 'Close menu' : 'Open menu'"
      >
        <span v-if="!mobileMenuOpen">☰</span>
        <span v-else>✕</span>
      </button>

      <!-- Mobile Sidebar Overlay -->
      <div
        v-if="isMobile && mobileMenuOpen"
        class="mobile-overlay"
        @click="closeMobileMenu"
      ></div>

      <!-- Main Content -->
      <main class="main-content">
        <div class="main-content-wrapper">
          <router-view @symbol-click="handleSymbolClick" />
        </div>
        
        <!-- Footer -->
        <footer class="glass-panel border-t border-gray-700/50 mt-auto">
          <div class="max-w-7xl mx-auto px-6 py-4 text-center text-gray-400 text-sm">
            <p>{{ $t('footer.text') }}</p>
          </div>
        </footer>
      </main>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { computed, ref, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { setLocale } from './i18n'
import flagVi from './assets/icons/flag-vi.svg'
import flagEn from './assets/icons/flag-en.svg'

const router = useRouter()
const { locale } = useI18n()

// Use reactive locale from vue-i18n for proper reactivity in UI
const currentLocale = computed(() => locale.value)

// Sidebar state
const sidebarCollapsed = ref(false)
const mobileMenuOpen = ref(false)
const isMobile = ref(false)

// Check if mobile on mount and resize
function checkMobile() {
  isMobile.value = window.innerWidth < 768
  if (isMobile.value) {
    mobileMenuOpen.value = false
  }
}

onMounted(() => {
  checkMobile()
  window.addEventListener('resize', checkMobile)
  
  // Load sidebar collapse preference from localStorage
  const saved = localStorage.getItem('sidebarCollapsed')
  if (saved !== null && !isMobile.value) {
    sidebarCollapsed.value = saved === 'true'
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', checkMobile)
})

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
  localStorage.setItem('sidebarCollapsed', sidebarCollapsed.value.toString())
}

function toggleMobileMenu() {
  mobileMenuOpen.value = !mobileMenuOpen.value
}

function closeMobileMenu() {
  mobileMenuOpen.value = false
}

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

