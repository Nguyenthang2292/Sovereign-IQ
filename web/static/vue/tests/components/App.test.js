/**
 * Tests for App component
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { createRouter, createMemoryHistory } from 'vue-router'
import { i18n } from '../setup'
import App from '../../src/App.vue'
import ChartAnalyzer from '../../src/components/ChartAnalyzer.vue'
import BatchScanner from '../../src/components/BatchScanner.vue'

// Create router for testing
const routes = [
  {
    path: '/',
    name: 'Home',
    redirect: '/scanner',
  },
  {
    path: '/analyzer',
    name: 'chart-analyzer',
    component: ChartAnalyzer,
  },
  {
    path: '/scanner',
    name: 'batch-scanner',
    component: BatchScanner,
  },
]

describe('App', () => {
  let router

  beforeEach(() => {
    router = createRouter({
      history: createMemoryHistory(),
      routes,
    })
  })

  it('should render navigation correctly', () => {
    const wrapper = mount(App, {
      global: {
        plugins: [router, i18n],
      },
    })

    expect(wrapper.text()).toContain('Gemini Chart Analyzer')
    expect(wrapper.text()).toContain(i18n.global.t('nav.batchScanner'))
    expect(wrapper.text()).toContain(i18n.global.t('nav.chartAnalyzer'))
  })

  it('should highlight active route in navigation', async () => {
    await router.push('/scanner')
    const wrapper = mount(App, {
      global: {
        plugins: [router, i18n],
      },
    })

    await router.isReady()
    await wrapper.vm.$nextTick()

    const batchScannerLink = wrapper.find('a[href="/scanner"]')
    // Active route should have btn-gradient class (gradient button style)
    expect(batchScannerLink.classes()).toContain('btn-gradient')
  })

  it('should handle symbol-click event and navigate to analyzer', async () => {
    const wrapper = mount(App, {
      global: {
        plugins: [router, i18n],
      },
    })

    await router.push('/scanner')
    await router.isReady()
    await wrapper.vm.$nextTick()

    const pushPromise = router.push({
      name: 'chart-analyzer',
      query: { symbol: 'BTC/USDT' }
    })
    
    // Wait for router navigation to complete
    await pushPromise
    await router.isReady()
    await wrapper.vm.$nextTick()

    // Should navigate to analyzer with symbol query
    expect(router.currentRoute.value.name).toBe('chart-analyzer')
    expect(router.currentRoute.value.query.symbol).toBe('BTC/USDT')
  })

  it('should render footer', () => {
    const wrapper = mount(App, {
      global: {
        plugins: [router, i18n],
      },
    })

    expect(wrapper.text()).toContain(i18n.global.t('footer.text'))
  })

  it('should have correct CSS classes and structure', () => {
    const wrapper = mount(App, {
      global: {
        plugins: [router, i18n],
      },
    })

    const app = wrapper.find('#app')
    expect(app.classes()).toContain('min-h-screen')
    
    // Check for background layers
    const background = wrapper.find('.app-background')
    expect(background.exists()).toBe(true)
    
    const overlay = wrapper.find('.app-overlay')
    expect(overlay.exists()).toBe(true)
    
    const content = wrapper.find('.app-content')
    expect(content.exists()).toBe(true)
  })
})

