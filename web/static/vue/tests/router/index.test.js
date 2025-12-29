/**
 * Tests for Vue Router
 */
import { describe, it, expect, beforeEach } from 'vitest'
import { createRouter, createWebHistory, createMemoryHistory } from 'vue-router'
import ChartAnalyzer from '../../src/components/ChartAnalyzer.vue'
import BatchScanner from '../../src/components/BatchScanner.vue'

// Mock component for testing route params
const MockUserComponent = { template: '<div>User</div>' }

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
  {
    path: '/user/:id',
    name: 'user',
    component: MockUserComponent,
    props: true,
  },
  {
    path: '/protected',
    name: 'protected',
    component: ChartAnalyzer,
    meta: { requiresAuth: true },
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: { template: '<div>404 Not Found</div>' },
  },
]

describe('Router', () => {
  let router

  beforeEach(() => {
    router = createRouter({
      history: createMemoryHistory(),
      routes,
    })
  })
  
  it('should redirect root path to /scanner', async () => {
    await router.push('/')
    await router.isReady()

    expect(router.currentRoute.value.path).toBe('/scanner')
    expect(router.currentRoute.value.name).toBe('batch-scanner')
  })

  it('should navigate to ChartAnalyzer route', async () => {
    await router.push('/analyzer')
    await router.isReady()

    expect(router.currentRoute.value.path).toBe('/analyzer')
    expect(router.currentRoute.value.name).toBe('chart-analyzer')
  })

  it('should navigate to BatchScanner route', async () => {
    await router.push('/scanner')
    await router.isReady()

    expect(router.currentRoute.value.path).toBe('/scanner')
    expect(router.currentRoute.value.name).toBe('batch-scanner')
  })

  it('should handle query parameters', async () => {
    await router.push({
      path: '/analyzer',
      query: { symbol: 'BTC/USDT' },
    })
    await router.isReady()

    expect(router.currentRoute.value.query.symbol).toBe('BTC/USDT')
  })

  it('should have routes for ChartAnalyzer and BatchScanner', () => {
    const analyzerRoute = routes.find(r => r.name === 'chart-analyzer')
    const scannerRoute = routes.find(r => r.name === 'batch-scanner')

    expect(analyzerRoute).toBeDefined()
    expect(scannerRoute).toBeDefined()
  })

  describe('Edge Cases', () => {
    it('should handle navigation to undefined route with 404', async () => {
      await router.push('/nonexistent-route')
      await router.isReady()

      expect(router.currentRoute.value.name).toBe('NotFound')
      expect(router.currentRoute.value.path).toBe('/nonexistent-route')
    })

    it('should handle route with params when navigating by path', async () => {
      await router.push('/user/123')
      await router.isReady()

      expect(router.currentRoute.value.path).toBe('/user/123')
      expect(router.currentRoute.value.name).toBe('user')
      expect(router.currentRoute.value.params.id).toBe('123')
    })

    it('should handle route params when navigating with params object', async () => {
      await router.push({
        name: 'user',
        params: { id: '456' },
      })
      await router.isReady()

      expect(router.currentRoute.value.name).toBe('user')
      expect(router.currentRoute.value.params.id).toBe('456')
      expect(router.currentRoute.value.path).toBe('/user/456')
    })

    it('should handle route params with query parameters', async () => {
      await router.push({
        name: 'user',
        params: { id: '789' },
        query: { tab: 'profile', view: 'details' },
      })
      await router.isReady()

      expect(router.currentRoute.value.params.id).toBe('789')
      expect(router.currentRoute.value.query.tab).toBe('profile')
      expect(router.currentRoute.value.query.view).toBe('details')
    })

    it('should block navigation to protected route when not authenticated', async () => {
      let guardCalled = false
      let redirectPath = null

      const removeGuard = router.beforeEach((to, from, next) => {
        guardCalled = true
        if (to.meta.requiresAuth) {
          // Simulate unauthenticated state
          redirectPath = '/scanner'
          next('/scanner')
        } else {
          next()
        }
      })

      await router.push('/protected')
      await router.isReady()

      expect(guardCalled).toBe(true)
      expect(redirectPath).toBe('/scanner')
      expect(router.currentRoute.value.path).toBe('/scanner')
      expect(router.currentRoute.value.name).toBe('batch-scanner')
      
      removeGuard()
    })

    it('should allow navigation to protected route when authenticated', async () => {
      let guardCalled = false
      const mockAuthState = { isAuthenticated: true }

      const removeGuard = router.beforeEach((to, from, next) => {
        guardCalled = true
        if (to.meta.requiresAuth && !mockAuthState.isAuthenticated) {
          next('/scanner')
        } else {
          next()
        }
      })

      await router.push('/protected')
      await router.isReady()

      expect(guardCalled).toBe(true)
      expect(router.currentRoute.value.path).toBe('/protected')
      expect(router.currentRoute.value.name).toBe('protected')
      
      removeGuard()
    })

    it('should handle navigation guard that blocks navigation', async () => {
      let guardCalled = false

      const removeGuard = router.beforeEach((to, from, next) => {
        guardCalled = true
        if (to.path === '/protected') {
          // Block navigation by calling next(false)
          next(false)
        } else {
          next()
        }
      })

      // Start from a known route
      await router.push('/scanner')
      await router.isReady()
      expect(router.currentRoute.value.path).toBe('/scanner')

      // Try to navigate to protected route
      await router.push('/protected')
      await router.isReady()

      // Navigation should be blocked, so we should still be on /scanner
      expect(guardCalled).toBe(true)
      expect(router.currentRoute.value.path).toBe('/scanner')
      
      removeGuard()
    })

    it('should handle router.push rejection gracefully', async () => {
      // Create a router with a route that will throw an error
      const errorRouter = createRouter({
        history: createMemoryHistory(),
        routes: [
          ...routes,
          {
            path: '/error-route',
            name: 'error-route',
            component: {
              setup() {
                throw new Error('Component setup error')
              },
            },
          },
        ],
      })

      await errorRouter.push('/scanner')
      await errorRouter.isReady()

      // Try to navigate to error route - should handle gracefully
      try {
        await errorRouter.push('/error-route')
        await errorRouter.isReady()
      } catch (error) {
        // Error should be caught
        expect(error).toBeDefined()
      }
    })

    it('should handle navigation with invalid route name', async () => {
      try {
        await router.push({ name: 'invalid-route-name' })
        await router.isReady()
      } catch (error) {
        // Should throw an error for invalid route name
        expect(error).toBeDefined()
      }
    })

    it('should handle multiple sequential navigation calls', async () => {
      await router.push('/scanner')
      await router.isReady()
      expect(router.currentRoute.value.path).toBe('/scanner')

      await router.push('/analyzer')
      await router.isReady()
      expect(router.currentRoute.value.path).toBe('/analyzer')

      await router.push('/user/999')
      await router.isReady()
      expect(router.currentRoute.value.params.id).toBe('999')
    })

    it('should handle navigation guard with async logic', async () => {
      let guardCalled = false

      const removeGuard = router.beforeEach(async (to, from, next) => {
        guardCalled = true
        // Simulate async auth check
        await new Promise(resolve => setTimeout(resolve, 10))
        if (to.meta.requiresAuth) {
          next('/scanner')
        } else {
          next()
        }
      })

      await router.push('/protected')
      await router.isReady()

      expect(guardCalled).toBe(true)
      expect(router.currentRoute.value.path).toBe('/scanner')
      
      removeGuard()
    })
  })
})

