/**
 * Test setup file for Vue tests
 */
import { vi } from 'vitest'

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor(callback, options) {
    this.callback = callback
    this.options = options
    this.entries = []
    this.observedTargets = new Set()
  }
  disconnect() {
    this.entries = []
    this.observedTargets.clear()
  }
  observe(target) {
    if (!target || this.observedTargets.has(target)) {
      return
    }
    this.observedTargets.add(target)
    
    // Create a properly shaped IntersectionObserverEntry
    const entry = {
      boundingClientRect: {
        bottom: 0,
        height: 0,
        left: 0,
        right: 0,
        top: 0,
        width: 0,
        x: 0,
        y: 0,
        toJSON: () => ({})
      },
      intersectionRatio: 0,
      intersectionRect: {
        bottom: 0,
        height: 0,
        left: 0,
        right: 0,
        top: 0,
        width: 0,
        x: 0,
        y: 0,
        toJSON: () => ({})
      },
      isIntersecting: false,
      rootBounds: null,
      target: target,
      time: Date.now()
    }
    
    this.entries.push(entry)
    
    // Immediately invoke callback with the entry
    if (this.callback) {
      // Use setTimeout to ensure callback is invoked asynchronously (matching real behavior)
      setTimeout(() => {
        this.callback([entry], this)
      }, 0)
    }
  }
  takeRecords() {
    const records = [...this.entries]
    this.entries = []
    return records
  }
  unobserve(target) {
    if (!target) {
      return
    }
    this.observedTargets.delete(target)
    // Remove any entries matching this target
    this.entries = this.entries.filter(entry => entry.target !== target)
  }
}

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor(callback) {
    this.callback = callback
    this.entries = []
    this.observedElements = new Set()
  }
  disconnect() {
    this.entries = []
    this.observedElements.clear()
  }  
  observe(target) {
    if (this.observedElements.has(target)) {
      // Already observing this target, do nothing (idempotent)
      return
    }
    this.observedElements.add(target)
    const entry = { target }
    this.entries.push(entry)
    if (typeof this.callback === 'function') {
      this.callback([entry], this)
    }
  }
  unobserve(target) {
    this.observedElements.delete(target)
    this.entries = this.entries.filter(entry => entry.target !== target)
  }
}

// Mock scrollTo and scrollIntoView for HTMLElement
if (typeof HTMLElement !== 'undefined') {
  if (!HTMLElement.prototype.scrollTo) {
    HTMLElement.prototype.scrollTo = function (xOrOptions, y) {
      if (typeof xOrOptions === 'object' && xOrOptions !== null) {
        if (typeof xOrOptions.left === 'number') {
          this.scrollLeft = xOrOptions.left;
        }
        if (typeof xOrOptions.top === 'number') {
          this.scrollTop = xOrOptions.top;
        }
      } else if (typeof xOrOptions === 'number') {
        this.scrollLeft = xOrOptions;
        if (typeof y === 'number') {
          this.scrollTop = y;
        }
      }
    };
  }

  if (!HTMLElement.prototype.scrollIntoView) {
    HTMLElement.prototype.scrollIntoView = function(/*options*/) {
      // No-op in test environment
    };
  }
}


// Mock vue-i18n
import { createI18n } from 'vue-i18n'
import viLocales from '../src/i18n/locales/vi.json'
import enLocales from '../src/i18n/locales/en.json'

const i18n = createI18n({
  legacy: false,
  locale: 'vi',
  fallbackLocale: 'vi',
  messages: {
    vi: viLocales,
    en: enLocales,
  },
})

// Make i18n available globally for tests
global.i18n = i18n

// Export i18n for use in test files
export { i18n }

// Helper function to mount components with i18n
import { mount } from '@vue/test-utils'

export function mountWithI18n(component, options = {}) {
  return mount(component, {
    ...options,
    global: {
      ...(options.global || {}),
      plugins: [i18n, ...(options.global?.plugins || [])],
    },
  })
}

