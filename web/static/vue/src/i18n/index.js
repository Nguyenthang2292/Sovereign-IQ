import { createI18n } from 'vue-i18n'
import vi from './locales/vi.json'
import en from './locales/en.json'

const STORAGE_KEY = 'gemini-chart-analyzer-locale'

// Get saved locale from localStorage, default to 'vi'
function getSavedLocale() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved && (saved === 'vi' || saved === 'en') ? saved : 'vi'
  } catch (e) {
    return 'vi'
  }
}

// Create i18n instance
const i18n = createI18n({
  legacy: false, // Use Composition API mode
  locale: getSavedLocale(),
  fallbackLocale: 'vi',
  messages: {
    vi,
    en,
  },
})

const SUPPORTED_LOCALES = ['vi', 'en']
const DEFAULT_LOCALE = 'vi'

// Function to change locale and save to localStorage
export function setLocale(locale) {
  if (SUPPORTED_LOCALES.includes(locale)) {
    i18n.global.locale.value = locale
    try {
      localStorage.setItem(STORAGE_KEY, locale)
    } catch (e) {
      console.warn('Failed to save locale to localStorage:', e)
    }
  }
}

// Function to get current locale
export function getLocale() {
  return i18n.global.locale.value
}

export default i18n

