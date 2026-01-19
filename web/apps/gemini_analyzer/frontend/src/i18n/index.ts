import { createI18n } from 'vue-i18n'
import vi from './locales/vi.json'
import en from './locales/en.json'

const STORAGE_KEY = 'gemini-chart-analyzer-locale'

// Get saved locale from localStorage, default to 'vi'
function getSavedLocale(): string {
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

const SUPPORTED_LOCALES = ['vi', 'en'] as const
type Locale = typeof SUPPORTED_LOCALES[number]

// Function to change locale and save to localStorage
export function setLocale(locale: string): void {
    if (SUPPORTED_LOCALES.includes(locale as Locale)) {
        i18n.global.locale.value = locale as Locale
        try {
            localStorage.setItem(STORAGE_KEY, locale)
        } catch (e) {
            console.warn('Failed to save locale to localStorage:', e)
        }
    }
}

// Function to get current locale
export function getLocale(): Locale {
    return i18n.global.locale.value as Locale
}

export default i18n
