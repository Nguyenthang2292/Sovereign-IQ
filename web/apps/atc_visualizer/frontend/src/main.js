import { createApp } from 'vue'
import VueApexCharts from "vue3-apexcharts"
import App from './App.vue'

// Import shared styles first
import '@shared/styles/variables.css'
import '@shared/styles/base.css'
import '@shared/styles/components.css'
import '@shared/styles/layouts.css'
import '@shared/styles/effects.css'

// Import app styles (Tailwind config)
import './style.css'

const app = createApp(App)
app.use(VueApexCharts)
app.mount('#app')
