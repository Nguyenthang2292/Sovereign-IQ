import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import i18n from './i18n'
import './style.css'

const app = createApp(App)

// Optional: Global error handler for better production error tracking
if (import.meta.env.PROD) {
    app.config.errorHandler = (err, _instance, info) => {
        // Replace the following line with integration to your error reporting service if needed
        console.error('Global error handler:', err, info)
    }
}

app.use(router)
app.use(i18n)

app.mount('#app')
