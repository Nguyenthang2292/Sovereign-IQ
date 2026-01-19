import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'

const ChartAnalyzer = () => import('../components/ChartAnalyzer.vue')
const BatchScanner = () => import('../components/BatchScanner.vue')
const WorkflowDiagrams = () => import('../components/WorkflowDiagrams.vue')
const NotFound = {
    template: `
      <div style="text-align: center; padding: 2rem;">
        <h1>404 - Page Not Found</h1>
        <p>The page you're looking for doesn't exist.</p>
        <a href="/">Return to Home</a>
      </div>
    `
}

const routes: Array<RouteRecordRaw> = [
    {
        path: '/',
        name: 'Home',
        redirect: '/scanner'
    },
    {
        path: '/analyzer',
        name: 'chart-analyzer',
        component: ChartAnalyzer
    },
    {
        path: '/scanner',
        name: 'batch-scanner',
        component: BatchScanner
    },
    {
        path: '/workflow',
        name: 'workflow-diagrams',
        component: WorkflowDiagrams
    },
    {
        path: '/:catchAll(.*)',
        name: 'NotFound',
        component: NotFound
    }
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

export default router
