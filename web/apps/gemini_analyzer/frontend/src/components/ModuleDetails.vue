<template>
  <div class="module-details space-y-4">
    <h3 class="text-xl font-bold text-white mb-4">{{ $t('workflowDiagrams.moduleDetails.title') }}</h3>
    
    <div
      v-for="(module, index) in modules"
      :key="module.name"
      :id="`module-${getModuleId(module.name)}`"
      :class="[
        'glass-panel rounded-lg p-4 border transition-all duration-300',
        highlightedModule === module.name
          ? 'border-purple-500 shadow-lg shadow-purple-500/50'
          : 'border-gray-700/50 hover:border-purple-500/50'
      ]"
    >
      <!-- Module Header -->
      <div
        @click="toggleModule(index)"
        class="cursor-pointer flex items-center justify-between"
      >
        <div class="flex-1">
          <h4 class="text-lg font-semibold text-white mb-1">{{ module.name }}</h4>
          <p class="text-sm text-gray-400">{{ module.description }}</p>
        </div>
        <button
          class="ml-4 text-gray-400 hover:text-white transition-colors"
          :class="{ 'rotate-180': expandedModules[index] }"
          :aria-expanded="expandedModules[index] ? 'true' : 'false'"
          :aria-label="expandedModules[index] ? $t('workflowDiagrams.moduleDetails.collapse') : $t('workflowDiagrams.moduleDetails.expand')"
        >
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      <!-- Expanded Content -->
      <div
        v-if="expandedModules[index]"
        class="mt-4 pt-4 border-t border-gray-700/50 space-y-4"
      >
        <!-- Path -->
        <div>
          <h5 class="text-sm font-semibold text-gray-300 mb-2">{{ $t('workflowDiagrams.moduleDetails.modulePath') }}</h5>
          <code class="text-xs text-purple-400 bg-gray-900/50 px-2 py-1 rounded">{{ module.path }}</code>
        </div>

        <!-- Inputs -->
        <div>
          <h5 class="text-sm font-semibold text-gray-300 mb-2">{{ $t('workflowDiagrams.moduleDetails.inputs') }}</h5>
          <ul class="list-disc list-inside space-y-1">
            <li v-for="(input, i) in module.inputs" :key="i" class="text-sm text-gray-400">
              {{ input }}
            </li>
          </ul>
        </div>

        <!-- Outputs -->
        <div>
          <h5 class="text-sm font-semibold text-gray-300 mb-2">{{ $t('workflowDiagrams.moduleDetails.outputs') }}</h5>
          <ul class="list-disc list-inside space-y-1">
            <li v-for="(output, i) in module.outputs" :key="i" class="text-sm text-gray-400">
              {{ output }}
            </li>
          </ul>
        </div>

        <!-- Key Files -->
        <div v-if="module.keyFiles && module.keyFiles.length > 0">
          <h5 class="text-sm font-semibold text-gray-300 mb-2">{{ $t('workflowDiagrams.moduleDetails.keyFiles') }}</h5>
          <ul class="space-y-1">
            <li v-for="(file, i) in module.keyFiles" :key="i">
              <code class="text-xs text-blue-400 bg-gray-900/50 px-2 py-1 rounded">{{ file }}</code>
            </li>
          </ul>
        </div>

        <!-- Key Functions -->
        <div v-if="module.keyFunctions && module.keyFunctions.length > 0">
          <h5 class="text-sm font-semibold text-gray-300 mb-2">{{ $t('workflowDiagrams.moduleDetails.keyFunctions') }}</h5>
          <ul class="space-y-1">
            <li v-for="(func, i) in module.keyFunctions" :key="i">
              <code class="text-xs text-green-400 bg-gray-900/50 px-2 py-1 rounded">{{ func }}</code>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

interface ModuleInfo {
  name: string
  description: string
  path: string
  inputs: string[]
  outputs: string[]
  keyFiles?: string[]
  keyFunctions?: string[]
}

interface Props {
  modules: ModuleInfo[]
  highlightedModule?: string | null
}

const props = withDefaults(defineProps<Props>(), {
  highlightedModule: null
})

const expandedModules = ref<Record<number, boolean>>({})

function toggleModule(index: number) {
  expandedModules.value[index] = !expandedModules.value[index]
}

function getModuleId(moduleName: string): string {
  // Convert module name to a valid ID
  return moduleName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
}

// Expose method to scroll to module
function scrollToModule(moduleName: string) {
  const moduleId = getModuleId(moduleName)
  const element = document.getElementById(`module-${moduleId}`)
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'center' })
    // Auto-expand the module
    const index = props.modules.findIndex(m => m.name === moduleName)
    if (index !== -1) {
      expandedModules.value[index] = true
    }
  } else {
    // Try to find by text content as fallback
    // const fallback = ...
    // No debug output
  }
}

// Expose methods
defineExpose({
  scrollToModule,
  getModuleId
})
</script>

<style scoped>
code {
  font-family: 'Courier New', monospace;
}
</style>
