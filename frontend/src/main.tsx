import { Amplify } from 'aws-amplify'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import outputs from '../amplify_outputs.json'
import { App } from './App'
import './index.css'

Amplify.configure(outputs)

const root = document.getElementById('root')
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}
