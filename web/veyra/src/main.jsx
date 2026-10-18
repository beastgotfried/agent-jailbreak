import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './styles.css'

if (new URLSearchParams(location.search).has('flat')) {
  document.body.classList.add('flat')
}

createRoot(document.getElementById('root')).render(<App />)
