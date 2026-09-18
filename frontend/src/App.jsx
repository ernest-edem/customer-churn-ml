import { BrowserRouter, Route, Routes, useNavigate } from 'react-router-dom'

import AssessmentPage from './pages/AssessmentPage'
import LandingPage from './pages/LandingPage'
import ModelPage from './pages/ModelPage'
import ResultsPage from './pages/ResultsPage'

function AppRoutes() {
  const navigate = useNavigate()

  return (
    <Routes>
      <Route
        path="/"
        element={
          <LandingPage
            onStartAssessment={() => navigate('/assessment')}
          />
        }
      />

      <Route
        path="/assessment"
        element={<AssessmentPage />}
      />

      <Route
        path="/results"
        element={<ResultsPage />}
      />

      <Route
        path="/model"
        element={<ModelPage />}
      />
    </Routes>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  )
}

export default App