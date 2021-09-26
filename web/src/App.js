import React, { useEffect, useMemo, useState, useRef } from 'react'
import { hot } from 'react-hot-loader'
import "./App.css"

const API_URL = 'http://localhost:5000'

const App = () => {
  const [question, setQuestion] = useState('')
  const questionInput = useRef()
  const [loadingAnswer, setLoadingAnswer] = useState(false)
  const [answer, setAnswer] = useState()
  const [context, setContext] = useState()
  const [wikiUrl, setWikiUrl] = useState()

  const [page, setPage] = useState('')

  const contextBeforeAnswer = useMemo(() => {
    if (answer && context) {
      const startIndex = context.indexOf(answer)
      return context.substring(0, startIndex)
    }
  }, [answer, context])

  const contextAfterAnswer = useMemo(() => {
    if (answer && context) {
      const startIndex = context.indexOf(answer)
      const endIndex = startIndex + answer.length
      return context.substring(endIndex)
    }
  }, [answer, context])

  useEffect(() => {
    if (loadingAnswer) {
      fetch(`${API_URL}/?page=${page}&question=${question}`)
        .then(response => response.json())
        .then(data => {
          setAnswer(data.answer)
          setContext(data.context)
          setWikiUrl(data.url)
          setLoadingAnswer(false)
        })
    }
  }, [loadingAnswer, page, question])

  useEffect(() => {
    if (page.length > 0) {
      localStorage.setItem('page', page)
    }
  }, [page])

  useEffect(() => {
    setPage(localStorage.getItem('page'))
  }, [])

  useEffect(() => {
    if (questionInput) {
      questionInput.current.focus()
    }
  }, [questionInput])

  return <div className="App">
    <h2>Hey,</h2>
    <input type="text" 
      id="questionInput"
      ref={questionInput}
      placeholder="What's your question?"
      value={question} 
      onChange={e => setQuestion(e.target.value)}
      onKeyDown={e => {
        if (e.key === 'Enter') {
          setLoadingAnswer(true)
        }
      }}
    />
    {loadingAnswer && <p className="loadingState">Thinking...</p>}
    {answer && (
      <div className="answerContainer">
        <p>From the Wiki page on <a href={wikiUrl}>{page}</a>:</p>
        <p>"{contextBeforeAnswer}<b>{answer}</b>{contextAfterAnswer}"</p>
      </div>
    )}
    <div className="pageSelection">
      <h4 className="pageSelectionLabel">Page:</h4>
      <input type="text" id="pageInput" value={page} onChange={e => setPage(e.target.value)}/>
    </div>
  </div>
}

export default hot(module)(App)