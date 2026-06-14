import { useEffect, useState } from 'react'

// Streams text into view one character at a time (visual streaming of the
// final answer). The API returns the answer whole; this animates its reveal.
export function Typewriter({ text, speed = 12 }: { text: string; speed?: number }) {
  const [shown, setShown] = useState('')
  useEffect(() => {
    setShown('')
    let i = 0
    const id = setInterval(() => {
      i += 1
      setShown(text.slice(0, i))
      if (i >= text.length) clearInterval(id)
    }, speed)
    return () => clearInterval(id)
  }, [text, speed])
  return <span data-testid="typewriter">{shown}</span>
}
